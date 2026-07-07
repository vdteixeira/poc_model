"""Agente de reunião pt-BR treinado com RL — espelho do notebook ART-E.

Mapeamento direto ART-E -> Tag AI:
    ART-E                         Este exemplo
    -----------------------       ----------------------------------
    caixa de e-mail               transcrição de reunião (segmentos)
    search_inbox(keywords)        search_transcript(keywords)
    read_email(message_id)        read_segment(segment_idx)
    return_final_answer(...)      return_final_answer(...)
    dataset de perguntas Enron    dataset_ptbr.py (sintético, pt-BR)
    RULER (juiz relativo)         RULER (idêntico) — sem rótulos
    juiz de correção (validação)  judge_correctness (validação, opcional)

Por que isto resolve o "não tenho dados de português": o RULER pontua as
tentativas comparando-as entre si; você só precisa de CENÁRIOS (transcrição
+ pergunta), que geramos à mão em dataset_ptbr.py. Nenhum rótulo de treino.

Requisitos:
    - macOS/Apple Silicon ou máquina sem CUDA: use ServerlessBackend
      pip install openpipe-art openai tenacity langdetect
      export WANDB_API_KEY=...
    - Linux com GPU NVIDIA/CUDA: pode usar LocalBackend
      pip install "openpipe-art[backend]" openai tenacity langdetect
    export OPENAI_API_KEY=...   # usado só pelo juiz RULER e pelo juiz de correção

Rode:
    python train_meeting_agent.py   # NUM_STEPS=5 por padrão -> smoke test do ciclo

Backend:
    ART_BACKEND=auto       # padrão: local se Linux+CUDA, senão serverless
    ART_BACKEND=serverless # força backend gerenciado; não precisa vLLM local
    ART_BACKEND=local      # força backend local; exige Linux+CUDA+openpipe-art[backend]
"""

import asyncio
import json
import os
import platform
import re

import art
from art.rewards import ruler_score_group
from art.trajectories import History
from openai import AsyncOpenAI

try:
    # Métrica de consistência de idioma (não é reward — só monitora drift p/ inglês).
    from langdetect import detect  # pip install langdetect
except ImportError:  # degrada com elegância se langdetect não estiver instalado
    detect = None

from dataset_ptbr import MeetingScenario, train_scenarios, val_scenarios

# ---------------------------------------------------------------------------
# Config — espelha o `training_config` do ART-E, em escala menor.
# ---------------------------------------------------------------------------
# Qwen3.6-27B: DENSO (não MoE), 27B, pós-treinado, 262K de contexto nativo.
# Vantagem de ser denso vs o 35B-A3B MoE: QLoRA 4-bit FUNCIONA (a limitação
# de QLoRA valia só para MoE). Estimativas de VRAM (aritmética 27B×bytes/param,
# NÃO valores medidos numa doc do 3.6 — confirme no Unsloth quando publicarem):
#   - QLoRA 4-bit: ~14GB de pesos -> treino provavelmente em GPU de 24-48GB
#   - bf16 LoRA:   ~54GB de pesos -> A100/H100 80GB
# É a melhor escolha da linha 3.6 para servir de subagente denso e barato.
BASE_MODEL = "Qwen/Qwen3.6-27B"

RULER_JUDGE_MODEL = "openai/o4-mini"   # juiz relativo do RULER (reward do treino)
CORRECTNESS_JUDGE = "gpt-5.4"          # juiz de correção (só métrica de validação)

# ATENÇÃO: com o dataset pequeno (5 cenários de treino), rode poucos steps
# como SMOKE TEST — o modelo memoriza as reuniões em poucos passos e o sinal
# do RULER vira ruído. Para treino real, gere 100+ cenários sintéticos.
NUM_STEPS = 5                          # smoke test; suba só com dataset maior
ROLLOUTS_PER_GROUP = 4                 # N tentativas por cenário (o "grupo" do GRPO)
MAX_TURNS = 6                          # teto de turnos do agente (como no ART-E)
LEARNING_RATE = 1.2e-5
VALIDATE_EVERY = 5

# Cliente separado só para os juízes (API OpenAI real). O modelo em treino
# usa model.openai_client() (endpoint local servido pelo backend do ART).
judge_client = AsyncOpenAI()


def _cuda_available() -> bool:
    """Retorna True só quando há CUDA utilizável neste processo.

    Importante: Apple Silicon tem GPU, mas não CUDA; `LocalBackend` do ART
    precisa do runtime vLLM/CUDA, então macOS/MPS não serve para esse backend.
    """
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def make_backend():
    """Escolhe um backend que não tente importar/instalar vLLM em macOS.

    `LocalBackend` inicia um runtime vLLM local para inferência/treino. Esse
    caminho é apropriado apenas em Linux com GPU NVIDIA/CUDA e dependências do
    extra `openpipe-art[backend]`. Em macOS/Apple Silicon, o fix correto para
    `ModuleNotFoundError: No module named 'vllm'` é usar `ServerlessBackend`.
    """
    requested = os.getenv("ART_BACKEND", "auto").strip().lower()
    if requested not in {"auto", "local", "serverless"}:
        raise ValueError(
            "ART_BACKEND deve ser 'auto', 'local' ou 'serverless' "
            f"(recebido: {requested!r})."
        )

    local_supported = platform.system() == "Linux" and _cuda_available()

    if requested == "local" and not local_supported:
        raise RuntimeError(
            "ART_BACKEND=local exige Linux com GPU NVIDIA/CUDA. "
            "Neste ambiente, use ART_BACKEND=serverless para evitar o erro "
            "`ModuleNotFoundError: No module named 'vllm'`."
        )

    if requested == "serverless" or (requested == "auto" and not local_supported):
        if not os.getenv("WANDB_API_KEY"):
            raise RuntimeError(
                "ServerlessBackend requer WANDB_API_KEY. Defina "
                "`export WANDB_API_KEY=...` ou use ART_BACKEND=local em uma "
                "máquina Linux com GPU NVIDIA/CUDA."
            )
        from art.serverless.backend import ServerlessBackend

        print("Backend: ServerlessBackend (sem vLLM local).")
        return ServerlessBackend()

    from art.local.backend import LocalBackend

    print("Backend: LocalBackend (Linux+CUDA/vLLM local).")
    return LocalBackend()


# ---------------------------------------------------------------------------
# Ferramentas do agente (schema OpenAI). Executadas localmente no rollout.
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_transcript",
            "description": "Busca segmentos da transcrição que contenham as palavras-chave.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Palavras-chave a buscar nos segmentos.",
                    }
                },
                "required": ["keywords"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_segment",
            "description": "Lê o texto completo de um segmento pelo seu índice.",
            "parameters": {
                "type": "object",
                "properties": {
                    "segment_idx": {"type": "integer"},
                },
                "required": ["segment_idx"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "return_final_answer",
            "description": "Devolve a resposta final ao usuário e encerra a tarefa.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "segment_refs": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Índices dos segmentos que embasam a resposta.",
                    },
                },
                "required": ["answer"],
            },
        },
    },
]


def _search(scenario: MeetingScenario, keywords: list[str]) -> str:
    kws = [k.lower() for k in keywords]
    hits = [
        {"segment_idx": s.idx, "speaker": s.speaker, "snippet": s.text[:80]}
        for s in scenario.segments
        if any(k in s.text.lower() or k in s.speaker.lower() for k in kws)
    ]
    return json.dumps({"results": hits[:10]}, ensure_ascii=False)


def _read(scenario: MeetingScenario, idx: int) -> str:
    for s in scenario.segments:
        if s.idx == idx:
            return json.dumps(
                {"segment_idx": s.idx, "speaker": s.speaker, "text": s.text},
                ensure_ascii=False,
            )
    return json.dumps({"error": f"segmento {idx} não existe"}, ensure_ascii=False)


def _strip_think(text: str) -> str:
    """Remove blocos <think>...</think> — imita o chat template do Qwen 3.x,
    que descarta o thinking de turnos ANTERIORES na renderização."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def _choice_as_context(choice) -> dict:
    """Converte um `Choice` (turno treinável) em mensagem simples de contexto,
    exatamente como o template a renderizaria num turno posterior: assistant
    sem <think>, preservando os tool_calls."""
    msg = choice.message
    out: dict = {"role": "assistant", "content": _strip_think(msg.content or "")}
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return out


SYSTEM_PROMPT = (
    "Você é um assistente de reuniões em português do Brasil. Sua tarefa é "
    "responder à pergunta do usuário consultando a transcrição de uma reunião "
    "por meio das ferramentas disponíveis. Use search_transcript para "
    "localizar segmentos relevantes e read_segment para lê-los na íntegra. "
    "Responda SEMPRE em português, de forma objetiva e fiel à transcrição — "
    "não invente tarefas, decisões ou nomes que não estejam nos segmentos. "
    "Quando tiver a resposta, chame return_final_answer."
)


# ---------------------------------------------------------------------------
# rollout — uma execução do agente sobre um cenário -> uma trajetória.
# Estrutura idêntica à do ART-E: loop de tool-calling até MAX_TURNS.
# ---------------------------------------------------------------------------
async def rollout(model: art.Model, scenario: MeetingScenario) -> art.Trajectory:
    client = model.openai_client()

    system_msg = {"role": "system", "content": SYSTEM_PROMPT}
    user_msg = {
        "role": "user",
        "content": (
            f"Reunião: {scenario.title} ({scenario.date}). "
            f"A transcrição tem {len(scenario.segments)} segmentos.\n\n"
            f"Pergunta: {scenario.question}"
        ),
    }

    trajectory = art.Trajectory(
        messages_and_choices=[system_msg, user_msg],
        metadata={"scenario_id": scenario.id, "split": scenario.split},
    )

    final_answer: str | None = None

    for _turn in range(MAX_TURNS):
        completion = await client.chat.completions.create(
            model=model.name,
            messages=trajectory.messages(),
            tools=TOOLS,
            max_tokens=1024,
        )
        choice = completion.choices[0]
        trajectory.messages_and_choices.append(choice)

        tool_calls = choice.message.tool_calls or []
        if not tool_calls:
            break  # modelo respondeu sem chamar ferramenta -> encerra

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "search_transcript":
                result = _search(scenario, args.get("keywords", []))
            elif name == "read_segment":
                result = _read(scenario, int(args.get("segment_idx", -1)))
            elif name == "return_final_answer":
                final_answer = args.get("answer", "")
                result = json.dumps({"ok": True}, ensure_ascii=False)
            else:
                result = json.dumps({"error": "ferramenta desconhecida"}, ensure_ascii=False)

            trajectory.messages_and_choices.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result}
            )

        if final_answer is not None:
            break

    # --- Workaround multi-turn do Qwen 3.x (inclui 3.6) -----------------
    # O chat template do Qwen 3 remove os tokens <think> de turnos ANTERIORES,
    # o que corrompe o recálculo de logprobs em treino multi-turno. Padrão do
    # ART (doc "additional_histories"): cada turno do assistente é o turno
    # FINAL (um `Choice`, treinável) de exatamente uma history; os turnos
    # anteriores viram mensagens simples de contexto (dict, assistant sem
    # <think>) — idêntico ao que o template renderiza na inferência. Assim
    # cada turno é treinado exatamente uma vez, com o prompt fiel.
    msgs = trajectory.messages_and_choices
    assistant_positions = [i for i, m in enumerate(msgs) if not isinstance(m, dict)]

    if len(assistant_positions) > 1:
        first = assistant_positions[0]

        def _history_ending_at(pos: int) -> list:
            # Contexto (tudo antes de `pos`) com Choices achatados + o turno
            # final mantido como Choice (é ele que treina nesta history).
            context = [
                m if isinstance(m, dict) else _choice_as_context(m)
                for m in msgs[:pos]
            ]
            return context + [msgs[pos]]

        # History principal: contexto inicial + primeiro turno do assistente.
        trajectory.messages_and_choices = msgs[: first + 1]
        # Uma additional_history por turno subsequente.
        trajectory.additional_histories = [
            History(messages_and_choices=_history_ending_at(pos))
            for pos in assistant_positions[1:]
        ]

    # --- Métricas (NÃO são reward; o reward é 100% RULER) ---------------
    trajectory.metrics["answered"] = 1.0 if final_answer else 0.0
    # Consistência de idioma: o paper mostrou GRPO colapsar pt p/ inglês sem
    # sinal de idioma. Aqui monitoramos (langdetect) para pegar drift cedo.
    if detect and final_answer:
        try:
            trajectory.metrics["is_pt"] = 1.0 if detect(final_answer) == "pt" else 0.0
        except Exception:
            trajectory.metrics["is_pt"] = 0.0
    trajectory.metadata["final_answer"] = final_answer or ""

    return trajectory


# ---------------------------------------------------------------------------
# Juiz de correção — SÓ para validação (como no ART-E). Usa `expected`.
# Não entra no treino; o treino é 100% RULER.
# ---------------------------------------------------------------------------
async def judge_correctness(scenario: MeetingScenario, answer: str) -> float:
    if not scenario.expected:
        return 0.0
    prompt = (
        "Você avalia a resposta de um assistente de reuniões. Diga se a "
        "resposta captura corretamente os pontos da referência (mesmo com "
        "outras palavras). Responda só 'sim' ou 'não'.\n\n"
        f"Pergunta: {scenario.question}\n"
        f"Referência: {scenario.expected}\n"
        f"Resposta do agente: {answer}\n"
    )
    resp = await judge_client.chat.completions.create(
        model=CORRECTNESS_JUDGE,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8,
    )
    verdict = (resp.choices[0].message.content or "").strip().lower()
    return 1.0 if verdict.startswith("sim") else 0.0


async def validate(model: art.Model) -> None:
    scenarios = val_scenarios()
    trajs = await asyncio.gather(*(rollout(model, s) for s in scenarios))
    scores = await asyncio.gather(
        *(
            judge_correctness(s, t.metadata.get("final_answer", ""))
            for s, t in zip(scenarios, trajs)
        )
    )
    acc = sum(scores) / len(scores) if scores else 0.0
    pt_rate = (
        sum(t.metrics.get("is_pt", 0.0) for t in trajs) / len(trajs)
        if trajs and detect
        else float("nan")
    )
    print(
        f"  [validação] correção média: {acc:.2%} ({len(scores)} cenários) | "
        f"respostas em pt: {pt_rate:.0%}"
    )


# ---------------------------------------------------------------------------
# Loop de treino — mesma forma do ART-E: gather -> RULER -> train.
# ---------------------------------------------------------------------------
async def main() -> None:
    backend = make_backend()

    model = art.TrainableModel(
        name="meeting-agent-ptbr-001",
        project="tag-ai-meeting-agent",
        base_model=BASE_MODEL,
    )
    await model.register(backend)

    scenarios = train_scenarios()

    for step in range(await model.get_step(), NUM_STEPS):
        print(f"\n=== Step {step} ===")

        groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, scenario) for _ in range(ROLLOUTS_PER_GROUP)
                )
                for scenario in scenarios
            ),
            pbar_desc="gather",
            after_each=lambda group: ruler_score_group(
                group,
                RULER_JUDGE_MODEL,
                swallow_exceptions=True,
            ),
        )

        await model.train(groups, config=art.TrainConfig(learning_rate=LEARNING_RATE))
        print(f"Step {step} concluído — pesos LoRA atualizados.")

        if step % VALIDATE_EVERY == 0:
            await validate(model)

    print("\nTreino finalizado. Sirva o modelo pelo backend do ART (API")
    print("compatível com OpenAI) e pluge-o como subagente no deep agent.")


if __name__ == "__main__":
    asyncio.run(main())
