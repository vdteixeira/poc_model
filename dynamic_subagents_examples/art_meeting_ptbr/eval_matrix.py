"""Eval matricial — compara o treino entre TAMANHOS de modelo e ESTÁGIOS.

A proposta de avaliação:

    Dimensão 1 (tamanho):  Qwen3 4B | 8B | 14B | Qwen3.6-27B ...
    Dimensão 2 (estágio):  stock | sft | sft+rl

    Métricas por célula (média nos cenários de validação, 3 repetições):
      correctness   — juiz LLM compara a resposta com `expected` (0..1)
      is_pt         — resposta em português (langdetect)
      answered      — chamou return_final_answer dentro de MAX_TURNS
      no_hallucin   — no cenário de controle "sem decisões", respondeu
                      "nenhuma" em vez de inventar tarefas (0..1)
      avg_turns     — turnos usados (eficiência de busca)
      avg_out_toks  — tokens de saída (proxy de custo de inferência)

    Regra de decisão: escolha o MENOR tamanho cujo estágio final passe a
    barra de qualidade (ex.: correctness >= 0.9 e is_pt == 1.0). É isso que
    define o subagente mais barato viável — não o maior score absoluto.

Design: o eval é DESACOPLADO do ART. Qualquer endpoint compatível com a API
OpenAI serve — checkpoints servidos pelo backend do ART, modelos stock via
vLLM/Ollama, ou APIs públicas. Cada candidato é só (nome, base_url, api_key,
model_id). Assim a mesma matriz avalia stock, SFT e SFT+RL sem tocar no
código de treino.

Uso:
    export OPENAI_API_KEY=...          # juiz de correção
    python eval_matrix.py              # edita CANDIDATES abaixo antes
"""

import asyncio
import json
from dataclasses import dataclass

from openai import AsyncOpenAI

try:
    from langdetect import detect
except ImportError:
    detect = None

from dataset_ptbr import MeetingScenario, val_scenarios
from train_meeting_agent import (
    SYSTEM_PROMPT,
    TOOLS,
    _read,
    _search,
    judge_correctness,
)

MAX_TURNS = 6
REPEATS = 3          # repetições por cenário (mediana reduz variância)
CONTROL_ID = "empty-decisions"  # cenário que mede alucinação de tarefas


@dataclass
class Candidate:
    """Um modelo a avaliar — qualquer endpoint OpenAI-compat."""
    name: str        # rótulo na matriz, ex. "27B/sft+rl"
    model_id: str    # id do modelo no endpoint
    base_url: str
    api_key: str = "not-needed"


# EDITE AQUI: a matriz tamanho × estágio que você quer comparar.
# Exemplos de linhas típicas:
#   stock via vLLM local, checkpoint SFT e checkpoint SFT+RL servidos pelo ART.
CANDIDATES: list[Candidate] = [
    Candidate("4B/stock", "Qwen/Qwen3-4B", "http://localhost:8001/v1"),
    Candidate("14B/stock", "Qwen/Qwen3-14B", "http://localhost:8002/v1"),
    Candidate("27B/stock", "Qwen/Qwen3.6-27B", "http://localhost:8003/v1"),
    # Candidate("27B/sft", "meeting-sft-001", "http://localhost:8000/v1"),
    # Candidate("27B/sft+rl", "meeting-agent-ptbr-001", "http://localhost:8000/v1"),
]


async def run_agent_once(
    client: AsyncOpenAI, model_id: str, scenario: MeetingScenario
) -> dict:
    """Mesmo loop de agente do treino, mas independente do ART."""
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Reunião: {scenario.title} ({scenario.date}). "
                f"A transcrição tem {len(scenario.segments)} segmentos.\n\n"
                f"Pergunta: {scenario.question}"
            ),
        },
    ]
    final_answer, turns, out_tokens = None, 0, 0

    for _ in range(MAX_TURNS):
        resp = await client.chat.completions.create(
            model=model_id, messages=messages, tools=TOOLS,
            max_tokens=1024, temperature=0.0,
        )
        turns += 1
        if resp.usage:
            out_tokens += resp.usage.completion_tokens or 0
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            break
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if tc.function.name == "search_transcript":
                result = _search(scenario, args.get("keywords", []))
            elif tc.function.name == "read_segment":
                result = _read(scenario, int(args.get("segment_idx", -1)))
            elif tc.function.name == "return_final_answer":
                final_answer = args.get("answer", "")
                result = json.dumps({"ok": True}, ensure_ascii=False)
            else:
                result = json.dumps({"error": "?"}, ensure_ascii=False)
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result}
            )
        if final_answer is not None:
            break

    return {"answer": final_answer, "turns": turns, "out_tokens": out_tokens}


async def eval_candidate(cand: Candidate) -> dict:
    client = AsyncOpenAI(base_url=cand.base_url, api_key=cand.api_key)
    scenarios = val_scenarios()

    runs = await asyncio.gather(*(
        run_agent_once(client, cand.model_id, s)
        for s in scenarios
        for _ in range(REPEATS)
    ))
    # Agrupa por cenário (REPEATS execuções consecutivas por cenário).
    per_scenario = [
        (s, runs[i * REPEATS:(i + 1) * REPEATS])
        for i, s in enumerate(scenarios)
    ]

    correctness, is_pt, answered, no_hallucin = [], [], [], []
    all_turns, all_toks = [], []

    for scenario, reps in per_scenario:
        for r in reps:
            ans = r["answer"] or ""
            answered.append(1.0 if r["answer"] is not None else 0.0)
            all_turns.append(r["turns"])
            all_toks.append(r["out_tokens"])
            if detect and ans:
                try:
                    is_pt.append(1.0 if detect(ans) == "pt" else 0.0)
                except Exception:
                    is_pt.append(0.0)
            score = await judge_correctness(scenario, ans)
            correctness.append(score)
            if scenario.id == CONTROL_ID:
                no_hallucin.append(score)  # correto == disse "nenhuma tarefa"

    avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return {
        "name": cand.name,
        "correctness": avg(correctness),
        "is_pt": avg(is_pt),
        "answered": avg(answered),
        "no_hallucin": avg(no_hallucin),
        "avg_turns": avg(all_turns),
        "avg_out_toks": avg(all_toks),
    }


async def main() -> None:
    rows = []
    for cand in CANDIDATES:  # sequencial p/ não misturar carga entre endpoints
        print(f"Avaliando {cand.name}...")
        rows.append(await eval_candidate(cand))

    header = f"{'modelo':<16}{'correct':>9}{'is_pt':>7}{'answ':>6}{'no_hall':>9}{'turns':>7}{'out_tok':>9}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:<16}{r['correctness']:>9.2f}{r['is_pt']:>7.2f}"
            f"{r['answered']:>6.2f}{r['no_hallucin']:>9.2f}"
            f"{r['avg_turns']:>7.1f}{r['avg_out_toks']:>9.0f}"
        )
    print(
        "\nRegra de decisão: menor tamanho com correctness>=0.90, is_pt=1.00 "
        "e no_hallucin=1.00 é o subagente mais barato viável."
    )


if __name__ == "__main__":
    asyncio.run(main())
