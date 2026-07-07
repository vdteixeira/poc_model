"""Treina o subagente `scanner` (do padrão 6, loop-until-done) com RL.

Este é o ciclo completo do ART:
    1. rollout()  — roda o agente-scanner num cenário e devolve a trajetória
    2. gather_trajectory_groups() — N tentativas por cenário (o "group" do GRPO)
    3. ruler_score_group() — juiz LLM ranqueia as tentativas SEM reward manual
    4. model.train() — GRPO atualiza os pesos LoRA reforçando as melhores

Por que o `scanner` é o melhor primeiro alvo:
    - tarefa estreita e repetitiva (ler arquivo -> listar vulns em JSON)
    - saída avaliável -> RULER pontua de graça
    - nos padrões fan-out/loop ele é chamado dezenas de vezes por execução,
      então trocar um modelo de fronteira por um Qwen-14B treinado economiza
      muito custo/latência

Requisitos:
    GPU (>= 24GB p/ 14B com LoRA; use um 7B em placas menores)
    pip install "openpipe-art[backend]"
    export OPENAI_API_KEY=...   # usado só pelo juiz RULER (o3/o4-mini)

Rode:
    python train_scanner.py
"""

import asyncio
import json

import art
from art.local.backend import LocalBackend
from art.rewards import ruler_score_group

from scenarios import SCENARIOS, ScanScenario

# ---------------------------------------------------------------------------
# System prompt: EXATAMENTE o mesmo do subagente `scanner` em
# 06_loop_until_done.py. Treinamos o modelo na tarefa que ele fará em produção.
# ---------------------------------------------------------------------------
SCANNER_SYSTEM_PROMPT = (
    "Você é um scanner de segurança. Analise o arquivo indicado em busca de "
    "vulnerabilidades. Responda APENAS com um objeto JSON no formato:\n"
    '{"findings": [{"type": "...", "line": <int>, "severity": '
    '"critica|alta|media|baixa", "evidence": "..."}]}\n'
    "Se não houver vulnerabilidades reais, responda {\"findings\": []}. "
    "Não invente problemas: falsos positivos são piores que omissões."
)

# Juiz do RULER. Qualquer modelo forte compatível com a API OpenAI serve.
# (o guia da Unsloth usa o3; o4-mini é mais barato para iterar.)
RULER_JUDGE_MODEL = "openai/o4-mini"

# Base Unsloth/ART-compatível. Qwen3.6 é a geração atual (abr/2026) e é
# suportada pelo ART e pelo Unsloth (GRPO usa a inferência do Unsloth, não vLLM).
#   - "Qwen/Qwen3.6-27B"      -> denso, "sweet spot" p/ dev local (padrão aqui)
#   - "Qwen/Qwen3.6-35B-A3B"  -> MoE com ~3B ativos: inferência barata p/ o tamanho,
#                                ótimo p/ subagentes de volume; GRPO com MoE é um
#                                pouco mais sensível — valide antes de comprometer
# GPU menor? Use uma variante Qwen3.6 de menor porte suportada pelo ART.
BASE_MODEL = "Qwen/Qwen3.6-27B"


async def rollout(model: art.Model, scenario: ScanScenario) -> art.Trajectory:
    """Uma execução do scanner sobre um cenário -> uma trajetória."""
    client = model.openai_client()

    trajectory = art.Trajectory(
        messages_and_choices=[
            {"role": "system", "content": SCANNER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Arquivo: {scenario.filename}\n\n"
                    f"```python\n{scenario.code}\n```"
                ),
            },
        ],
        # `metadata` fica visível para o juiz RULER como contexto opcional.
        metadata={"scenario_id": scenario.id},
    )

    completion = await client.chat.completions.create(
        messages=trajectory.messages(),
        model=model.name,
        max_tokens=1024,
    )
    choice = completion.choices[0]
    trajectory.messages_and_choices.append(choice)

    # Reward opcional "de sanidade": penaliza JSON malformado. O RULER faz o
    # trabalho pesado de qualidade; isto só dá um empurrão barato e objetivo.
    try:
        json.loads(choice.message.content)
        trajectory.metrics["valid_json"] = 1.0
    except (json.JSONDecodeError, TypeError):
        trajectory.metrics["valid_json"] = 0.0

    return trajectory


async def main() -> None:
    backend = LocalBackend()

    model = art.TrainableModel(
        name="scanner-rl-001",
        project="security-scanner",  # constante entre runs -> agrupa métricas
        base_model=BASE_MODEL,
    )
    await model.register(backend)

    NUM_STEPS = 20
    ROLLOUTS_PER_GROUP = 8  # 8 tentativas por cenário -> o "grupo" do GRPO

    for step in range(await model.get_step(), NUM_STEPS):
        print(f"\n=== Step {step} ===")

        # Um grupo por cenário; cada grupo tem ROLLOUTS_PER_GROUP tentativas.
        groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, scenario) for _ in range(ROLLOUTS_PER_GROUP)
                )
                for scenario in SCENARIOS
            ),
            pbar_desc="gather",
            # RULER pontua cada grupo assim que ele termina de ser coletado.
            after_each=lambda group: ruler_score_group(
                group,
                RULER_JUDGE_MODEL,
                swallow_exceptions=True,  # falha do juiz -> descarta o grupo
            ),
        )

        await model.train(
            groups,
            config=art.TrainConfig(learning_rate=1e-5),
        )
        print(f"Step {step} concluído — pesos LoRA atualizados.")

    print("\nTreino finalizado. O modelo é servido em uma API compatível")
    print("com OpenAI via model.openai_client() / backend.")


if __name__ == "__main__":
    asyncio.run(main())
