"""Padrão 4 — Generate and Filter (Gerar e Filtrar).

Foco: QUALIDADE. Vários subagentes tentam resolver o MESMO problema de
forma independente; depois um avaliador escolhe a melhor solução.

Como funciona: o agente dispara N subagentes "designer" em paralelo (um
por abordagem), cada um grava seu design num arquivo próprio (quando se
grava em arquivo, não é preciso responseSchema). Em seguida um subagente
avaliador pontua os candidatos e recomenda o vencedor.

Gatilho no prompt: "explore N abordagens diferentes" + "recomende a mais
forte com justificativa" (as abordagens criam diversidade; a escolha é o
filtro).
"""

import asyncio

from deepagents import create_deep_agent
from langchain_quickjs import CodeInterpreterMiddleware

from common import MODEL, print_result

agent = create_deep_agent(
    model=MODEL,
    subagents=[
        {
            "name": "system-designer",
            "description": "Produz um esboço de design de sistema conciso com trade-offs",
            "system_prompt": (
                "Você é um arquiteto de sistemas. Produza um esboço de design "
                "conciso para o algoritmo solicitado: visão geral, estrutura "
                "de dados, complexidade, trade-offs e considerações de escala."
            ),
        },
        {
            "name": "design-evaluator",
            "description": "Avalia designs candidatos e recomenda o mais forte",
            "system_prompt": (
                "Você é um avaliador de designs. Compare os candidatos usando "
                "os critérios fornecidos, monte uma matriz de pontuação e "
                "recomende o mais forte com justificativa clara."
            ),
        },
    ],
    middleware=[CodeInterpreterMiddleware()],
)


async def main() -> None:
    result = await agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": (
                "Nosso rate limiter atual não é o design certo para a nossa "
                "escala. Execute um workflow que explore quatro redesigns "
                "diferentes: token bucket, sliding window counter, leaky "
                "bucket e GCRA. Escreva cada um como um esboço de design "
                "conciso em seu próprio arquivo no diretório candidates/. "
                "Depois avalie-os por precisão, custo de memória, facilidade "
                "de implementação distribuída e comportamento em burst, e "
                "recomende o mais forte com uma justificativa clara."
            ),
        }]
    })
    print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
