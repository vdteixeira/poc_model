"""Padrão 5 — Tournament (Torneio).

Foco: escolha quando "melhor" é SUBJETIVO e difícil de pontuar direto.
É o primo competitivo do generate-and-filter: em vez de avaliar tudo de
uma vez, as variantes competem em pares; um subagente juiz compara A vs B,
os vencedores avançam, e as rodadas continuam até restar um.

Como funciona: geração dos candidatos (fan-out para "writer", cada um em
seu arquivo), depois rodada 1 com chamadas ao juiz em pares, rodada 2
(final) com os vencedores, até sobrar o campeão.

Gatilho no prompt: "compare-as frente a frente em um torneio, avançando
as vencedoras até que uma se destaque".
"""

import asyncio

from deepagents import create_deep_agent
from langchain_quickjs import CodeInterpreterMiddleware

from common import MODEL, print_result

agent = create_deep_agent(
    model=MODEL,
    subagents=[
        {
            "name": "writer",
            "description": "Reescreve código com uma prioridade específica",
            "system_prompt": (
                "Você é um engenheiro de software sênior. Reescreva o código "
                "indicado otimizando para a prioridade que receber "
                "(legibilidade, robustez, desempenho ou diff mínimo). "
                "Mantenha o comportamento externo idêntico."
            ),
        },
        {
            "name": "judge",
            "description": "Compara duas reescritas frente a frente e declara a vencedora",
            "system_prompt": (
                "Você é um juiz de código. Receberá duas versões candidatas "
                "(e o original como referência). Compare-as frente a frente "
                "quanto a correção, clareza e manutenibilidade, declare a "
                "vencedora e justifique em poucas frases."
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
                "O handler de criação de pedidos em src/orders/create.py está "
                "confuso e quero a melhor reescrita possível. Execute um "
                "workflow que produza quatro reescritas candidatas com "
                "prioridades diferentes — legibilidade, robustez, desempenho "
                "e diff mínimo — salvando cada uma em seu próprio arquivo no "
                "diretório candidates/. Depois compare-as frente a frente em "
                "um torneio, avançando as vencedoras de cada rodada até que "
                "uma se destaque como campeã. Mostre a chave do torneio no "
                "resultado final."
            ),
        }]
    })
    print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
