"""Padrão 1 — Classify and Act (Classificar e Agir).

Foco: ROTEAMENTO. Entradas misturadas, cada tipo precisa de tratamento
diferente por um subagente especializado.

Como funciona: o agente escreve código no interpretador, classifica cada
item e despacha via `task()` para o subagente certo (bug-fixer,
feature-analyst ou support-agent), tudo em paralelo.

Gatilho no prompt: "execute um workflow" + instruções de manuseio por tipo.
"""

import asyncio

from deepagents import create_deep_agent
from langchain_quickjs import CodeInterpreterMiddleware

from common import MODEL, print_result

agent = create_deep_agent(
    model=MODEL,
    subagents=[
        {
            "name": "bug-fixer",
            "description": "Investiga relatos de bug e fornece passos de reprodução",
            "system_prompt": (
                "Você é um especialista em triagem de bugs. Investigue cada "
                "relato e forneça passos de reprodução claros e uma hipótese "
                "de causa raiz."
            ),
        },
        {
            "name": "feature-analyst",
            "description": "Avalia pedidos de funcionalidade (viabilidade e esforço)",
            "system_prompt": (
                "Você é um analista de produto. Avalie cada pedido de "
                "funcionalidade quanto a viabilidade técnica, esforço estimado "
                "e impacto potencial."
            ),
        },
        {
            "name": "support-agent",
            "description": "Responde perguntas de usuários com base na documentação",
            "system_prompt": (
                "Você é um especialista de suporte. Responda as perguntas dos "
                "usuários de forma clara e objetiva."
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
                "Execute um workflow para triar os tickets do arquivo "
                "tickets.jsonl. Classifique cada um como bug, feature request "
                "ou pergunta. Para bugs, quero passos de reprodução; para "
                "features, uma avaliação de viabilidade; para perguntas, uma "
                "resposta direta. Ao final, me dê um resumo da triagem."
            ),
        }]
    })
    print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
