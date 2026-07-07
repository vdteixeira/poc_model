"""Padrão 6 — Loop Until Done (Loop Até a Conclusão).

Foco: EXAUSTIVIDADE. O agente executa uma passada, deduplica contra o que
já encontrou, executa outra passada, e só para quando uma passada completa
não encontra nada de novo (ou atinge o limite de passadas).

Como funciona: a cada rodada o agente constrói a lista consolidada das
descobertas anteriores e a injeta como contexto dos scanners da rodada
seguinte ("procure apenas problemas que as passadas anteriores NÃO
encontraram"). A persistência de variáveis entre chamadas `eval` é o que
torna a deduplicação possível.

Gatilho no prompt: definir completude como condição de parada — "todo",
"não pare até que uma passada não encontre nada de novo".
"""

import asyncio

from deepagents import create_deep_agent
from langchain_quickjs import CodeInterpreterMiddleware

from common import MODEL, print_result

agent = create_deep_agent(
    model=MODEL,
    subagents=[
        {
            "name": "scanner",
            "description": "Varre código em busca de problemas de segurança, evitando duplicatas",
            "system_prompt": (
                "Você é um scanner de segurança. Analise o arquivo indicado "
                "em busca de vulnerabilidades. Se receber uma lista de "
                "problemas já encontrados em passadas anteriores, reporte "
                "APENAS problemas novos que não estejam nessa lista."
            ),
        }
    ],
    middleware=[CodeInterpreterMiddleware()],
)


async def main() -> None:
    result = await agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": (
                "Execute um workflow de revisão de segurança completa em "
                "src/utils. Trabalhe em passadas: cada passada deve procurar "
                "apenas problemas que as passadas anteriores não encontraram "
                "(passe a lista consolidada do que já foi achado como "
                "contexto para os scanners). Continue enquanto novos "
                "problemas surgirem, com limite de 3 passadas. Ao final, "
                "apresente as descobertas consolidadas de todas as passadas."
            ),
        }]
    })
    print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
