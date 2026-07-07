"""Padrão 2 — Fan-out and Synthesize (Expandir e Sintetizar).

Foco: COBERTURA. A mesma tarefa aplicada a N itens em paralelo, com os
resultados combinados numa única saída final.

Como funciona: o agente monta um array com os itens (ex.: arquivos de um
diretório), dispara um `task()` por item com `Promise.all`, e depois
achata/ordena os resultados numa síntese única.

Gatilho no prompt: palavras como "todos", "cada um", "não pule nenhum" +
pedir uma saída única combinada ("um relatório priorizado").

Exemplo de código que o PRÓPRIO AGENTE escreve dentro do interpretador:

    const paths = ["src/auth.ts", "src/routes/api.ts"];
    const reviews = await Promise.all(
      paths.map((path) => task({
        description: `Review ${path} for security issues`,
        subagentType: "reviewer",
        responseSchema: { ... },   // schema gerado dinamicamente
      })),
    );
"""

import asyncio

from deepagents import create_deep_agent
from langchain_quickjs import CodeInterpreterMiddleware

from common import MODEL, print_result

agent = create_deep_agent(
    model=MODEL,
    subagents=[
        {
            "name": "reviewer",
            "description": "Revisa código em busca de falhas de segurança, citando linhas e severidade",
            "system_prompt": (
                "Você é um revisor de código focado em segurança. Leia o "
                "arquivo com atenção e reporte qualquer problema de "
                "autenticação, autorização ou injeção, com número da linha e "
                "severidade (crítica/alta/média/baixa)."
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
                "Execute um workflow de revisão de segurança em src/utils. "
                "Inclua TODOS os arquivos do diretório, não pule nenhum. "
                "Depois me entregue um único relatório priorizado dos "
                "principais riscos, ordenado por severidade."
            ),
        }]
    })
    print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
