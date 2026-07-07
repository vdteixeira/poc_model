"""Padrão 3 — Adversarial Verification (Verificação Adversária).

Foco: PRECISÃO. Quando uma resposta errada custa caro e é preferível
deixar algo passar do que reportar um falso positivo.

Como funciona em duas passadas:
  1ª passada (fan-out): subagentes "auditor" geram descobertas candidatas.
  2ª passada (verify):  cada candidata é enviada a um subagente
     "verifier" independente; só entram no relatório as confirmadas.

Detalhe importante do vídeo: variáveis criadas numa chamada `eval`
persistem entre chamadas — o agente cria `allFindings` na 1ª passada e a
reutiliza no código da 2ª, o que viabiliza workflows multi-etapa.

Gatilho no prompt: pedir confiança explicitamente — "verifique cada uma
independentemente", "só problemas reais e confirmados, nada de talvez".
"""

import asyncio

from deepagents import create_deep_agent
from langchain_quickjs import CodeInterpreterMiddleware

from common import MODEL, print_result

agent = create_deep_agent(
    model=MODEL,
    subagents=[
        {
            "name": "auditor",
            "description": "Audita código em busca de vulnerabilidades (gera candidatas)",
            "system_prompt": (
                "Você é um auditor de segurança. Analise o arquivo e liste "
                "TODAS as vulnerabilidades potenciais com tipo, linha e "
                "evidência. Nesta fase, cobertura importa mais que precisão."
            ),
        },
        {
            "name": "verifier",
            "description": "Verifica de forma independente se uma descoberta é real",
            "system_prompt": (
                "Você é um verificador cético. Receberá uma descoberta de "
                "segurança e o código relevante. Tente REFUTÁ-LA. Confirme "
                "apenas se a evidência for concreta; caso contrário rejeite "
                "e explique o motivo."
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
                "Execute um workflow de auditoria de segurança em src/utils. "
                "Analise todos os arquivos em busca de vulnerabilidades. "
                "IMPORTANTE: verifique cada descoberta de forma independente "
                "antes de incluí-la no relatório. Quero apenas problemas "
                "reais e confirmados — nada de 'talvez' ou riscos teóricos. "
                "Ao final, um relatório breve com as descobertas confirmadas "
                "e quantas foram rejeitadas na verificação."
            ),
        }]
    })
    print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
