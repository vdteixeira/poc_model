"""Pluga o scanner treinado de volta no deep agent (padrão 6).

O ART serve o modelo treinado numa API compatível com OpenAI. O deep agent
continua orquestrado pelo Claude, mas o subagente `scanner` — o mais
chamado nos padrões fan-out/loop — passa a usar SEU modelo treinado, por
uma fração do custo de um modelo de fronteira.

A única mudança em relação a 06_loop_until_done.py é o campo `model` do
subagente `scanner`, apontado para o endpoint local do ART via
ChatOpenAI (LangChain).

Pré-requisitos:
    - train_scanner.py já rodou (existe um checkpoint LoRA)
    - o backend do ART está servindo o modelo (veja SERVE_URL abaixo)
"""

import asyncio

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langchain_quickjs import CodeInterpreterMiddleware

# Endpoint OpenAI-compat exposto pelo backend do ART para o modelo treinado.
# Ajuste host/porta conforme sua configuração de serving (LocalBackend expõe
# um servidor vLLM). O nome do modelo é o `name` usado no TrainableModel.
SERVE_BASE_URL = "http://localhost:8000/v1"
TRAINED_MODEL_NAME = "scanner-rl-001"

trained_scanner_model = ChatOpenAI(
    model=TRAINED_MODEL_NAME,
    base_url=SERVE_BASE_URL,
    api_key="not-needed-for-local",  # backend local não exige chave real
    temperature=0.0,
)

agent = create_deep_agent(
    # Orquestrador continua sendo um modelo de fronteira.
    model="anthropic:claude-opus-4-8",
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
            # <<< A troca: subagente usa o modelo treinado, não o Claude.
            "model": trained_scanner_model,
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
                "src/utils. Trabalhe em passadas, deduplicando entre elas, "
                "com limite de 3 passadas. Apresente as descobertas "
                "consolidadas ao final."
            ),
        }]
    })
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
