"""Utilitários compartilhados pelos 6 exemplos de Dynamic Subagents.

Deep Agents é construído sobre LangGraph — `create_deep_agent()` retorna um
grafo LangGraph compilado, então `ainvoke`/`astream` funcionam como em
qualquer grafo.

Requisitos:
    Python 3.11+
    pip install deepagents "langchain-quickjs>=0.1.0" "langchain[anthropic]"
    export ANTHROPIC_API_KEY=...
"""

MODEL = "anthropic:claude-opus-4-8"


def print_result(result: dict) -> None:
    """Imprime a resposta final do agente."""
    final_message = result["messages"][-1]
    print("\n" + "=" * 70)
    print("RESPOSTA FINAL DO AGENTE")
    print("=" * 70)
    print(final_message.content if isinstance(final_message.content, str)
          else final_message.content)
