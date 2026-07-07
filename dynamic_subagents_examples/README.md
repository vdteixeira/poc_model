# Dynamic Subagents — 6 Padrões (Deep Agents / LangGraph)

Exemplos executáveis dos 6 padrões de orquestração apresentados no vídeo da
LangChain sobre **Dynamic Subagents** (https://www.youtube.com/watch?v=5AkdMangfNk).

O Deep Agents SDK é construído sobre **LangGraph** — `create_deep_agent()`
retorna um grafo LangGraph compilado. O `CodeInterpreterMiddleware` dá ao
agente uma ferramenta `eval` (sandbox QuickJS em memória) que expõe a global
`task()`, permitindo que o agente **escreva código para criar e coordenar
subagentes programaticamente** em vez de chamar a ferramenta de task um turno
por vez.

## Setup

```bash
# Python 3.11+
pip install deepagents "langchain-quickjs>=0.1.0" "langchain[anthropic]"
export ANTHROPIC_API_KEY=sk-ant-...
```

## Como o agente despacha subagentes (código que ELE escreve no interpretador)

```javascript
const review = await task({
  description: "Review src/auth/login.ts for auth issues. Cite line numbers.",
  subagentType: "reviewer",
  responseSchema: {          // opcional — retorna resultado já tipado
    type: "object",
    properties: { issues: { type: "array", items: { type: "object" } } },
  },
});
const critical = review.issues.filter(i => i.severity === "high");
```

Pontos-chave do vídeo:
- A palavra **"workflow"** no prompt é o sinal para o agente orquestrar via código.
- `responseSchema` devolve resultado tipado → permite loops e ramificações.
- Variáveis **persistem entre chamadas `eval`** → workflows multi-etapa.
- Alternativa ao schema: mandar cada subagente **gravar em arquivo**.
- Cada subagente roda em contexto isolado → o contexto do agente principal fica limpo.

## Os 6 padrões

| # | Arquivo | Padrão | Foco | Gatilho no prompt |
|---|---------|--------|------|-------------------|
| 1 | `01_classify_and_act.py` | Classify & Act | Roteamento | "descubra o que cada um é e trate do jeito certo" |
| 2 | `02_fanout_and_synthesize.py` | Fan-out & Synthesize | Cobertura | "todos", "não pule nenhum" + "um relatório único" |
| 3 | `03_adversarial_verification.py` | Adversarial Verification | Precisão | "verifique cada uma independentemente", "só confirmados" |
| 4 | `04_generate_and_filter.py` | Generate & Filter | Qualidade | "explore N abordagens" + "recomende a melhor" |
| 5 | `05_tournament.py` | Tournament | Escolha subjetiva | "compare frente a frente, avançando as vencedoras" |
| 6 | `06_loop_until_done.py` | Loop Until Done | Exaustividade | "em passadas", "não pare até nada de novo aparecer" |

## Executar

```bash
python 01_classify_and_act.py
```

Os exemplos 2, 3, 5 e 6 referenciam caminhos como `src/utils` — ajuste para
um diretório real do seu projeto, ou crie arquivos de exemplo. O exemplo 1
espera um `tickets.jsonl` (uma linha JSON por ticket, ex.:
`{"id": 1, "subject": "...", "body": "..."}`).

## Observabilidade

Para ver os traces como no vídeo (chamadas `eval`, fan-outs de `task()`,
schemas gerados), configure o LangSmith:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=...
```
