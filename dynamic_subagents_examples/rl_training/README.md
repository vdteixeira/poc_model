# Passo prático: treinar o subagente `scanner` com RL (ART + Unsloth)

Este diretório fecha o ciclo entre os **Dynamic Subagents** (Deep Agents /
LangGraph) e o guia da Unsloth de **Training AI Agents with RL**. Pegamos o
subagente mais chamado dos padrões de alto volume — o `scanner` do padrão 6
(`../06_loop_until_done.py`) — e o treinamos com GRPO + RULER para que um
modelo pequeno (Qwen-14B) faça a tarefa com a confiabilidade de um modelo de
fronteira, por uma fração do custo.

## Por que o `scanner` é o alvo certo

| Critério do RL | O scanner atende? |
|---|---|
| Tarefa estreita e repetitiva | ✅ ler arquivo → listar vulns em JSON |
| Resultado avaliável (juiz possível) | ✅ RULER compara tentativas sem reward manual |
| Chamado em escala | ✅ fan-out (padrão 2) e loop (padrão 6) o disparam dezenas de vezes/execução |
| Custo da API de fronteira incomoda | ✅ é exatamente o gargalo que o RL resolve |

## Os arquivos

| Arquivo | Papel |
|---|---|
| `scenarios.py` | Dataset de treino: trechos de código com vulnerabilidades plantadas + um arquivo limpo (ensina a NÃO alucinar). Troque pelos seus próprios trechos. |
| `train_scanner.py` | O ciclo completo do ART: `rollout` → `gather_trajectory_groups` → `ruler_score_group` → `model.train`. |
| `use_trained_scanner.py` | Pluga o modelo treinado de volta no deep agent (só muda o campo `model` do subagente). |

## O mapeamento conceitual (Dynamic Subagents ↔ ART)

- O **system prompt** de treino em `train_scanner.py` é o mesmo do subagente
  `scanner` em produção — treinamos na tarefa exata que ele fará.
- **RULER** é a mesma ideia dos seus padrões 3 (verificação adversária) e 5
  (torneio): um LLM julgando saídas relativas. A diferença é que aqui o
  julgamento vira **sinal de gradiente** que atualiza os pesos.
- O ART serve o modelo numa **API compatível com OpenAI**, e o
  `create_deep_agent` aceita qualquer modelo LangChain — por isso a troca em
  produção é uma linha (`"model": trained_scanner_model`).

## Como rodar

```bash
# 1. Instalar (precisa de GPU NVIDIA/CUDA; não roda em Apple Silicon)
#    Modelo base: Qwen/Qwen3.6-27B (geração atual, suportado por ART + Unsloth)
pip install "openpipe-art[backend]" deepagents "langchain-quickjs>=0.1.0" \
    langchain-openai "langchain[anthropic]"

# 2. Chave do juiz RULER (o4-mini via API OpenAI)
export OPENAI_API_KEY=sk-...

# 3. Treinar (20 steps, 8 rollouts por cenário)
cd rl_training
python train_scanner.py

# 4. Servir o modelo treinado pelo backend do ART (vLLM, OpenAI-compat)
#    e então usá-lo no deep agent:
export ANTHROPIC_API_KEY=sk-ant-...
python use_trained_scanner.py
```

## Loop de melhoria contínua (o gancho mais valioso)

Em produção, os padrões 3 e 5 já **geram julgamentos** (verifier confirma/
rejeita; torneio elege vencedor). Esses julgamentos são sinal de treino de
graça: registre as trajetórias reais + seus veredictos e alimente o
`train_scanner.py` com elas. O agente melhora sozinho com o próprio uso —
que é a promessa central do guia da Unsloth.

## Ajustes comuns

- **Modelo base (Qwen3.6, geração atual):** `Qwen/Qwen3.6-27B` (denso, padrão) ou `Qwen/Qwen3.6-35B-A3B` (MoE, ~3B ativos — inferência barata; GRPO com MoE é mais sensível, valide antes). GPU menor: use uma variante Qwen3.6 de menor porte suportada pelo ART.
- **Iterar barato:** mantenha `RULER_JUDGE_MODEL = "openai/o4-mini"`; suba para `o3` só na validação final.
- **Sem GPU local:** o ART tem backends serverless/SkyPilot — troque `LocalBackend` conforme a doc do ART.
- **Observabilidade:** exporte `WANDB_API_KEY` para ver as curvas de reward por step (o `project` agrupa as métricas).
