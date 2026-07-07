# Agente de reunião pt-BR com RL — espelho do notebook ART-E

Réplica do [ART-E (Enron email agent)](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/art-e.ipynb)
adaptada para o domínio do Tag AI (assistente de reuniões) **e com um
dataset pequeno em português** que você não precisa ter de antemão — ele é
gerado à mão em `dataset_ptbr.py`.

## Por que isto resolve o "não tenho dados de português"

O RULER (o juiz do ART) pontua as tentativas do modelo **comparando-as entre
si** — não precisa de respostas rotuladas. Você só precisa de **cenários**:
uma transcrição + uma pergunta. Este exemplo traz 8 cenários sintéticos de
reunião em pt-BR (5 treino, 3 validação), incluindo um cenário de controle
"sem decisões" que ensina o modelo a não inventar tarefas.

## Mapeamento ART-E → Tag AI

| ART-E (e-mail) | Este exemplo (reunião) |
|---|---|
| caixa de e-mail | transcrição em segmentos (falante + texto) |
| `search_inbox(keywords)` | `search_transcript(keywords)` |
| `read_email(message_id)` | `read_segment(segment_idx)` |
| `return_final_answer(...)` | `return_final_answer(answer, segment_refs)` |
| dataset de perguntas Enron | `dataset_ptbr.py` (sintético, pt-BR) |
| RULER (reward relativo) | RULER — **idêntico**, sem rótulos |
| juiz de correção (validação) | `judge_correctness` (validação, opcional) |

A estrutura é 1:1 com o notebook: agente multi-turno com tool-calling (até 6
turnos), `gather_trajectory_groups` → `ruler_score_group` → `model.train`,
com validação a cada N steps.

## Arquivos

| Arquivo | Papel |
|---|---|
| `dataset_ptbr.py` | 8 cenários de reunião em pt-BR (transcrição + pergunta) |
| `train_meeting_agent.py` | Agente + ferramentas + rollout + RULER + loop de treino |

## Como rodar

```bash
# GPU NVIDIA/CUDA obrigatória (não roda em Apple Silicon)
pip install "openpipe-art[backend]" openai tenacity langdetect

export OPENAI_API_KEY=sk-...   # juiz RULER + juiz de correção

python train_meeting_agent.py   # NUM_STEPS=5 -> smoke test do ciclo
```

## Escolha do modelo base — Qwen3.6-27B (denso)

O script usa `Qwen/Qwen3.6-27B`: **denso** (não MoE), 27B, pós-treinado, 262K
de contexto nativo. A vantagem de ser denso vs o 35B-A3B (MoE) é decisiva
para custo: **QLoRA 4-bit funciona** (a limitação de QLoRA valia só para MoE).

| Caminho | VRAM de pesos | GPU típica |
|---|---|---|
| QLoRA 4-bit | ~14GB | 24–48GB (RTX 4090 / A6000) |
| bf16 LoRA | ~54GB | A100/H100 80GB |

Ou seja: o 27B denso treina em GPU muito mais barata que o 35B-A3B MoE, sem
a armadilha do QLoRA-em-MoE que a pesquisa apontou.

## Correções aplicadas nesta versão

1. **Modelo base `Qwen/Qwen3.6-27B`** (denso; id verificado no HuggingFace).
2. **Workaround multi-turn do Qwen 3.x** — o chat template remove tokens
   `<think>` de turnos anteriores, corrompendo o treino multi-turno; o rollout
   agora expõe cada turno via `additional_histories` (padrão oficial do ART).
3. **Métrica de consistência de idioma** (`langdetect`) — monitora drift do
   português para o inglês, que a pesquisa mostrou ocorrer em GRPO sem sinal
   de idioma. É métrica de monitoramento, não reward.
4. **`NUM_STEPS=5`** por padrão — smoke test honesto: com 5 cenários, treinar
   além disso só causa overfitting. Suba junto com o tamanho do dataset.

## Do treino ao produto

Depois de treinado, o backend do ART serve o modelo numa **API compatível com
OpenAI**. Pluga-se como subagente no deep agent exatamente como em
`../rl_training/use_trained_scanner.py` — `ChatOpenAI(model="meeting-agent-ptbr-001",
base_url=<endpoint do ART>)` no campo `model` do subagente de reunião.

## Eval matricial (tamanhos × estágios) — `eval_matrix.py`

Compara o treino entre tamanhos de modelo (4B/8B/14B/27B...) e estágios
(stock/SFT/SFT+RL) nos mesmos cenários de validação. Métricas por célula:
`correctness` (juiz LLM), `is_pt`, `answered`, `no_hallucin` (cenário de
controle), `avg_turns` e `avg_out_toks` (proxies de custo). Desacoplado do
ART: qualquer endpoint OpenAI-compat entra como candidato. **Regra de
decisão: o menor tamanho que passa a barra (correctness≥0.90, is_pt=1.00,
no_hallucin=1.00) é o subagente mais barato viável.**

## Destilação barata — `distill_generate.py`

Gera o dataset sintético do cold start com o **Gemini 2.5 Flash-Lite**
(US$0,10/1M in, US$0,40/1M out — o mais barato de Google/OpenAI em jul/2026,
empatado com o GPT-4.1 Nano, que fica como alternativa de uma linha).
Produz `sft_pairs.jsonl` (fase 1, SFT) e `scenarios_gen.jsonl` (fase 2,
RL/RULER + eval), com ~15% de cenários de controle sem decisões e
transcrições de 25-60 segmentos (busca não-trivial). ~US$0,25 por 200
cenários. ⚠️ Confira os termos de uso do provedor sobre treinar modelos com
outputs; alternativa sem restrição: professor open-weight (Qwen maior).

## O caminho recomendado (da pesquisa de viabilidade)

Este exemplo é **RL puro** para espelhar o ART-E. Mas a pesquisa mostrou que,
para reforço de idioma em modelos leves, o melhor custo-benefício é
**SFT/destilação primeiro, GRPO+RULER como refinamento**. Quando você tiver
transcrições reais consentidas, gere pares (transcrição → resposta) com um
modelo de fronteira, faça SFT, e só então use este loop de RULER por cima.
Aqui o RULER sozinho já serve como prova de conceito do ciclo.
