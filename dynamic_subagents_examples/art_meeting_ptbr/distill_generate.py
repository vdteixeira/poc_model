"""Gerador de dataset sintético para o cold start (SFT por destilação).

Professor: **Gemini 2.5 Flash-Lite** — o modelo mais barato de Google/OpenAI
em jul/2026 (US$0,10/1M input, US$0,40/1M output; empatado com o GPT-4.1
Nano da OpenAI). Acessado pelo endpoint OpenAI-compat do Google, então o
código é o mesmo se você trocar para o Nano (mude PROVIDER abaixo).

Custo estimado: 200 cenários × ~3K tokens ≈ US$0,25. A destilação é a parte
BARATA do pipeline — se a qualidade pt-BR do professor decepcionar na
inspeção manual, subir para Gemini Flash (não-Lite) custa centavos a mais.

⚠️ Termos de uso: OpenAI e Google restringem usar outputs para treinar
modelos que concorram com os deles. Para um subagente interno de produto o
risco é baixo, mas leia os termos vigentes antes de escalar; a alternativa
sem restrição é destilar de um professor open-weight (ex.: um Qwen maior,
Apache 2.0).

Saída (dois arquivos):
    sft_pairs.jsonl        — {"messages": [...]} por linha: transcrição ->
                             resposta do professor. Alimenta o SFT (fase 1).
    scenarios_gen.jsonl    — cenários no formato do dataset_ptbr (sem
                             resposta). Alimenta o RL/RULER (fase 2) e o eval.

Uso:
    export GEMINI_API_KEY=...      # https://aistudio.google.com/apikey
    python distill_generate.py --n 200
"""

import argparse
import asyncio
import json
import os
import random

from openai import AsyncOpenAI

# --- Professor: troque de provider mudando este bloco ------------------
PROVIDER = "google"  # "google" | "openai"

if PROVIDER == "google":
    TEACHER_MODEL = "gemini-2.5-flash-lite"
    teacher = AsyncOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.environ["GEMINI_API_KEY"],
    )
else:
    TEACHER_MODEL = "gpt-4.1-nano"  # mesmo preço; alternativa OpenAI
    teacher = AsyncOpenAI()  # usa OPENAI_API_KEY

# Variedade controlada: sorteamos domínio × tipo de pergunta × dificuldade.
DOMAINS = [
    "planejamento de produto", "revisão de incidente de produção",
    "reunião comercial com cliente", "design review de software",
    "compliance e privacidade de dados", "contratação e time",
    "precificação e finanças", "retrospectiva de sprint",
    "alinhamento de marketing", "suporte e sucesso do cliente",
]
QUESTION_TYPES = [
    "tarefas atribuídas com responsáveis e prazos",
    "decisões tomadas e seus motivos",
    "riscos ou bloqueios levantados",
    "próximos passos por pessoa",
    "o que ficou pendente sem dono",
]
# ~15% de reuniões SEM decisões/tarefas: ensina a não alucinar.
CONTROL_RATE = 0.15
# Transcrições longas o bastante p/ busca não-trivial (ver avaliação: com
# poucos segmentos a política ótima é "ler tudo", que não generaliza).
MIN_SEGMENTS, MAX_SEGMENTS = 25, 60

GEN_PROMPT = """Gere uma reunião de trabalho FICTÍCIA e realista em português do Brasil.

Domínio: {domain}
A reunião {control} conter tarefas/decisões concretas.
Número de segmentos de fala: entre {min_seg} e {max_seg} (segmentos curtos, \
1-3 frases, como saída real de diarização; inclua conversa natural, \
digressões e small talk entre os pontos importantes).

Depois da transcrição, formule UMA pergunta sobre: {qtype}.
Por fim, escreva a resposta ideal (objetiva, fiel à transcrição; se não \
houver tarefas/decisões, diga isso explicitamente).

Responda APENAS com JSON válido:
{{"title": "...", "date": "AAAA-MM-DD",
  "segments": [{{"speaker": "Nome", "text": "..."}}],
  "question": "...", "expected": "..."}}"""


async def generate_one(i: int) -> dict | None:
    rng = random.Random(i)  # semente por índice -> reprodutível
    is_control = rng.random() < CONTROL_RATE
    prompt = GEN_PROMPT.format(
        domain=rng.choice(DOMAINS),
        control="NÃO deve" if is_control else "deve",
        min_seg=MIN_SEGMENTS,
        max_seg=MAX_SEGMENTS,
        qtype=rng.choice(QUESTION_TYPES),
    )
    resp = await teacher.chat.completions.create(
        model=TEACHER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=4096,
        temperature=1.0,  # diversidade entre cenários
    )
    try:
        data = json.loads(resp.choices[0].message.content or "")
        assert data["segments"] and data["question"] and data["expected"]
        data["id"] = f"gen-{i:04d}"
        data["control"] = is_control
        return data
    except (json.JSONDecodeError, KeyError, AssertionError):
        return None  # descarta malformados; o loop repõe


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    print(f"Gerando {args.n} cenários com {TEACHER_MODEL}...")
    results = await asyncio.gather(*(generate_one(i) for i in range(args.n)))
    scenarios = [r for r in results if r]
    print(f"{len(scenarios)} válidos ({args.n - len(scenarios)} descartados)")

    with open("scenarios_gen.jsonl", "w", encoding="utf-8") as f:
        for s in scenarios:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Pares de SFT: (system + transcrição + pergunta) -> resposta do professor.
    # No SFT a transcrição vai INTEIRA no prompt (sem ferramentas): o cold
    # start ensina idioma/formato/tarefa; o comportamento de busca multi-turno
    # fica para a fase de RL.
    with open("sft_pairs.jsonl", "w", encoding="utf-8") as f:
        for s in scenarios:
            transcript = "\n".join(
                f"[{i}] {seg['speaker']}: {seg['text']}"
                for i, seg in enumerate(s["segments"])
            )
            f.write(json.dumps({"messages": [
                {"role": "system",
                 "content": "Você é um assistente de reuniões em português do "
                            "Brasil. Responda de forma objetiva e fiel à "
                            "transcrição; não invente tarefas ou decisões."},
                {"role": "user",
                 "content": f"Transcrição da reunião \"{s['title']}\" "
                            f"({s['date']}):\n{transcript}\n\n"
                            f"Pergunta: {s['question']}"},
                {"role": "assistant", "content": s["expected"]},
            ]}, ensure_ascii=False) + "\n")

    n_control = sum(1 for s in scenarios if s["control"])
    print(f"sft_pairs.jsonl e scenarios_gen.jsonl gravados "
          f"({n_control} cenários de controle sem decisões).")
    print("Antes de treinar: INSPECIONE ~10 amostras manualmente — se o "
          "pt-BR do professor decepcionar, suba para gemini-2.5-flash.")


if __name__ == "__main__":
    asyncio.run(main())
