#!/usr/bin/env python3
"""Teste rápido da API do serve_meeting_model.py.

Uso (com o serviço no ar):

    python test_api.py                          # health + models + completion + tools
    python test_api.py --agent                  # inclui o agente completo (multi-turno)
    python test_api.py --base-url http://host:8100 --api-key segredo

Funciona em qualquer Python (é só um cliente HTTP — não precisa do ART).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import httpx

DEMO_TOOL = [{"type": "function", "function": {
    "name": "search_transcript",
    "description": "Busca segmentos da transcrição que contenham as palavras-chave.",
    "parameters": {"type": "object", "properties": {
        "keywords": {"type": "array", "items": {"type": "string"}}},
        "required": ["keywords"]}}}]


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'✅' if ok else '❌'} {label}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description="Testa a API do meeting-agent")
    p.add_argument("--base-url", default="http://localhost:8100")
    p.add_argument("--api-key", default=os.environ.get("SERVICE_API_KEY", "test"))
    p.add_argument("--model", default="meeting-agent-ptbr")
    p.add_argument("--agent", action="store_true",
                   help="roda também o agente completo (multi-turno, mais tokens)")
    args = p.parse_args()

    base = args.base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {args.api_key}"}
    client = httpx.Client(timeout=120)
    failures = 0

    print(f"Testando {base}\n")

    # 1. Health
    try:
        r = client.get(f"{base}/health")
        d = r.json()
        ok = r.status_code == 200 and d.get("status") == "ok"
        failures += not check("GET /health", ok,
                              f"step={d.get('checkpoint_step')} "
                              f"({d.get('inference_name')})")
    except Exception as e:
        check("GET /health", False, f"{type(e).__name__}: {e}")
        print("\nO serviço está no ar? Suba com:\n"
              "  /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 "
              "serve_meeting_model.py")
        return 1

    # 2. Models
    r = client.get(f"{base}/v1/models", headers=headers)
    ids = [m["id"] for m in r.json().get("data", [])] if r.status_code == 200 else []
    failures += not check("GET /v1/models", args.model in ids, f"ids={ids}")

    # 3. Chat completion simples (cronometrada)
    t0 = time.perf_counter()
    r = client.post(f"{base}/v1/chat/completions", headers=headers, json={
        "model": args.model, "max_tokens": 32,
        "messages": [{"role": "user",
                      "content": "Responda apenas com a palavra: pronto"}]})
    dt = time.perf_counter() - t0
    content = ""
    if r.status_code == 200:
        content = (r.json()["choices"][0]["message"].get("content") or "").strip()
    failures += not check("POST /v1/chat/completions", r.status_code == 200,
                          f"{dt:.1f}s, resposta={content[:40]!r}")

    # 4. Tool calling
    r = client.post(f"{base}/v1/chat/completions", headers=headers, json={
        "model": args.model, "max_tokens": 128, "tools": DEMO_TOOL,
        "messages": [
            {"role": "system", "content": "Você é um assistente de reuniões."},
            {"role": "user", "content": "Reunião: Teste. A transcrição tem 3 "
             "segmentos.\n\nPergunta: Quais tarefas foram atribuídas?"}]})
    tcs = []
    if r.status_code == 200:
        tcs = r.json()["choices"][0]["message"].get("tool_calls") or []
    failures += not check("tool calling", bool(tcs),
                          f"chamou {tcs[0]['function']['name']}" if tcs
                          else f"status={r.status_code}, sem tool_calls")

    # 5. Streaming deve ser recusado com mensagem clara
    r = client.post(f"{base}/v1/chat/completions", headers=headers, json={
        "model": args.model, "stream": True,
        "messages": [{"role": "user", "content": "oi"}]})
    failures += not check("stream=true rejeitado com 400", r.status_code == 400)

    # 6. Agente completo (opcional)
    if args.agent:
        import asyncio
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from meeting_agent import (MeetingTranscript, answer_question,
                                   connect_openai_compatible)

        async def run_agent() -> None:
            oa_client, model_name = connect_openai_compatible(
                base_url=f"{base}/v1", api_key=args.api_key,
                model_name=args.model)
            transcript = MeetingTranscript.from_rows(
                title="Planejamento Q3", date="2026-07-09",
                rows=[("Ana", "Precisamos fechar o escopo do Q3."),
                      ("Carla", "Eu assumo a captura de áudio do desktop."),
                      ("Bruno", "Eu fico com a integração do Teams, POC até dia 20.")])
            t0 = time.perf_counter()
            res = await answer_question(oa_client, model_name, transcript,
                                        "Quais tarefas foram atribuídas e a quem?")
            dt = time.perf_counter() - t0
            ok = bool(res.answer) and "carla" in res.answer.lower()
            check("agente completo", ok,
                  f"{dt:.1f}s, {res.tool_calls} tool calls, "
                  f"refs={res.segment_refs}")
            print(f"\n  Resposta: {res.answer[:200]}")

        asyncio.run(run_agent())

    print(f"\n{'Tudo OK ✅' if failures == 0 else f'{failures} falha(s) ❌'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
