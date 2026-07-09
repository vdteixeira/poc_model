"""Serviço OpenAI-compatible para o modelo treinado no Tinker.

Resolve o problema do cold start: o registro no Tinker (ART TinkerNativeBackend)
acontece UMA vez, no startup do serviço — não a cada request — e um keepalive
periódico mantém o sampler do Tinker aquecido. A aplicação enxerga um endpoint
OpenAI padrão com um nome de modelo estável ("meeting-agent-ptbr"), sem saber
que o Tinker existe.

Subir o serviço (da pasta deste arquivo ou de qualquer cwd):

    python serve_meeting_model.py                  # porta 8100

Usar na aplicação (qualquer SDK OpenAI):

    from meeting_agent import connect_openai_compatible, answer_question
    client, model_name = connect_openai_compatible(
        base_url="http://localhost:8100/v1",
        api_key="qualquer-coisa",                  # ou SERVICE_API_KEY, se definido
        model_name="meeting-agent-ptbr")

Variáveis de ambiente (.env da raiz do repo é carregado automaticamente):
    TINKER_API_KEY       obrigatória
    SERVE_PORT           porta pública (default 8100)
    SERVICE_API_KEY      se definida, exige Authorization: Bearer <chave>
    KEEPALIVE_SECONDS    intervalo do ping de aquecimento (default 240; 0 desliga)
    MODEL_NAME/PROJECT/BASE_MODEL  para servir outro checkpoint
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("serve-meeting-model")

PUBLIC_ALIAS = "meeting-agent-ptbr"
MODEL_NAME = os.environ.get("MODEL_NAME", "meeting-agent-ptbr-002")
PROJECT = os.environ.get("PROJECT", "tag-ai-meeting-agent")
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3.6-35B-A3B")
DEFAULT_ART_PATH = str(Path(__file__).resolve().parent / ".art")


class _State:
    model: Any = None
    inference_name: str = ""
    internal_base_url: str = ""
    internal_api_key: str = ""
    step: int = 0
    http: httpx.AsyncClient | None = None
    keepalive_task: asyncio.Task | None = None


S = _State()


async def _keepalive(interval: int) -> None:
    """Ping mínimo periódico: mantém o sampler do Tinker aquecido.

    Custo: ~60 tokens de prefill por ping (centavos/mês) em troca de eliminar
    o cold start de ~7s extras da primeira chamada após ociosidade.
    """
    while True:
        await asyncio.sleep(interval)
        try:
            assert S.http is not None
            r = await S.http.post(
                f"{S.internal_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {S.internal_api_key}"},
                json={"model": S.inference_name, "max_tokens": 1,
                      "messages": [{"role": "user", "content": "ping"}]},
                timeout=60,
            )
            log.info("keepalive: %s", r.status_code)
        except Exception as e:
            log.warning("keepalive falhou: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from dotenv import load_dotenv, find_dotenv
        load_dotenv(find_dotenv(usecwd=True))
    except ImportError:
        pass
    assert os.environ.get("TINKER_API_KEY"), "TINKER_API_KEY é obrigatória"

    import art
    from art.tinker_native import TinkerNativeBackend

    log.info("Registrando %s/%s no Tinker (uma vez)…", PROJECT, MODEL_NAME)
    backend = TinkerNativeBackend(path=DEFAULT_ART_PATH)
    model = art.TrainableModel(name=MODEL_NAME, project=PROJECT, base_model=BASE_MODEL)
    await model.register(backend)
    step = await model.get_step()
    if step == 0:
        raise RuntimeError(
            "Modelo no step 0 (stock) — estado do treino não encontrado em "
            f"{DEFAULT_ART_PATH}. Restaure com o notebook de marcos.")

    S.model = model
    S.step = step
    S.inference_name = model.get_inference_name()
    S.internal_base_url = model.inference_base_url.rstrip("/")
    S.internal_api_key = model.inference_api_key
    S.http = httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0))

    interval = int(os.environ.get("KEEPALIVE_SECONDS", "240"))
    if interval > 0:
        S.keepalive_task = asyncio.create_task(_keepalive(interval))
    log.info("Pronto: step=%s, alias público '%s' -> %s",
             step, PUBLIC_ALIAS, S.inference_name)
    yield
    if S.keepalive_task:
        S.keepalive_task.cancel()
    if S.http:
        await S.http.aclose()


app = FastAPI(title="meeting-agent-ptbr (Tinker)", lifespan=lifespan)


def _check_auth(request: Request) -> None:
    required = os.environ.get("SERVICE_API_KEY")
    if not required:
        return
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {required}":
        raise HTTPException(status_code=401, detail="API key inválida")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "model": PUBLIC_ALIAS,
            "checkpoint_step": S.step, "inference_name": S.inference_name}


@app.get("/v1/models")
async def models(request: Request) -> dict[str, Any]:
    _check_auth(request)
    return {"object": "list",
            "data": [{"id": PUBLIC_ALIAS, "object": "model",
                      "owned_by": "tag-ai", "root": BASE_MODEL}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    _check_auth(request)
    body = await request.json()
    if body.get("stream"):
        raise HTTPException(
            status_code=400,
            detail="stream=true não é suportado pelo proxy do Tinker; "
                   "use respostas não-streaming.")
    # Alias público -> nome interno resolvido pelo checkpoint atual.
    body["model"] = S.inference_name
    assert S.http is not None
    resp = await S.http.post(
        f"{S.internal_base_url}/chat/completions",
        headers={"Authorization": f"Bearer {S.internal_api_key}"},
        json=body,
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])
    payload = resp.json()
    payload["model"] = PUBLIC_ALIAS  # esconde o nome interno da aplicação
    return payload


if __name__ == "__main__":
    port = int(os.environ.get("SERVE_PORT", "8100"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
