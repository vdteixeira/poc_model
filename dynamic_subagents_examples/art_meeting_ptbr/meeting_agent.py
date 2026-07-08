"""Agente de reunião pt-BR — módulo de aplicação.

Porta o rollout do notebook de treino (ART + Tinker) para uso em aplicações:
o loop multi-turno com as 3 ferramentas (search_transcript, read_segment,
return_final_answer), o system prompt treinado e as blindagens de produção
(argumentos malformados, resposta forçada no último turno).

Uso com o modelo treinado servido pelo Tinker (dev/interno):

    from meeting_agent import connect_tinker, answer_question, MeetingTranscript

    client, model_name = await connect_tinker()          # lê .art/ + TINKER_API_KEY
    transcript = MeetingTranscript.from_rows(
        title="Planejamento Q3", date="2026-07-10",
        rows=[("Ana", "Precisamos fechar o escopo."), ("Bruno", "Eu cuido da POC.")],
    )
    result = await answer_question(client, model_name, transcript,
                                   "Quais tarefas foram atribuídas?")
    print(result.answer)

Uso com qualquer endpoint OpenAI-compatible (vLLM, Fireworks, OpenAI):

    from meeting_agent import connect_openai_compatible

    client, model_name = connect_openai_compatible(
        base_url="http://localhost:8000/v1", api_key="...",
        model_name="meeting-agent-ptbr")
    # Para modelos gpt-5.x da OpenAI: token_limit_param="max_completion_tokens".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

MAX_TURNS = 10

# O ART grava o estado em .art/ relativo ao cwd; ancoramos na pasta deste
# módulo (onde os notebooks treinaram) para funcionar de qualquer cwd.
DEFAULT_ART_PATH = str(Path(__file__).resolve().parent / ".art")

SYSTEM_PROMPT = (
    "Você é um assistente de reuniões em português do Brasil. Sua tarefa é "
    "responder à pergunta do usuário consultando a transcrição de uma reunião "
    "por meio das ferramentas disponíveis. Use search_transcript para "
    "localizar segmentos relevantes e read_segment para lê-los na íntegra. "
    "Responda SEMPRE em português, de forma objetiva e fiel à transcrição — "
    "não invente tarefas, decisões ou nomes que não estejam nos segmentos. "
    "Quando tiver a resposta, chame return_final_answer."
)

TOOLS = [
    {"type": "function", "function": {
        "name": "search_transcript",
        "description": "Busca segmentos da transcrição que contenham as palavras-chave.",
        "parameters": {"type": "object", "properties": {
            "keywords": {"type": "array", "items": {"type": "string"}}},
            "required": ["keywords"]}}},
    {"type": "function", "function": {
        "name": "read_segment",
        "description": "Lê o texto completo de um segmento pelo seu índice.",
        "parameters": {"type": "object", "properties": {
            "segment_idx": {"type": "integer"}}, "required": ["segment_idx"]}}},
    {"type": "function", "function": {
        "name": "return_final_answer",
        "description": "Devolve a resposta final ao usuário e encerra a tarefa.",
        "parameters": {"type": "object", "properties": {
            "answer": {"type": "string"},
            "segment_refs": {"type": "array", "items": {"type": "integer"}}},
            "required": ["answer"]}}},
]


@dataclass
class Segment:
    idx: int
    speaker: str
    text: str


@dataclass
class MeetingTranscript:
    title: str
    date: str
    segments: list[Segment]

    @classmethod
    def from_rows(cls, title: str, date: str,
                  rows: list[tuple[str, str]]) -> "MeetingTranscript":
        return cls(title=title, date=date,
                   segments=[Segment(i, sp, tx) for i, (sp, tx) in enumerate(rows)])


@dataclass
class AgentAnswer:
    answer: str
    segment_refs: list[int] = field(default_factory=list)
    via_tool: bool = True     # False = resposta veio de texto livre do modelo
    forced: bool = False      # True = precisou da chamada forçada do último turno
    tool_calls: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)  # trace p/ debug


def _search(transcript: MeetingTranscript, keywords: Any) -> str:
    # O modelo às vezes manda keywords como string ("a, b, c") em vez de lista.
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",")]
    kws = [k.lower() for k in keywords or [] if isinstance(k, str) and k.strip()]
    hits = [{"segment_idx": s.idx, "speaker": s.speaker, "snippet": s.text[:80]}
            for s in transcript.segments
            if any(k in s.text.lower() or k in s.speaker.lower() for k in kws)]
    return json.dumps({"results": hits[:10]}, ensure_ascii=False)


def _read(transcript: MeetingTranscript, idx: int) -> str:
    for s in transcript.segments:
        if s.idx == idx:
            return json.dumps(
                {"segment_idx": s.idx, "speaker": s.speaker, "text": s.text},
                ensure_ascii=False)
    return json.dumps({"error": f"segmento {idx} não existe"}, ensure_ascii=False)


def _strip_think(text: str | None) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def _assistant_to_message(msg: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        out["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name,
                          "arguments": tc.function.arguments or "{}"}}
            for tc in msg.tool_calls]
    return out


def _dispatch_tool(transcript: MeetingTranscript, name: str,
                   args: dict[str, Any]) -> tuple[str, str | None, list[int]]:
    """Executa uma tool call. Retorna (resultado_json, resposta_final, refs)."""
    final_answer, refs = None, []
    try:
        if name == "search_transcript":
            result = _search(transcript, args.get("keywords", []))
        elif name == "read_segment":
            idx = args.get("segment_idx", -1)
            if isinstance(idx, list):  # o modelo às vezes manda lista
                idx = idx[0] if idx else -1
            result = _read(transcript, int(idx))
        elif name == "return_final_answer":
            final_answer = args.get("answer", "")
            raw_refs = args.get("segment_refs") or []
            refs = [int(r) for r in raw_refs if isinstance(r, (int, str))
                    and str(r).lstrip("-").isdigit()]
            result = json.dumps({"ok": True}, ensure_ascii=False)
        else:
            result = json.dumps({"error": "ferramenta desconhecida"}, ensure_ascii=False)
    except Exception as e:  # argumento malformado vira feedback, nunca crash
        result = json.dumps({"error": f"argumentos inválidos: {e}"}, ensure_ascii=False)
    return result, final_answer, refs


async def answer_question(
    client: AsyncOpenAI,
    model_name: str,
    transcript: MeetingTranscript,
    question: str,
    *,
    max_turns: int = MAX_TURNS,
    max_tokens: int = 1024,
    token_limit_param: str = "max_tokens",  # gpt-5.x: "max_completion_tokens"
) -> AgentAnswer:
    """Roda o agente completo (multi-turno + ferramentas) e devolve a resposta."""
    token_kwargs = {token_limit_param: max_tokens}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Reunião: {transcript.title} ({transcript.date}). "
            f"A transcrição tem {len(transcript.segments)} segmentos.\n\n"
            f"Pergunta: {question}")},
    ]
    final_answer: str | None = None
    refs: list[int] = []
    n_tool_calls = 0

    for _ in range(max_turns):
        completion = await client.chat.completions.create(
            model=model_name, messages=messages, tools=TOOLS, **token_kwargs)
        msg = completion.choices[0].message
        messages.append(_assistant_to_message(msg))
        tool_calls = msg.tool_calls or []
        if not tool_calls:
            # Texto livre sem tool call: usa como resposta (melhor que vazio).
            content = _strip_think(msg.content)
            if content:
                return AgentAnswer(answer=content, via_tool=False,
                                   tool_calls=n_tool_calls, messages=messages)
            break
        for tc in tool_calls:
            n_tool_calls += 1
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result, answer, tc_refs = _dispatch_tool(transcript, tc.function.name, args)
            if answer is not None:
                final_answer, refs = answer, tc_refs
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result})
        if final_answer is not None:
            break

    forced = False
    if final_answer is None:
        # Último recurso: força a resposta com o que já foi lido.
        forced = True
        messages.append({
            "role": "user",
            "content": ("Os turnos de busca acabaram. Responda AGORA chamando a "
                        "ferramenta return_final_answer com a melhor resposta "
                        "possível com base no que você já leu da transcrição."),
        })
        completion = await client.chat.completions.create(
            model=model_name, messages=messages, tools=TOOLS,
            tool_choice={"type": "function",
                         "function": {"name": "return_final_answer"}},
            **token_kwargs)
        msg = completion.choices[0].message
        messages.append(_assistant_to_message(msg))
        for tc in msg.tool_calls or []:
            n_tool_calls += 1
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            _, answer, tc_refs = _dispatch_tool(transcript, tc.function.name, args)
            if answer is not None:
                final_answer, refs = answer, tc_refs
            messages.append(
                {"role": "tool", "tool_call_id": tc.id,
                 "content": json.dumps({"ok": True}, ensure_ascii=False)})

    return AgentAnswer(
        answer=_strip_think(final_answer) or "",
        segment_refs=refs, forced=forced,
        tool_calls=n_tool_calls, messages=messages)


async def connect_tinker(
    name: str = "meeting-agent-ptbr-002",
    project: str = "tag-ai-meeting-agent",
    base_model: str = "Qwen/Qwen3.6-35B-A3B",
    art_path: str | None = None,
) -> tuple[AsyncOpenAI, str]:
    """Conecta no modelo treinado servido pelo Tinker (via ART TinkerNativeBackend).

    Requer TINKER_API_KEY no ambiente e o estado do treino no .art/ desta pasta
    (onde os notebooks treinaram; restaurável pelo notebook de marcos). Use
    art_path para apontar outro diretório de estado.
    """
    import art
    from art.tinker_native import TinkerNativeBackend

    backend = TinkerNativeBackend(path=art_path or DEFAULT_ART_PATH)
    model = art.TrainableModel(name=name, project=project, base_model=base_model)
    await model.register(backend)
    step = await model.get_step()
    if step == 0:
        raise RuntimeError(
            "Modelo no step 0 (stock): estado do treino não encontrado em .art/. "
            "Rode a partir do checkout do treino ou restaure com o notebook de marcos.")
    return model.openai_client(), model.get_inference_name()


def connect_openai_compatible(
    base_url: str, api_key: str, model_name: str,
) -> tuple[AsyncOpenAI, str]:
    """Conecta em qualquer endpoint OpenAI-compatible (vLLM, Fireworks, OpenAI...)."""
    return AsyncOpenAI(base_url=base_url, api_key=api_key), model_name


if __name__ == "__main__":
    import asyncio

    async def _demo() -> None:
        try:
            from dotenv import load_dotenv, find_dotenv
            load_dotenv(find_dotenv(usecwd=True))
        except ImportError:
            pass
        client, model_name = await connect_tinker()
        transcript = MeetingTranscript.from_rows(
            title="Planejamento Q3 — Produto", date="2026-07-10",
            rows=[
                ("Ana", "Bom dia. Hoje precisamos fechar o escopo do Q3."),
                ("Bruno", "A prioridade continua sendo o app de desktop, certo?"),
                ("Ana", "Sim, tem que sair até o fim de julho."),
                ("Carla", "Eu assumo a captura de áudio do desktop então."),
                ("Bruno", "E eu fico com a integração do Teams, POC até dia 20."),
            ])
        result = await answer_question(
            client, model_name, transcript, "Quais tarefas foram atribuídas e a quem?")
        print(f"\nResposta ({result.tool_calls} tool calls"
              f"{', forçada' if result.forced else ''}):\n{result.answer}")
        if result.segment_refs:
            print("Segmentos citados:", result.segment_refs)

    asyncio.run(_demo())
