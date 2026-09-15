import asyncio
import json
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services import retriever, llm
from services.applog import log_step
from services.intent import detect_intent
from services.location import resolve_origin
from services.tools import execute_tools

router = APIRouter(prefix="/chat", tags=["chat"])


class Message(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []
    stream: bool = True
    user_lat: float | None = None
    user_lon: float | None = None


class ChatResponse(BaseModel):
    reply: str
    intent: dict = {}
    sources: list[dict] = []
    origin: dict = {}


@router.post("", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    t0 = time.monotonic()
    history = [m.model_dump() for m in req.history]
    log_step("query", message=req.message[:120], stream=req.stream)

    t1 = time.monotonic()
    origin, intent = await asyncio.gather(
        resolve_origin(req.message, history),
        detect_intent(req.message, history),
    )
    print("INTENT", intent)
    origin = origin or {}
    log_step("origin", extracted=origin.get("raw_input"), name=origin.get("name"),
             confidence=origin.get("confidence"), ms=round((time.monotonic() - t1) * 1000))

    tools_names = [t["name"] for t in intent.get("tools", [])]
    log_step("intent", tools=",".join(tools_names) or "none",
             use_rag=intent.get("use_rag"), ms=round((time.monotonic() - t1) * 1000))

    for tool_call in intent.get("tools", []):
        if tool_call.get("name") == "plan_route_to_somnath":
            tool_call["params"]["origin"] = origin

    t2 = time.monotonic()
    structured, products = execute_tools(intent["tools"], req.user_lat, req.user_lon)
    log_step("tools", tools=",".join(tools_names) or "none",
             result_chars=len(structured), ms=round((time.monotonic() - t2) * 1000))

    t3 = time.monotonic()
    chunks = retriever.search(req.message) if intent["use_rag"] else []
    log_step("rag", chunks=len(chunks),
             top_score=round(chunks[0]["score"], 3) if chunks else None,
             ms=round((time.monotonic() - t3) * 1000))

    t4 = time.monotonic()
    if req.stream:
        sources = [
            {
                "chunk_id": c["chunk_id"],
                "heading": c.get("heading"),
                "page_url": c.get("page_url"),
                "score": c["score"],
            }
            for c in chunks
        ]
        timings = {
            "intent_origin_ms": round((t2 - t1) * 1000),
            "tools_ms":         round((t3 - t2) * 1000),
            "rag_ms":           round((t4 - t3) * 1000),
        }

        async def token_stream():
            first = True
            async for token in llm.chat_stream(req.message, history, chunks, structured):
                if first:
                    timings["ttft_ms"] = round((time.monotonic() - t0) * 1000)
                    first = False
                yield token
            timings["llm_total_ms"] = round((time.monotonic() - t4) * 1000)
            log_step("llm", model="main", mode="stream",
                     ms=timings["llm_total_ms"])
            meta = {
                "elapsed_ms": round((time.monotonic() - t0) * 1000),
                "intent": intent,
                "sources": sources,
                "origin": origin,
                "timings": timings,
                "products": products,
            }
            yield "\x00" + json.dumps(meta)

        return StreamingResponse(token_stream(), media_type="text/plain")

    reply = await llm.chat(req.message, history, chunks, structured)
    log_step("llm", model="main", mode="sync", reply_chars=len(reply),
             ms=round((time.monotonic() - t4) * 1000))
    log_step("done", total_ms=round((time.monotonic() - t0) * 1000))
    sources = [
        {
            "chunk_id": c["chunk_id"],
            "heading": c.get("heading"),
            "page_url": c.get("page_url"),
            "score": c["score"],
        }
        for c in chunks
    ]
    return ChatResponse(reply=reply, intent=intent, sources=sources, origin=origin)
