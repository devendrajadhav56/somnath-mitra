import json
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services import llm, pipeline
from services.applog import log_debug, log_step

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

    plan = await pipeline.plan_and_execute(req.message, history, req.user_lat, req.user_lon)

    # ── Off-topic: refuse without a Generate call ────────────────────────────
    if plan.off_topic:
        if req.stream:
            async def _refusal_stream():
                yield pipeline.OFF_TOPIC_REPLY
                yield "\x00" + json.dumps({
                    "elapsed_ms": round((time.monotonic() - t0) * 1000),
                    "intent": plan.intent, "sources": [], "origin": {}, "timings": {}, "products": [],
                })
            return StreamingResponse(_refusal_stream(), media_type="text/plain")
        return ChatResponse(reply=pipeline.OFF_TOPIC_REPLY, intent=plan.intent, sources=[], origin={})

    sources = pipeline.build_sources(plan.chunks)

    # ── Generate ─────────────────────────────────────────────────────────────
    t_gen = time.monotonic()
    if req.stream:
        timings = plan.timings

        async def token_stream():
            first = True
            full_text: list[str] = []
            async for token in llm.chat_stream(req.message, history, plan.chunks, plan.structured):
                if first:
                    timings["ttft_ms"] = round((time.monotonic() - t0) * 1000)
                    first = False
                full_text.append(token)
                yield token
            full = "".join(full_text)
            for suffix in (
                llm.booking_link_suffix(req.message, full),
                llm.pooja_link_suffix(req.message, full),
            ):
                if suffix:
                    yield suffix
            timings["llm_total_ms"] = round((time.monotonic() - t_gen) * 1000)
            log_step("llm", model="main", mode="stream", ms=timings["llm_total_ms"])
            log_step("reply", chars=len(full), text=full[:300])
            log_debug("reply_full", text=full)
            meta = {
                "elapsed_ms": round((time.monotonic() - t0) * 1000),
                "intent": plan.intent,
                "sources": sources,
                "origin": plan.origin,
                "timings": timings,
                "products": plan.products,
            }
            yield "\x00" + json.dumps(meta)

        return StreamingResponse(token_stream(), media_type="text/plain")

    reply = await llm.chat(req.message, history, plan.chunks, plan.structured)
    reply += llm.booking_link_suffix(req.message, reply)
    reply += llm.pooja_link_suffix(req.message, reply)
    log_step("llm", model="main", mode="sync", reply_chars=len(reply),
             ms=round((time.monotonic() - t_gen) * 1000))
    log_step("reply", chars=len(reply), text=reply[:300])
    log_debug("reply_full", text=reply)
    log_step("done", total_ms=round((time.monotonic() - t0) * 1000))
    return ChatResponse(reply=reply, intent=plan.intent, sources=sources, origin=plan.origin)
