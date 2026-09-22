"""QA endpoint — simple GET interface for testing the full pipeline.

Usage:
    GET /qa?q=Which trains go to Somnath from Mumbai?

Returns full JSON: reply, intent (routing decision), sources (RAG chunks used),
and the structured tool data that was injected into the generator.

Shares the same Plan → Execute pipeline as /chat (services.pipeline), so its
routing, origin handling, and off-topic behaviour stay in lock-step with the
main chat endpoint.
"""
import time

from fastapi import APIRouter, Query
from pydantic import BaseModel

from services import llm, pipeline
from services.applog import log_step

router = APIRouter(prefix="/qa", tags=["qa"])


class QAResponse(BaseModel):
    question: str
    reply: str
    intent: dict
    sources: list[dict]
    structured_data: str
    origin: dict
    elapsed_ms: float


@router.get("", response_model=QAResponse)
async def qa_endpoint(
    q: str = Query(..., description="The question to ask Shivoham"),
    user_lat: float | None = Query(None, description="User latitude (optional)"),
    user_lon: float | None = Query(None, description="User longitude (optional)"),
):
    t0 = time.monotonic()
    log_step("query", message=q[:120])

    plan = await pipeline.plan_and_execute(q, [], user_lat, user_lon)

    if plan.off_topic:
        reply = pipeline.OFF_TOPIC_REPLY
    else:
        reply = await llm.chat(q, [], plan.chunks, plan.structured)
        reply += llm.booking_link_suffix(q, reply)
        reply += llm.pooja_link_suffix(q, reply)
        log_step("llm", model="main", reply_chars=len(reply))

    total_ms = round((time.monotonic() - t0) * 1000, 1)
    log_step("done", total_ms=total_ms)

    return QAResponse(
        question=q,
        reply=reply,
        intent=plan.intent,
        sources=pipeline.build_sources(plan.chunks),
        structured_data=plan.structured,
        origin=plan.origin,
        elapsed_ms=total_ms,
    )
