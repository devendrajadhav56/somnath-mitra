"""QA endpoint — simple GET interface for testing the full pipeline.

Usage:
    GET /qa?q=Which trains go to Somnath from Mumbai?
    GET /qa?q=What are the darshan timings?&debug=true

Returns full JSON: reply, intent (routing decision), sources (RAG chunks used),
and optional debug breakdown of what each step produced.
"""
import asyncio
import time

from fastapi import APIRouter, Query
from pydantic import BaseModel

from services import retriever, llm
from services.applog import log_step
from services.intent import detect_intent
from services.location import resolve_origin
from services.tools import execute_tools

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
    q: str = Query(..., description="The question to ask Somnath Mitra"),
    user_lat: float | None = Query(None, description="User latitude (optional)"),
    user_lon: float | None = Query(None, description="User longitude (optional)"),
):
    t0 = time.monotonic()
    log_step("query", message=q[:120])

    # resolve_origin and detect_intent are independent — run in parallel
    t1 = time.monotonic()
    origin_task = asyncio.create_task(resolve_origin(q, []))
    intent_task = asyncio.create_task(detect_intent(q, []))
    origin, intent = await asyncio.gather(origin_task, intent_task)
    origin = origin or {}
    log_step("origin", extracted=origin.get("raw_input"), name=origin.get("name"),
             confidence=origin.get("confidence"), ms=round((time.monotonic() - t1) * 1000))

    tools_names = [t["name"] for t in intent.get("tools", [])]
    log_step("intent", tools=",".join(tools_names) or "none", use_rag=intent.get("use_rag"),
             ms=round((time.monotonic() - t1) * 1000))

    # inject resolved origin into plan_route_to_somnath params
    for tool_call in intent.get("tools", []):
        if tool_call.get("name") == "plan_route_to_somnath":
            tool_call["params"]["origin"] = origin

    t2 = time.monotonic()
    structured = execute_tools(intent["tools"], user_lat, user_lon)
    log_step("tools", tools=",".join(tools_names) or "none",
             result_chars=len(structured), ms=round((time.monotonic() - t2) * 1000))

    t3 = time.monotonic()
    chunks = retriever.search(q) if intent["use_rag"] else []
    log_step("rag", chunks=len(chunks),
             top_score=round(chunks[0]["score"], 3) if chunks else None,
             ms=round((time.monotonic() - t3) * 1000))

    t4 = time.monotonic()
    
    reply = await llm.chat(q, [], chunks, structured)
    log_step("llm", model="main", reply_chars=len(reply), ms=round((time.monotonic() - t4) * 1000))

    sources = [
        {
            "chunk_id": c["chunk_id"],
            "heading": c.get("heading"),
            "page_url": c.get("page_url"),
            "score": round(c["score"], 4),
        }
        for c in chunks
    ]

    total_ms = round((time.monotonic() - t0) * 1000, 1)
    log_step("done", total_ms=total_ms)

    return QAResponse(
        question=q,
        reply=reply,
        intent=intent,
        sources=sources,
        structured_data=structured,
        origin=origin,
        elapsed_ms=total_ms,
    )
