"""Plan → Execute pipeline shared by the chat and qa routers.

Given a user message, this runs the two stages that are identical across every
entry point:

  1. Plan    — detect_intent (which tools + RAG, or off-topic) and, for travel
               queries, resolve the origin and inject it into the route tool.
  2. Execute — run the chosen tools and retrieve RAG chunks.

The final **Generate** stage (the main LLM call) and response shaping stay in
the routers, because they differ: /chat streams token-by-token, /qa returns a
single JSON body.

Returns a `Plan` carrying everything the Generate stage needs.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from services import llm, retriever
from services.applog import log_debug, log_step
from services.intent import detect_intent
from services.location import resolve_origin
from services.tools import execute_tools

# User-facing message when the planner classifies a query as off-topic.
OFF_TOPIC_REPLY = (
    "I can only help with questions about Somnath temple, travel to Somnath, "
    "and the local Somnath/Veraval area.\n\n"
    "मैं केवल सोमनाथ मंदिर, यहाँ की यात्रा और स्थानीय क्षेत्र से जुड़े सवालों में मदद कर सकता हूँ।\n\n"
    "હું ફક્ત સોમનાથ મંદિર, અહીંની યાત્રા અને સ્થાનિક વિસ્તાર સંબંધિત પ્રશ્નોમાં જ મદદ કરી શકું છું।"
)

# Injected as structured context when a travel query has no resolvable origin,
# so the generator asks the user where they are travelling from.
_NO_ORIGIN_PROMPT = (
    "[ROUTE PLANNER] No origin city was found in the user's message. "
    "Ask the user which city or town they are travelling FROM to Somnath. "
    "Ask in the same language the user is writing in. Do not give generic travel info."
)


@dataclass
class Plan:
    """Result of the Plan + Execute stages, consumed by the Generate stage."""
    intent: dict
    origin: dict = field(default_factory=dict)
    structured: str = ""
    products: list[dict] = field(default_factory=list)
    chunks: list[dict] = field(default_factory=list)
    off_topic: bool = False
    timings: dict = field(default_factory=dict)


async def plan_and_execute(
    message: str,
    history: list[dict],
    user_lat: float | None = None,
    user_lon: float | None = None,
) -> Plan:
    """Run Plan + Execute for one user message and return a Plan."""
    t_plan = time.monotonic()
    intent = await detect_intent(message, history)

    if intent.get("off_topic"):
        log_step("off_topic", action="refused", ms=round((time.monotonic() - t_plan) * 1000))
        return Plan(intent=intent, off_topic=True)

    # Origin resolution only matters for travel queries — skip the LLM call otherwise.
    is_travel = any(t.get("name") == "plan_route_to_somnath" for t in intent.get("tools", []))
    origin = (await resolve_origin(message, history)) if is_travel else None
    origin = origin or {}
    log_step("origin", extracted=origin.get("raw_input"), name=origin.get("name"),
             confidence=origin.get("confidence"), ms=round((time.monotonic() - t_plan) * 1000))

    tools_names = [t["name"] for t in intent.get("tools", [])]
    log_step("intent", tools=",".join(tools_names) or "none",
             use_rag=intent.get("use_rag"), ms=round((time.monotonic() - t_plan) * 1000))
    log_step("intent_raw", content=intent, ms=round((time.monotonic() - t_plan) * 1000))

    for tool_call in intent.get("tools", []):
        if tool_call.get("name") == "plan_route_to_somnath":
            tool_call["params"]["origin"] = origin

    # ── Execute ──────────────────────────────────────────────────────────────
    t_tools = time.monotonic()
    if is_travel and not origin:
        structured, products = _NO_ORIGIN_PROMPT, []
        log_step("origin_gate", action="asking_user")
    else:
        structured, products = execute_tools(intent["tools"], user_lat, user_lon)
    log_step("tools", tools=",".join(tools_names) or "none",
             result_chars=len(structured), ms=round((time.monotonic() - t_tools) * 1000))

    t_rag = time.monotonic()
    chunks = retriever.search(message) if intent.get("use_rag") else []
    log_step("rag", chunks=len(chunks),
             top_score=round(chunks[0]["score"], 3) if chunks else None,
             ms=round((time.monotonic() - t_rag) * 1000))
    if chunks:
        log_step("rag_hits", hits=" | ".join(
            f"{c.get('heading') or c.get('page_title') or '?'}@{round(c['score'], 2)}"
            for c in chunks
        ))
    log_debug("rag_full", chunks=[
        {"heading": c.get("heading"), "score": round(c["score"], 3), "content": c["content"]}
        for c in chunks
    ])

    timings = {
        "intent_origin_ms": round((t_tools - t_plan) * 1000),
        "tools_ms":         round((t_rag - t_tools) * 1000),
        "rag_ms":           round((time.monotonic() - t_rag) * 1000),
    }
    return Plan(intent=intent, origin=origin, structured=structured,
                products=products, chunks=chunks, off_topic=False, timings=timings)


def build_sources(chunks: list[dict]) -> list[dict]:
    """Shape RAG chunks into the `sources` list returned to clients."""
    return [
        {
            "chunk_id": c["chunk_id"],
            "heading": c.get("heading"),
            "page_url": c.get("page_url"),
            "score": c["score"],
        }
        for c in chunks
    ]
