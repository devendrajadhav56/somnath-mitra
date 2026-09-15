"""Intent detection via a focused LLM call.

Returns a routing decision — which tools to call and whether to use RAG —
before the main chat call is made.
"""
from __future__ import annotations

import json
import logging

from services.llm import get_client
import config

log = logging.getLogger(__name__)

_FALLBACK = {"tools": [], "use_rag": True}

INTENT_SYSTEM_PROMPT = """\
You are an intent classifier for Somnath Mitra, a pilgrim assistant chatbot for \
Somnath Jyotirlinga temple, Gujarat, India.

Analyse the user's message (and recent conversation history if provided) and decide:
1. Which data-fetching tools to call, with what parameters
2. Whether to also search the RAG knowledge base (scraped temple website content)

━━ Available tools ━━

search_restaurants
  Find restaurants, dhabas, and food places near the temple.
  params:
    radius_m   : int   — search radius in metres (default 2000)
    min_rating : float — minimum Google rating filter, null for none (default null)
    limit      : int   — max results (default 10)

search_pois
  Find places of interest near the temple.
  params:
    radius_m : int — search radius in metres (default 5000)
    category : str — filter by category, null for all (default null)
                     known values: pilgrimage_attraction, lodging,
                                   food, essential_services,
                                   transport_hub, tour_travel_agency
    limit    : int — max results (default 15)

get_temple_info
  Retrieve a structured document from the official temple database.
  params:
    key : str — one of:
      darshan_timings        opening hours, aarti times, AND light-and-sound show (use this for light show queries)
      visitor_rules          dress code, gadgets, photography, footwear, smoking
      pilgrim_facilities     guest houses, dormitories, bus service, room pricing
      nearest_places         nearby attractions with distances
      festivals_calendar     festival names and dates
      faqs                   frequently asked questions
      social_activities      trust's charitable and social work
      contact_info           phone numbers, emails, office addresses
      history_significance   temple history and religious significance
      heritage_and_temple_walks  heritage walk and temple walk information
      how_to_reach_by_air    nearest airports, airlines, and route options

search_trains
  Find train services to/from Veraval Junction (VRL) or Somnath station.
  Veraval is the main railhead, 6 km from Somnath temple.
  Most trains terminate at Veraval; a few continue to Somnath station itself.
  params:
    origin_city : str — filter by source or destination city name
                        (e.g. "Mumbai", "Ahmedabad", "Delhi"), null for all trains
    day         : str — filter by running day: "monday"…"sunday", null for all
    limit       : int — max results (default 10)

search_buses
  Find bus routes to/from Somnath from cities across Gujarat and beyond.
  params:
    origin_city      : str — filter by origin city name
                             (e.g. "Ahmedabad", "Junagadh", "Rajkot"), null for all
    destination_city : str — filter by destination city name (usually "Somnath"),
                             null for all
    limit            : int — max results (default 10)

plan_route_to_somnath
  Plan a multi-modal route from ANY origin city or town in India to Somnath.
  Use this when the user mentions a specific origin location and asks how to reach Somnath.
  This tool finds the nearest train access hub(s) and constructs complete journeys
  (road to hub → train to Veraval → taxi to Somnath).
  params:
    origin : object — the geocoded origin location (injected automatically by the system,
                      leave as {} — the chat endpoint fills this in)

search_shop
  Find products available at the official Somnath temple shop (somnathprasad.com).
  Products are prasad items offered by the temple: Sarees, Kotis (upper garments),
  Pitambers (lower garments), Kurtas, Prasad boxes/combos, Silver Coins, and Dhwaja (temple flag).
  Use this when users ask about buying prasad online, ordering temple items, sarees from Somnath,
  temple merchandise, or prices of shop items.
  params:
    category  : str — filter by category: "Saree", "Koti", "Pitamber", "Kurta",
                      "Prasad", "Silver Coin", "Dhwaja", null for all
    min_price : int — minimum price in INR, null for no limit
    max_price : int — maximum price in INR, null for no limit
    limit     : int — max results (default 8)

━━ Routing rules ━━

use_rag = true  → when the query involves general/historical information or could
                   benefit from scraped website content alongside structured data
use_rag = false → when structured tools fully cover the query
                   (e.g. pure restaurant, POI, transport searches)

tools = []      → when RAG alone is sufficient (history, general temple info)

You may call get_temple_info more than once with different keys if the query spans
multiple topics (e.g. "timings and rules" → both darshan_timings and visitor_rules).

"How to reach Somnath from [place]" OR "I am at [place], how do I get to Somnath" →
  use plan_route_to_somnath({}) + get_temple_info(key="how_to_reach_by_air")
  use_rag=false
  Do NOT call search_trains or search_buses for these queries — plan_route_to_somnath covers them.

Shop/prasad queries ("buy prasad online", "order saree from somnath", "temple shop", item prices) →
  use search_shop with appropriate category/price filters
  use_rag=false

Generic transport queries without a specific origin (train, bus, flight, how to reach) →
  call ALL relevant transport tools together:
    search_trains(origin_city=...) + search_buses(origin_city=...) + get_temple_info(key="how_to_reach_by_air")
  use_rag=false unless the query also asks about the temple itself.

Use conversation history to resolve follow-up queries correctly
(e.g. "what about vegetarian options?" after a restaurant query → search_restaurants).

━━ Output format ━━

Respond with ONLY a valid JSON object, no explanation, no markdown:
{"tools": [{"name": "...", "params": {...}}], "use_rag": true} if only RAG is needed, or {"tools": [...], "use_rag": false} if structured tools are sufficient.
"""


def _validate(data: dict) -> bool:
    if not isinstance(data.get("use_rag"), bool):
        return False
    if not isinstance(data.get("tools"), list):
        return False
    for t in data["tools"]:
        if not isinstance(t.get("name"), str) or not isinstance(t.get("params"), dict):
            return False
    return True


async def detect_intent(user_message: str, history: list[dict]) -> dict:
    """
    Call the LLM to classify intent.
    Returns {"tools": [...], "use_rag": bool}.
    Falls back to RAG-only on any failure.
    """
    # Pass the last 4 messages (2 turns) so follow-ups resolve correctly
    recent_history = history[-4:] if len(history) > 4 else history

    messages = [{"role": "system", "content": INTENT_SYSTEM_PROMPT}]
    messages.extend(recent_history)
    messages.append({"role": "user", "content": user_message})

    try:
        resp = await get_client().chat.completions.create(
            model=config.INTENT_MODEL,
            messages=messages,
            stream=False,
            temperature=0,
            max_tokens=512,
        )
        raw = (resp.choices[0].message.content or "").strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = "\n".join(raw.splitlines()[1:])
            raw = raw.rstrip("`").strip()

        # Some models prefix JSON with analysis text — extract the JSON object
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            raw = raw[start : end + 1]

        data = json.loads(raw)
        if not _validate(data):
            log.warning("intent: invalid schema — %s", data)
            return _FALLBACK

        return data

    except Exception as exc:
        log.warning("intent detection failed (%s) — falling back to RAG", exc)
        return _FALLBACK
