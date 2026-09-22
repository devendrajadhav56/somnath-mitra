"""Intent detection via a focused LLM call.

Returns a routing decision — which tools to call and whether to use RAG —
before the main chat call is made.
"""
from __future__ import annotations

import json

import ollama

import config
from services.applog import get_logger
from services.tools import planner_tool_docs

log = get_logger(__name__)

_INTENT_OPTIONS = {"num_ctx": 8192, "temperature": 0, "num_predict": 1024}

_FALLBACK = {"tools": [], "use_rag": True}

_INTENT_HEADER = """You are a planner for Shivoham, a pilgrim assistant chatbot for Somnath Jyotirlinga temple, Gujarat, India.

Analyse the user's message (and recent conversation history if provided) and decide:
1. Which data-fetching tools to call, with what parameters
2. Whether to also search the RAG knowledge base (scraped temple website content)

━━ Available tools ━━

"""

_INTENT_RULES = """━━ Routing rules ━━

use_rag = true  → when the query involves general/historical information or could
                   benefit from scraped website content alongside structured data
use_rag = false → when structured tools fully cover the query

tools = []      → when RAG alone is sufficient (history, general temple info)

ALL travel queries (trains, buses, flights, how to reach, journey planning,
  "from X to Somnath", "trains from X", "buses from X", "how do I get there") →
  plan_route_to_somnath({})
  use_rag=false
  Romanized examples also count as travel:
  e.g. (Romanized Gujarati) "rajkot thi somnath kevi rite pahochvu" → plan_route_to_somnath({})
  e.g. (Romanized Hindi) "rajkot se somnath kaise jaye" → plan_route_to_somnath({})

Questions specifically about WHICH railway station to use, where the entire query is
about station selection (e.g. "Somnath station or Veraval Junction?", "which station
is closer?", "should I get off at Veraval or Somnath?") →
  use_rag=true, tools=[]

IMPORTANT: If a station or location name appears only as a reference point (e.g.
"restaurants near Veraval Railway Station", "hotels near Somnath station") that is
NOT a station-selection query — treat it as a restaurant/POI/accommodation query
and use the appropriate tool (search_restaurants, search_pois, etc.).

You may call get_temple_info more than once with different keys if the query spans
multiple topics (e.g. "timings and rules" → both darshan_timings and visitor_rules).

Generic prasad queries ("what is prasad", "tell me about prasad", "prasad at somnath") →
  get_temple_info(key="prasad_info"), use_rag=false

Specific shop/buying queries with explicit purchase intent ("buy prasad online",
  "order saree", "show me sarees", "price of", "temple shop items") →
  search_shop with appropriate filters, use_rag=false

Food/restaurant queries anywhere in the area (near station, near temple, near Veraval,
  pure veg, non-veg, dhaba, etc.) → search_restaurants, use_rag=false
  e.g. "vegetarian restaurants near Veraval Railway Station" → search_restaurants
  e.g. "pure veg food near Somnath" → search_restaurants
  e.g. "Are there pure vegetarian restaurants near Veraval Railway Station and Somnath?" → search_restaurants
  e.g. "restaurants near the station" → search_restaurants
  e.g. "सोमनाथ के पास शाकाहारी रेस्टोरेंट दिखाओ" → search_restaurants
  e.g. "સોમનાથ પાસે શુદ્ધ શાકાહારી રેસ્ટોરેન્ટ બતાવો" → search_restaurants
  e.g. (Romanized Gujarati) "somnath pase saru jamvanu kya male" → search_restaurants
  e.g. (Romanized Hindi) "somnath ke paas veg restaurant batao" → search_restaurants

Hospital/medical queries → search_hospitals, use_rag=false
  e.g. "nearest hospital", "emergency near somnath", "doctor near me", "clinic nearby"
  e.g. "नजदीकी अस्पताल कहाँ है", "સૌથી નજીકની હોસ્પિટલ ક્યાં છે"
  e.g. (Romanized) "somnath pase hospital kya che", "nazdiki hospital kahan hai" → search_hospitals

Pharmacy/medicine queries → search_pharmacies, use_rag=false
  e.g. "pharmacy near somnath", "medical store", "chemist", "where to buy medicine"
  e.g. "नजदीकी दवाई की दुकान", "દવાની દુકાન ક્યાં છે"
  e.g. (Romanized) "davani dukan kya che", "medical store kahan hai" → search_pharmacies

Use conversation history to resolve follow-up queries correctly.
If the user's message is just a city or place name and the recent history shows a travel
query or the assistant asked where the user is travelling from, treat it as plan_route_to_somnath.

━━ Off-topic detection ━━

Set "off_topic": true when the query has NO connection to:
  - Somnath temple or the Jyotirlinga
  - Travel to / from Somnath or Veraval
  - Local area (Somnath, Veraval, Prabhas Patan)
  - Hindu pilgrimage, darshan, aarti, prasad, pooja
  - Accommodation, food, hospitals, pharmacies near Somnath

Examples that ARE off-topic (set off_topic: true):
  "What is the capital of France?", "Write me a Python script",
  "Who won the cricket match?", "Tell me a joke", "Recipe for biryani",
  "Weather in Mumbai", "Stock market tips"

Examples that are NOT off-topic (set off_topic: false):
  Anything about Somnath, Veraval, Gujarat temples, Indian trains/buses
  to Somnath, Hindu rituals, pilgrimage in general, local restaurants,
  hospitals, or travel within India heading toward Somnath.

When off_topic is true, set tools: [] and use_rag: false.

━━ Parameter rules ━━

Only include a param in the output when the user's query explicitly requires a non-default value:
• radius_m   — only set if the user mentions a specific distance ("within 5km", "500m radius")
• min_rating — only set if the user asks for highly rated / top-rated places (e.g. use 4.0)
• limit      — only set if the user asks for more/fewer results than the default
• category   — only set if the user specifies a clear category
• min_price / max_price — only set if the user states a price constraint
If the query gives no signal for a param, omit it entirely — do NOT guess or fill in defaults.

━━ Output format ━━

Respond with ONLY a valid JSON object, no explanation, no markdown:
{"tools": [{"name": "...", "params": {...}}], "use_rag": true, "off_topic": false}
"""

INTENT_SYSTEM_PROMPT = _INTENT_HEADER + planner_tool_docs() + "\n\n" + _INTENT_RULES


def _validate(data: dict) -> bool:
    if not isinstance(data.get("use_rag"), bool):
        return False
    if not isinstance(data.get("tools"), list):
        return False
    for t in data["tools"]:
        if not isinstance(t.get("name"), str) or not isinstance(t.get("params"), dict):
            return False
    # off_topic is optional; if present must be bool
    if "off_topic" in data and not isinstance(data["off_topic"], bool):
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
        client = ollama.AsyncClient(host=config.LLM_BASE_URL.replace("/v1", ""))
        resp = await client.chat(
            model=config.INTENT_MODEL,
            messages=messages,
            think=False,
            options=_INTENT_OPTIONS,
        )
        raw = (resp.message.content or "").strip()

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
