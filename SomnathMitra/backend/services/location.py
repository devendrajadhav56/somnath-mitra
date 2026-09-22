"""Origin location extraction and geocoding.

Two-step pipeline:
  1. extract_origin()  — LLM call that pulls the origin place name from
                         the user's message (e.g. "Dhari", "Junagadh bus stand")
  2. geocode()         — Nominatim lookup: place name → lat/lon + canonical name
  3. resolve_origin()  — chains both; returns None if no location found
"""
from __future__ import annotations

import json

import httpx
import ollama

import config
from services.applog import get_logger

log = get_logger(__name__)

_ORIGIN_OPTIONS = {"num_ctx": 8192, "temperature": 0, "num_predict": 256}

# ── Extraction ────────────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """\
You are a location extractor for a travel assistant.

Given a user message (and optional recent conversation history), extract the \
origin location the user is travelling FROM to reach Somnath temple.

Rules:
- Return ONLY a JSON object: {"location": "<place name>"} or {"location": null}
- The place name should be exactly as the user wrote it (city, town, village, \
landmark, address — anything)
- Return null if the message does not mention an origin location
- Do NOT translate or normalise the name — return it verbatim

Examples:
  "I am here at Dhari, I want to reach Somnath"         → {"location": "Dhari"}
  "I'm currently at Junagadh bus stand"                  → {"location": "Junagadh bus stand"}
  "How do I get to Somnath from Mumbai?"                 → {"location": "Mumbai"}
  "मैं अहमदाबाद में हूं, सोमनाथ कैसे जाऊं?"               → {"location": "अहमदाबाद"}
  "What are the darshan timings?"                        → {"location": null}
  "Can I take a bus?"   (after earlier message mentioning Rajkot) → {"location": null}
  "trains from Ahmedabad, include Sabarmati or other stations"    → {"location": "Ahmedabad"}
  "Give me trains from Ahmedabad to Somnath, include Sabarmati or other Railway stations which fall in Ahmedabad" → {"location": "Ahmedabad"}

Important: when the user mentions both a city AND specific stations/landmarks within that city,
return the CITY name, not the station or landmark name.
"""


async def extract_origin(message: str, history: list[dict]) -> str | None:
    """Return the raw origin place string from the message, or None."""
    recent = history[-4:] if len(history) > 4 else history
    messages = [{"role": "system", "content": _EXTRACT_SYSTEM}]
    messages.extend(recent)
    messages.append({"role": "user", "content": message})

    try:
        client = ollama.AsyncClient(host=config.LLM_BASE_URL.replace("/v1", ""))
        resp = await client.chat(
            model=config.INTENT_MODEL,
            messages=messages,
            think=False,
            options=_ORIGIN_OPTIONS,
        )
        raw = (resp.message.content or "").strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = "\n".join(raw.splitlines()[1:]).rstrip("`").strip()

        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            raw = raw[start : end + 1]

        data = json.loads(raw)
        location = data.get("location")
        return str(location).strip() if location else None

    except Exception as exc:
        log.warning("extract_origin failed (%s)", exc)
        return None


# ── Geocoding ─────────────────────────────────────────────────────────────────

async def geocode(place_name: str) -> dict | None:
    """
    Resolve a place name to coordinates via Nominatim.

    Returns:
        {
            "input":        original place_name,
            "display_name": full Nominatim display string,
            "name":         short canonical name,
            "lat":          float,
            "lon":          float,
            "state":        str | None,
            "district":     str | None,
            "confidence":   "high" | "low",  # low = multiple distinct results
        }
    or None if Nominatim finds nothing.
    """
    params = {
        "q": place_name,
        "countrycodes": "in",
        "format": "jsonv2",
        "limit": 3,
        "addressdetails": 1,
    }
    headers = {"User-Agent": config.NOMINATIM_USER_AGENT}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{config.NOMINATIM_URL}/search",
                params=params,
                headers=headers,
            )
            r.raise_for_status()
            results = r.json()
    except Exception as exc:
        log.warning("geocode(%r) failed (%s)", place_name, exc)
        return None

    if not results:
        return None

    top = results[0]
    addr = top.get("address", {})

    # Low confidence when multiple results land in different states
    confidence = "high"
    if len(results) > 1:
        states = {r.get("address", {}).get("state") for r in results}
        if len(states) > 1:
            confidence = "low"

    return {
        "input": place_name,
        "display_name": top.get("display_name", ""),
        "name": addr.get("city") or addr.get("town") or addr.get("village") or addr.get("hamlet") or place_name,
        "lat": float(top["lat"]),
        "lon": float(top["lon"]),
        "state": addr.get("state"),
        "district": addr.get("state_district") or addr.get("county"),
        "confidence": confidence,
    }


# ── Public API ────────────────────────────────────────────────────────────────

async def resolve_origin(message: str, history: list[dict]) -> dict | None:
    """
    Extract origin location from message then geocode it.
    Returns the geocode result dict (with added "raw_input" key), or None.
    """
    raw = await extract_origin(message, history)
    if not raw:
        return None

    result = await geocode(raw)
    if result:
        result["raw_input"] = raw
    else:
        log.info("resolve_origin: extracted %r but geocode returned nothing", raw)

    return result
