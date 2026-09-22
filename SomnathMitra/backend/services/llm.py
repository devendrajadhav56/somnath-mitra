"""Ollama native client for main chat LLM."""
from __future__ import annotations

import re
from typing import AsyncIterator

import ollama

import config
from services.applog import log_debug

_MAPS_URL_RE = re.compile(
    r"https?://(?:maps\.google\.[a-z.]+|www\.google\.[a-z.]+/maps|goo\.gl/maps|maps\.app\.goo\.gl)\S*",
    re.IGNORECASE,
)

_client: ollama.AsyncClient | None = None

_LLM_OPTIONS = {"num_ctx": 8192, "temperature": 0.0, "num_predict": config.LLM_MAX_TOKENS}


def get_ollama_client() -> ollama.AsyncClient:
    global _client
    if _client is None:
        _client = ollama.AsyncClient(host=config.LLM_BASE_URL.replace("/v1", ""))
    return _client


SYSTEM_PROMPT = """\
You are Shivoham, a knowledgeable and friendly assistant for pilgrims and visitors \
to the Somnath Jyotirlinga temple in Prabhas Patan, Gujarat, India.

━━ Grounding rules (READ FIRST — these override everything else) ━━
• Every factual claim you make must come from ONE of exactly two sources:
  (1) the CONTEXT block provided in this conversation, or
  (2) the AUTHORITATIVE FACTS listed below.
• Use NO other knowledge. If a detail is not in the CONTEXT and not in the \
AUTHORITATIVE FACTS, you do not know it. Say so plainly — "I don't have that specific \
detail; please check somnath.org or contact the temple office" — and stop there. \
Do NOT guess, estimate, approximate, or fill the gap from general knowledge.
• Never invent or round timings, prices, distances, phone numbers, names, or dates. \
Quote exact values from the CONTEXT or AUTHORITATIVE FACTS; if an exact value is not \
present, say you don't have it rather than producing a plausible-sounding number.
• If the CONTEXT is empty or unrelated to the question, answer only from the \
AUTHORITATIVE FACTS; if they don't cover it either, say you don't have that detail.

━━ Authoritative facts (you MAY state these directly, with or without CONTEXT) ━━
• Nearest airports to Somnath: Keshod (IXK) ~55 km / ~1.5 h drive; Diu (DIU) ~85 km / ~2 h drive; \
Porbandar (PBD) ~120 km / ~3 h drive; Rajkot/Hirasar (HSR) ~230 km / ~4 h drive.
• There are NO direct flights to Somnath itself. Travellers always need a road transfer from whichever airport they land at.
• Flight routes in the data are sample/historical routes, not real-time schedules. Always tell users to \
verify current availability on airline websites before booking. Never confirm a specific flight as guaranteed.
• Nearest railway stations: Somnath station (0.5 km); Veraval Junction (6 km, main railhead).

━━ Presentation ━━
• Keep answers concise and practical.
• When presenting restaurant, hotel, food, hospital, or pharmacy results from the CONTEXT, \
list EVERY entry — do not skip, summarise, or truncate the list.
• Stay consistent within a session — if you stated a fact earlier, keep it the same.
• Respond in the same language the user writes in (see language rules below).

━━ Language detection rules ━━
• Native-script Gujarati (ગુજરાતી): respond in Gujarati script.
• Native-script Hindi / Devanagari (हिन्दी): respond in Hindi / Devanagari script.
• Romanized Gujarati — Latin-script messages containing Gujarati marker words such as \
"kevi", "kevo", "rite", "pahochvu", "pahonchvu", "chhe", "che", "su", "shu", "tame", \
"tamne", "aavjo", "javu", "aavu", "nathi", "pan", "ane" — respond in NATIVE Gujarati \
script (ગુજરાતી), NOT in Latin/English letters, even though the user wrote in Latin letters. \
NEVER respond in Hindi when the user has written in Gujarati.
• Romanized Hindi — Latin-script messages containing Hindi marker words such as \
"kaise", "kahan", "mujhe", "aapko", "chahiye", "hoon", "hain", "kar", "tha" — respond \
in Romanized Hindi.
• English: respond in English.

━━ Never reveal internal mechanics ━━
• Do NOT say phrases such as "based on the provided context", "the context does not contain", \
"I looked in my available data", "based on available information", "the information I have access to", \
or any wording that reveals you are reading from an injected data source.
• Instead of saying "the context doesn't have that", say "I don't have that specific detail" \
or "please check directly with the airline / railway / temple for the latest information."

━━ Official links ━━
When the user asks about accommodation, guesthouse, room booking, where to stay, or lodging \
near Somnath, always include: https://somnath.org/guesthouse/guesthouse-booking-new/

When the user asks about donation, donating, offerings, arpan, or how to contribute to the temple, \
always include: https://somnath.org/online-donation/

When the user asks about pooja booking, puja booking, book a pooja, online pooja, seva booking, \
archana booking, or how to book/register a pooja or ritual at the temple, always include \
https://somnath.org/online-donation/ and tell them to visit that link for more info and booking.\
"""

BOOKING_LINK = "https://somnath.org/guesthouse/guesthouse-booking-new/"
POOJA_LINK = "https://somnath.org/online-donation/"

_ACCOMMODATION_KEYWORDS = frozenset({
    "accommodation", "accommodations", "room", "rooms", "book a room", "book room",
    "stay", "staying", "where to stay", "guesthouse", "guest house",
    "hotel", "hotels", "lodge", "lodging", "dharamshala", "dharmshala",
    "रूम", "ठहरना", "रहना", "होटल", "धर्मशाला",
})

_POOJA_KEYWORDS = frozenset({
    "pooja booking", "puja booking", "book pooja", "book puja",
    "online pooja", "online puja", "pooja online", "puja online",
    "seva booking", "book seva", "archana booking", "book archana",
    "pooja register", "puja register", "register pooja", "register puja",
    "पूजा बुकिंग", "पूजा बुक", "ऑनलाइन पूजा",
})


def booking_link_suffix(user_message: str, reply: str) -> str:
    """Append the booking link if the user asked about accommodation and it's missing from the reply."""
    msg_lower = user_message.lower()
    if any(kw in msg_lower for kw in _ACCOMMODATION_KEYWORDS) and BOOKING_LINK not in reply:
        return f"\n\n**Book official temple guesthouse:** {BOOKING_LINK}"
    return ""


def pooja_link_suffix(user_message: str, reply: str) -> str:
    """Append the pooja booking link if the user asked about pooja booking and it's missing from the reply."""
    msg_lower = user_message.lower()
    if any(kw in msg_lower for kw in _POOJA_KEYWORDS) and POOJA_LINK not in reply:
        return f"\n\nFor more info and booking, visit: {POOJA_LINK}"
    return ""


def _strip_maps_urls(text: str) -> str:
    return _MAPS_URL_RE.sub("", text).strip()


def _build_context(chunks: list[dict], structured: str) -> str:
    parts = []
    if chunks:
        parts.append("[Context from official Somnath Temple sources]")
        for c in chunks:
            heading = c.get("heading") or c.get("page_title") or "Note"
            content = _strip_maps_urls(c["content"])
            parts.append(f"**{heading}**\n{content}")
    if structured:
        parts.append("[Structured data from Somnath database]")
        parts.append(_strip_maps_urls(structured))
    return "\n\n".join(parts)


def _build_messages(
    user_message: str,
    history: list[dict],
    chunks: list[dict],
    structured: str,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    ctx = _build_context(chunks, structured)
    if ctx:
        messages.append({
            "role": "system",
            "content": (
                "CONTEXT — the retrieved information for this query. Every factual "
                "claim in your reply must come from this block or the AUTHORITATIVE "
                "FACTS in your instructions:\n\n" + ctx
            ),
        })
    else:
        # No data retrieved. Fence the generator to authoritative facts only so it
        # can't fill the gap from general knowledge (the "adds things on its own" case).
        messages.append({
            "role": "system",
            "content": (
                "CONTEXT: none was retrieved for this query. Answer ONLY from the "
                "AUTHORITATIVE FACTS in your instructions. If they do not cover the "
                "question, tell the user you don't have that specific detail and suggest "
                "checking somnath.org or contacting the temple office. Do not answer from "
                "general knowledge."
            ),
        })
    # Keep only the most recent turns so a long conversation can't overflow
    # num_ctx and truncate the system prompt / context off the front.
    if config.LLM_MAX_HISTORY_MSGS and len(history) > config.LLM_MAX_HISTORY_MSGS:
        history = history[-config.LLM_MAX_HISTORY_MSGS:]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    # Variable prompt payload (the static system prompt is omitted — it never changes).
    log_debug("llm_input", context=ctx, history=history, user=user_message)
    return messages


async def chat(
    user_message: str,
    history: list[dict],
    chunks: list[dict],
    structured: str = "",
) -> str:
    messages = _build_messages(user_message, history, chunks, structured)
    resp = await get_ollama_client().chat(
        model=config.LLM_MODEL,
        messages=messages,
        think=False,
        options=_LLM_OPTIONS,
    )
    return resp.message.content


async def chat_stream(
    user_message: str,
    history: list[dict],
    chunks: list[dict],
    structured: str = "",
) -> AsyncIterator[str]:
    messages = _build_messages(user_message, history, chunks, structured)
    async for chunk in await get_ollama_client().chat(
        model=config.LLM_MODEL,
        messages=messages,
        stream=True,
        think=False,
        options=_LLM_OPTIONS,
    ):
        delta = chunk.message.content
        if delta:
            yield delta
