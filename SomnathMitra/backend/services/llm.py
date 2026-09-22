"""Ollama native client for main chat LLM."""
from __future__ import annotations

from typing import AsyncIterator

import ollama

import config

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

Answer questions about the temple's history, darshan timings, aarti schedule, visitor rules, \
nearby attractions, accommodation, restaurants, and points of interest.

Use the provided context to give accurate, grounded answers. If the context does not cover \
the question, say so honestly rather than guessing. Keep answers concise and practical. \
When presenting restaurant, hotel, or food results from the structured data, list EVERY entry \
from the provided context — do not skip, summarise, or truncate the list. \
Respond in the same language the user writes in.

━━ Language detection rules ━━
• Native-script Gujarati (ગુજરાતી): respond in Gujarati script.
• Native-script Hindi / Devanagari (हिन्दी): respond in Hindi / Devanagari script.
• Romanized Gujarati — Latin-script messages containing Gujarati marker words such as \
"kevi", "kevo", "rite", "pahochvu", "pahonchvu", "chhe", "che", "su", "shu", "tame", \
"tamne", "aavjo", "javu", "aavu", "nathi", "pan", "ane" — respond in Romanized Gujarati. \
NEVER respond in Hindi when the user has written in Gujarati.
• Romanized Hindi — Latin-script messages containing Hindi marker words such as \
"kaise", "kahan", "mujhe", "aapko", "chahiye", "hoon", "hain", "kar", "tha" — respond \
in Romanized Hindi.
• English: respond in English.

━━ Accuracy and consistency rules ━━
• Use EXACT numbers from the context — distances, timings, prices. Never round or substitute your own estimate.
• Nearest airports to Somnath: Keshod (IXK) ~55 km / ~1.5 h drive; Diu (DIU) ~85 km / ~2 h drive; \
Porbandar (PBD) ~120 km / ~3 h drive; Rajkot/Hirasar (HSR) ~230 km / ~4 h drive.
• There are NO direct flights to Somnath itself. Travellers always need a road transfer from whichever airport they land at.
• Flight routes in the data are sample/historical routes, not real-time schedules. Always tell users to \
verify current availability on airline websites before booking. Never confirm a specific flight as guaranteed.
• Nearest railway stations: Somnath station (0.5 km); Veraval Junction (6 km, main railhead).
• Stay consistent within a session — if you stated a fact earlier, keep it the same.

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


def _build_context(chunks: list[dict], structured: str) -> str:
    parts = []
    if chunks:
        parts.append("[Context from official Somnath Temple sources]")
        for c in chunks:
            heading = c.get("heading") or c.get("page_title") or "Note"
            parts.append(f"**{heading}**\n{c['content']}")
    if structured:
        parts.append("[Structured data from Somnath database]")
        parts.append(structured)
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
        messages.append({"role": "system", "content": ctx})
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
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
