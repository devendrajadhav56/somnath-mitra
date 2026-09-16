"""Async Mistral client (OpenAI-compatible API)."""
from __future__ import annotations

from typing import AsyncIterator

from openai import AsyncOpenAI

import config

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    return _client


SYSTEM_PROMPT = """\
You are Somnath Mitra, a knowledgeable and friendly assistant for pilgrims and visitors \
to the Somnath Jyotirlinga temple in Prabhas Patan, Gujarat, India.

Answer questions about the temple's history, darshan timings, aarti schedule, visitor rules, \
nearby attractions, accommodation, restaurants, and points of interest.

Use the provided context to give accurate, grounded answers. If the context does not cover \
the question, say so honestly rather than guessing. Keep answers concise and practical. \
Respond in the same language the user writes in.\
"""


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
    resp = await get_client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        stream=False,
    )
    return resp.choices[0].message.content


async def chat_stream(
    user_message: str,
    history: list[dict],
    chunks: list[dict],
    structured: str = "",
) -> AsyncIterator[str]:
    messages = _build_messages(user_message, history, chunks, structured)
    stream = await get_client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        stream=True,
        temperature=0.2,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
