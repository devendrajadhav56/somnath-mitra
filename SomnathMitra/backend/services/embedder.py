"""Embeddings via ollama's OpenAI-compatible API (bge-m3, 1024-dim)."""
from __future__ import annotations

import numpy as np
from openai import OpenAI

import config

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key="ollama", base_url=config.EMBED_BASE_URL)
    return _client


def warmup() -> None:
    """Fire a test embedding at startup to verify the endpoint is reachable."""
    embed_one("warmup")
    print(f"[embedder] {config.EMBED_MODEL} ready via {config.EMBED_BASE_URL}")


def embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalised float32 embeddings, shape (len(texts), 1024)."""
    resp = _get_client().embeddings.create(model=config.EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]
