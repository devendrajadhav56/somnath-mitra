"""In-memory vector index over clean.knowledge_chunks.

Embeddings are stored in MongoDB (field: `embedding`). On startup this module
loads all embedded chunks into a numpy matrix for fast cosine similarity search.
Call `reload()` after running embed_knowledge_chunks.py to refresh without restart.
"""
from __future__ import annotations

import numpy as np

import config
import db
from services.embedder import embed_one

_chunk_ids: list[str] = []
_chunks: list[dict] = []
_matrix: np.ndarray | None = None  # shape (N, dim), L2-normalised


def reload() -> int:
    """Load/refresh the in-memory index from MongoDB. Returns number of chunks indexed."""
    global _chunk_ids, _chunks, _matrix

    clean = db.get_clean_db()
    docs = list(clean["knowledge_chunks"].find(
        {"embedding": {"$exists": True}},
        {"chunk_id": 1, "page_title": 1, "heading": 1, "content": 1, "page_url": 1, "embedding": 1},
    ))

    if not docs:
        _chunk_ids, _chunks, _matrix = [], [], None
        return 0

    _chunk_ids = [d["chunk_id"] for d in docs]
    _chunks = [{k: v for k, v in d.items() if k != "embedding"} for d in docs]
    _matrix = np.array([d["embedding"] for d in docs], dtype="float32")
    return len(docs)


def search(query: str, top_k: int | None = None) -> list[dict]:
    """Return top-k chunk dicts (with a `score` field added) for the query."""
    if _matrix is None or len(_chunks) == 0:
        return []

    k = top_k or config.TOP_K_CHUNKS
    q = embed_one(query).reshape(1, -1)  # (1, dim), already L2-normalised
    scores = (_matrix @ q.T).flatten()   # cosine sim = dot product on normalised vecs

    top_indices = np.argpartition(scores, -min(k, len(scores)))[-min(k, len(scores)):]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    return [
        {**_chunks[i], "score": float(scores[i])}
        for i in top_indices
    ]
