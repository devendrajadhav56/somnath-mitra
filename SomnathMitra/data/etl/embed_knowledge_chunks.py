"""
Compute and store bge-m3 embeddings (1024-dim) for every knowledge_chunk
that doesn't have one yet (or all, if --recompute is passed).

Uses ollama's OpenAI-compatible embeddings API — no local model load needed.

Run with:
    ../backend/venv/bin/python etl/embed_knowledge_chunks.py
    ../backend/venv/bin/python etl/embed_knowledge_chunks.py --recompute
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import get_clean_db  # noqa: E402

EMBED_MODEL = "bge-m3"
EMBED_BASE_URL = "http://localhost:11434/v1"
BATCH_SIZE = 32  # conservative for ollama


def _get_client():
    from openai import OpenAI
    return OpenAI(api_key="ollama", base_url=EMBED_BASE_URL)


def _embed_batch(client, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def run(recompute: bool = False):
    clean = get_clean_db()
    query = {} if recompute else {"embedding": {"$exists": False}}
    docs = list(clean["knowledge_chunks"].find(query, {"chunk_id": 1, "heading": 1, "content": 1}))

    if not docs:
        print("all chunks already embedded — pass --recompute to force refresh")
        return

    print(f"embedding {len(docs)} chunks with {EMBED_MODEL} via {EMBED_BASE_URL}...")
    client = _get_client()
    texts = [f"{d.get('heading') or ''}\n{d['content']}".strip() for d in docs]

    total = 0
    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[batch_start: batch_start + BATCH_SIZE]
        batch_docs = docs[batch_start: batch_start + BATCH_SIZE]

        vecs = _embed_batch(client, batch_texts)

        for doc, vec in zip(batch_docs, vecs):
            # L2-normalise before storing
            arr = np.array(vec, dtype="float32")
            arr = arr / max(float(np.linalg.norm(arr)), 1e-9)
            clean["knowledge_chunks"].update_one(
                {"chunk_id": doc["chunk_id"]},
                {"$set": {"embedding": arr.tolist()}},
            )

        total += len(batch_docs)
        print(f"  {total}/{len(docs)}", end="\r", flush=True)

    print(f"\ndone — embedded {total} chunks with {EMBED_MODEL} (1024-dim)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true", help="re-embed all chunks, not just missing ones")
    args = parser.parse_args()
    run(recompute=args.recompute)
