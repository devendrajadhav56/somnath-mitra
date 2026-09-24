"""
One-time fix for the misaligned Shree Somnath Trust admin/trustee chunks.

The scraper produced off-by-one chunks: each chunk's `heading` was the PREVIOUS
person's name while its `content` held the NEXT person + title. Since the
embedding text is "heading\ncontent", the garbage headings polluted retrieval,
and the fragmented one-person-per-chunk layout gave incoherent answers to
"who is the general manager / secretary / trustee" queries.

The `content` lines were internally correct ("<Name> <Title>"), and the roster
below is reconstructed from them (the GM is independently confirmed by the
contact page: "Shri Vijaysinh Chavda / General Manager / Shree Somnath Trust").

This script replaces the admin_team and trustee chunks with one clean, correctly
labelled chunk each, then re-embeds them via bge-m3. Idempotent: safe to re-run
(deletes by page_slug, re-inserts, re-embeds).

Run:
    ../backend/venv/bin/python etl/fix_admin_chunks.py
Then reload the retriever:
    curl -X POST http://localhost:7777/retriever/reload
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import get_clean_db  # noqa: E402

EMBED_MODEL = "bge-m3"
EMBED_BASE_URL = "http://localhost:11434/v1"

# Reconstructed rosters (title: name), from the correct `content` lines.
ADMIN_TEAM = [
    ("Secretary", "Shri Yogendra Desai"),
    ("General Manager", "Shri Vijaysinh Chavda"),
    ("Executive Officer", "Shri Dilip Chavda"),
]

TRUSTEES = [
    "Shri Narendrabhai Modi",
    "Shri Lal Krishna Advani",
    "Shri Amitbhai Shah",
    "Shri J. D. Parmar",
    "Shri Harshvardhan Neotia",
    "Shri Pravin K. Laheri",
    "Shri Vishad Mafatlal",
]

SOURCE_URL = "https://somnath.org/Administrative-Team"


def _admin_content() -> str:
    lines = ["Shree Somnath Trust — Administrative Team (office bearers):"]
    lines += [f"- {title}: {name}" for title, name in ADMIN_TEAM]
    return "\n".join(lines)


def _trustee_content() -> str:
    lines = ["Shree Somnath Trust — Board of Trustees:"]
    lines += [f"- {name} (Trustee)" for name in TRUSTEES]
    return "\n".join(lines)


def _embed(client, text: str) -> list[float]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    arr = np.array(resp.data[0].embedding, dtype="float32")
    arr = arr / max(float(np.linalg.norm(arr)), 1e-9)
    return arr.tolist()


def run():
    from openai import OpenAI

    clean = get_clean_db()
    col = clean["knowledge_chunks"]
    client = OpenAI(api_key="ollama", base_url=EMBED_BASE_URL)

    chunks = [
        {
            "chunk_id": "admin_team#roster",
            "page_url": SOURCE_URL, "page_slug": "admin_team",
            "page_title": "Administrative Team", "heading": "Administrative Team",
            "content": _admin_content(),
        },
        {
            "chunk_id": "trustee#roster",
            "page_url": SOURCE_URL, "page_slug": "trustee",
            "page_title": "Board of Trustees", "heading": "Board of Trustees",
            "content": _trustee_content(),
        },
    ]

    # Remove the old misaligned chunks for these pages.
    removed = col.delete_many({"page_slug": {"$in": ["admin_team", "trustee"]}}).deleted_count

    for c in chunks:
        c["heading_level"] = None
        c["position"] = 0
        c["source"] = "manual_fix"
        c["verified"] = True
        c["schema_version"] = 2
        c["word_count"] = len(c["content"].split())
        embed_text = f"{c['heading']}\n{c['content']}".strip()
        c["embedding"] = _embed(client, embed_text)
        col.replace_one({"chunk_id": c["chunk_id"]}, c, upsert=True)
        print(f"  wrote {c['chunk_id']} ({c['word_count']} words, embedded)")

    print(f"removed {removed} misaligned chunks, inserted {len(chunks)} clean chunks.")
    print("NEXT: curl -X POST http://localhost:7777/retriever/reload")


if __name__ == "__main__":
    run()
