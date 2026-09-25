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

# ── Board of Trustees ──────────────────────────────────────────────────────────
# Eight designated trustee slots (Central Govt of India + State Govt of Gujarat
# nominations); seven are named below. Chairman's title is set explicitly (it was
# lost in the source off-by-one).
CHAIRMAN = (
    "Shri Narendra Modi",
    "Chairman & Trustee — Prime Minister of India. Serves a 5-year tenure as chairman; "
    "the second Prime Minister to hold this position, after Morarji Desai.",
)
TRUSTEES = [
    ("Shri Lal Krishna (L.K.) Advani",
     "Veteran Bharatiya Janata Party (BJP) leader and former Deputy Prime Minister of India; "
     "a long-serving central nominee on the board."),
    ("Shri Amitbhai Shah",
     "Union Home Minister and Minister of Cooperation of India."),
    ("Shri Pravin K. Laheri",
     "Retired senior IAS officer and former Chief Secretary of Gujarat."),
    ("Shri J. D. Parmar",
     "Scholar and retired professor of Sanskrit from Veraval, Gujarat; an expert on "
     "religious and cultural traditions."),
    ("Shri Harshvardhan Neotia",
     "Industrialist; chairman of the Kolkata-based Ambuja Neotia Group."),
    ("Shri Vishad Mafatlal",
     "A leading Indian industrialist from the Mafatlal Group."),
]

# ── Key administrative officials (day-to-day operations) ────────────────────────
OFFICIALS = [
    ("Shri Vijaysinh Chavda", "General Manager",
     "Manages ground operations, security, and administrative decisions at the temple "
     "complex in Prabhas Patan (Veraval)."),
    ("Shri Dilipbhai Chavda", "Executive Officer",
     "Directs administrative affairs, overseeing the trust's office based in Ahmedabad."),
    ("Shri Yashodharbhai Bhatt", "Donation & Fund Officer",
     "Handles financial contributions."),
    ("Shri Dhanjaybhai Dave", "Main Priest (Chief Pujari)",
     "Manages the daily Vedic rituals and aartis."),
]

SOURCE_URL = "https://somnath.org/Administrative-Team"


def _admin_content() -> str:
    lines = ["Shree Somnath Trust — Key Administrative Officials (day-to-day operations):"]
    lines += [f"- {role}: {name} — {bio}" for name, role, bio in OFFICIALS]
    return "\n".join(lines)


def _trustee_content() -> str:
    lines = [
        "Shree Somnath Trust — Chairman and Board of Trustees.",
        "The board has eight designated trustee slots, with nominations from the Central "
        "Government of India and the State Government of Gujarat; the named members are:",
        f"- Chairman: {CHAIRMAN[0]} — {CHAIRMAN[1]}",
    ]
    lines += [f"- {name} (Trustee): {bio}" for name, bio in TRUSTEES]
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
            "page_title": "Key Administrative Officials",
            "heading": "Key Administrative Officials",
            "content": _admin_content(),
        },
        {
            "chunk_id": "trustee#roster",
            "page_url": SOURCE_URL, "page_slug": "trustee",
            "page_title": "Chairman and Board of Trustees",
            "heading": "Chairman and Board of Trustees",
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
