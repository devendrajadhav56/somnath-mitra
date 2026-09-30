"""
Seed hand-authored RAG chunks for key nearby pilgrimage places that were
missing coherent coverage in knowledge_chunks: Bhalka Tirth, Triveni Sangam,
Shri Ram Mandir, and Ahilyabai (Old Somnath) Temple.

One coherent chunk per place (distance + significance + travel + timings) so a
single query returns a complete answer rather than fragments — same rationale as
fix_admin_chunks.py. Embeds via bge-m3 and upserts by chunk_id. Idempotent:
safe to re-run.

Run:
    ../backend/venv/bin/python etl/seed_nearby_place_chunks.py
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
SOURCE_URL = "https://somnath.org/Nearest-Places"

# Each entry: slug, heading (place name + intent keywords for retrieval), content.
# The place name is repeated inside content, and query words users actually type
# ("distance", "how far", "timings", "entry fee", "how to reach") are included
# so short queries clear RAG_MIN_SCORE (0.3).
PLACES = [
    {
        "slug": "bhalka_tirth",
        "heading": "Bhalka Tirth — significance, distance from Somnath, timings",
        "content": (
            "Bhalka Tirth is a sacred site about 4-5 km from the Somnath Mandir, "
            "located to the northwest on the Prabhas-Veraval highway.\n"
            "Significance: it marks the place of Lord Krishna's Antim Leela (his final "
            "earthly act). As per Hindu mythology, while Krishna was meditating under a "
            "Peepal tree, a hunter named Jara mistook his foot for a deer and shot a "
            "poisonous arrow. This event signifies the end of Dwapara Yuga and the dawn "
            "of Kali Yuga. The temple has a distinctive semi-reclining marble idol of "
            "Lord Krishna.\n"
            "Timings: open 6:00 AM to 9:00 PM daily. There is no entry fee."
        ),
    },
    {
        "slug": "triveni_sangam",
        "heading": "Triveni Sangam (Triveni Ghat) — significance, distance from Somnath, timings, how to reach",
        "content": (
            "Triveni Sangam, also called Triveni Ghat, is about 1 to 1.2 km from the "
            "Somnath Mandir — easily reachable on foot or by a short drive.\n"
            "Significance: it is the holy confluence (sangam) where three sacred rivers — "
            "Hiran, Kapila, and Saraswati — meet before joining the Arabian Sea. It holds "
            "deep cosmic value in Hinduism, representing the cycle of birth, life, and "
            "death. Pilgrims take holy dips here to cleanse sins, and it is a paramount "
            "site for ancestral rites such as Shraddha, Tarpan, and Pind Daan.\n"
            "How to reach: a 10-15 minute walk, local auto-rickshaws, or the Somnath "
            "Trust hop-on bus service.\n"
            "Timings: open 6:00 AM to 6:00 PM daily."
        ),
    },
    {
        "slug": "shri_ram_mandir",
        "heading": "Shri Ram Mandir — significance, distance from Somnath, timings, how to reach",
        "content": (
            "Shri Ram Mandir is about 1 km from the Somnath Mandir, situated right next "
            "to the Triveni Sangam Ghat.\n"
            "Significance: built by the Shree Somnath Trust in 2017 using Bansi Paharpur "
            "pink sandstone, it is the newest grand temple in the complex. Dedicated to "
            "Lord Rama, Sita Devi, and Lakshmana, it features marble idols, a distinctive "
            "bow-shaped (dhanushya) entrance gate, and an attached auditorium that screens "
            "shows on the life of Shri Ram.\n"
            "How to reach: walking from the main Somnath temple area, auto-rickshaws, or "
            "local taxis.\n"
            "Timings: open 6:00 AM to 7:00 PM daily."
        ),
    },
    {
        "slug": "ahilyabai_temple",
        "heading": "Ahilyabai Mandir (Old Somnath Temple) — significance, distance from Somnath, timings",
        "content": (
            "Ahilyabai Mandir, also known as the Old Somnath Temple, sits directly "
            "opposite the main Somnath Temple complex — less than 100 to 200 metres away.\n"
            "Significance: built in the late 18th century by the Maratha queen Maharani "
            "Ahilyabai Holkar of Indore. When the main historical temple was repeatedly "
            "targeted by invaders, this temple was constructed adjacent to it as a safe "
            "alternative to keep the prayers active. Many devotees believe it houses the "
            "original Swayambhu Jyotirlinga, hidden safely in an underground sanctum to "
            "protect it from desecration.\n"
            "How to reach: walking only — it is immediately opposite the entry points of "
            "the main shrine.\n"
            "Timings: matches the main temple, 6:00 AM to 9:00 PM daily."
        ),
    },
]


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

    for p in PLACES:
        doc = {
            "chunk_id": f"nearby#{p['slug']}",
            "page_url": SOURCE_URL,
            "page_slug": f"nearby_{p['slug']}",
            "page_title": p["heading"].split(" — ")[0].split(" (")[0],
            "heading": p["heading"],
            "heading_level": None,
            "position": 0,
            "content": p["content"],
            "source": "manual_seed",
            "verified": True,
            "schema_version": 2,
            "word_count": len(p["content"].split()),
        }
        embed_text = f"{doc['heading']}\n{doc['content']}".strip()
        doc["embedding"] = _embed(client, embed_text)
        col.replace_one({"chunk_id": doc["chunk_id"]}, doc, upsert=True)
        print(f"  wrote {doc['chunk_id']} ({doc['word_count']} words, embedded)")

    print(f"seeded {len(PLACES)} nearby-place chunks.")
    print("NEXT: curl -X POST http://localhost:7777/retriever/reload")


if __name__ == "__main__":
    run()
