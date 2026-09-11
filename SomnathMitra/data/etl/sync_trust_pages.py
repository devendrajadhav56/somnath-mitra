"""
Idempotent sync: raw.trust_pages -> clean.knowledge_chunks

raw.trust_pages holds one doc per crawled somnath.org page, each with a
`sections` array of {heading, heading_level, position, content: [str, ...]}.
Most sections turn out to be site-wide nav/sidebar/footer boilerplate that
repeats near-identically on almost every page (5 signatures repeat ~85/88
times), which would otherwise flood a RAG corpus with duplicate junk.

This script:
  - computes each section's (heading, content) signature across all pages
  - drops any signature repeated on more than BOILERPLATE_MIN_PAGES pages
  - drops near-empty leftover sections (likely stray links/icons)
  - upserts the rest as one chunk doc per section into clean.knowledge_chunks

Many pages (FAQ, history, darshan info, festivals, nearest-places, ...)
contribute *zero* sections-based chunks — their `sections` array only ever
matched boilerplate, even though `page_text` for those same pages holds real,
substantial content (thousands of characters). For any page with no kept
section chunks, this script falls back to `page_text`: strips out the known
boilerplate strings (collected while scanning sections) and splits what's
left into fixed-size chunks.

Safe to re-run whenever raw.trust_pages changes; upserts on chunk_id.
"""
import hashlib
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import get_clean_db, get_raw_db  # noqa: E402

BOILERPLATE_MIN_PAGES = 5
MIN_CONTENT_CHARS = 10
FALLBACK_CHUNK_CHARS = 900


def section_signature(section):
    key = (section.get("heading"), tuple(section.get("content") or []))
    return hashlib.md5(str(key).encode("utf-8")).hexdigest()

def page_slug(url):
    path = urlparse(url).path.strip("/")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", path).strip("_").lower()
    return slug or "home"


def find_boilerplate(pages):
    """Returns (boilerplate signature hashes, the boilerplate text strings themselves)."""
    counts = {}
    texts_by_sig = {}
    for page in pages:
        for section in (page.get("sections") or []):
            sig = section_signature(section)
            counts[sig] = counts.get(sig, 0) + 1
            texts_by_sig[sig] = section.get("content") or []

    boilerplate_sigs = {sig for sig, count in counts.items() if count > BOILERPLATE_MIN_PAGES}
    boilerplate_texts = set()
    for sig in boilerplate_sigs:
        for text in texts_by_sig[sig]:
            if text and text.strip():
                boilerplate_texts.add(text.strip())
    return boilerplate_sigs, boilerplate_texts


def strip_boilerplate_text(page_text, boilerplate_texts):
    cleaned = page_text or ""
    for text in boilerplate_texts:
        cleaned = cleaned.replace(text, " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def chunk_text(text, size):
    words = text.split(" ")
    chunks, current = [], []
    current_len = 0
    for word in words:
        current.append(word)
        current_len += len(word) + 1
        if current_len >= size:
            chunks.append(" ".join(current))
            current, current_len = [], 0
    if current:
        chunks.append(" ".join(current))
    return chunks


def run():
    raw_db = get_raw_db()
    clean_db = get_clean_db()

    pages = list(raw_db["trust_pages"].find({}))
    boilerplate_sigs, boilerplate_texts = find_boilerplate(pages)

    kept_sections, dropped_boilerplate, dropped_empty = 0, 0, 0
    kept_fallback, pages_with_no_sections = 0, 0

    for page in pages:
        url = page.get("source_url") or page.get("canonical_url")
        slug = page_slug(url)
        page_title = (page.get("title") or "").split("|")[0].strip() or None

        page_chunk_count = 0
        for section in (page.get("sections") or []):
            if section_signature(section) in boilerplate_sigs:
                dropped_boilerplate += 1
                continue

            content_list = section.get("content") or []
            content = "\n".join(c.strip() for c in content_list if c and c.strip())
            if len(content) < MIN_CONTENT_CHARS:
                dropped_empty += 1
                continue

            chunk_id = f"{slug}#{section.get('position')}"
            chunk = {
                "chunk_id": chunk_id,
                "page_url": url,
                "page_slug": slug,
                "page_title": page_title,
                "heading": section.get("heading"),
                "heading_level": section.get("heading_level"),
                "position": section.get("position"),
                "content": content,
                "word_count": len(content.split()),
                "scraped_at": page.get("scraped_at"),
                "source": "sections",
                "verified": False,
                "schema_version": 1,
            }
            clean_db["knowledge_chunks"].replace_one({"chunk_id": chunk_id}, chunk, upsert=True)
            kept_sections += 1
            page_chunk_count += 1

        if page_chunk_count == 0:
            pages_with_no_sections += 1
            cleaned_text = strip_boilerplate_text(page.get("page_text"), boilerplate_texts)
            for idx, piece in enumerate(chunk_text(cleaned_text, FALLBACK_CHUNK_CHARS)):
                if len(piece) < MIN_CONTENT_CHARS:
                    continue
                chunk_id = f"{slug}#text{idx}"
                chunk = {
                    "chunk_id": chunk_id,
                    "page_url": url,
                    "page_slug": slug,
                    "page_title": page_title,
                    "heading": page_title,
                    "heading_level": None,
                    "position": idx,
                    "content": piece,
                    "word_count": len(piece.split()),
                    "scraped_at": page.get("scraped_at"),
                    "source": "page_text_fallback",
                    "verified": False,
                    "schema_version": 1,
                }
                clean_db["knowledge_chunks"].replace_one({"chunk_id": chunk_id}, chunk, upsert=True)
                kept_fallback += 1

    print(
        f"synced {kept_sections + kept_fallback} knowledge chunks from {len(pages)} pages "
        f"({kept_sections} from sections, {kept_fallback} from page_text fallback across "
        f"{pages_with_no_sections} pages with no usable sections; "
        f"{dropped_boilerplate} boilerplate sections dropped across {len(boilerplate_sigs)} "
        f"signatures, {dropped_empty} near-empty sections dropped)"
    )


if __name__ == "__main__":
    run()
