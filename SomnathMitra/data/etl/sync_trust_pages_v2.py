"""
Improved sync: raw.trust_pages -> clean.knowledge_chunks   (v2)

Rewrites the chunking policy to fix the issues found in the RAG audit:

  * Over-fragmentation — 34 chunks were < 100 chars (org-chart / contact lists
    split one-per-person). v2 MERGES consecutive tiny sections into a single
    coherent chunk instead of emitting fragments.
  * One 19,908-char monster — section chunks had no size cap (only the fallback
    path was chunked). v2 SPLITS any oversized section into ~TARGET-sized pieces
    with word overlap.
  * Low-value pages — privacy policy, tenders, receipts polluted the corpus.
    v2 DENYLISTS them.
  * Broken provenance — every chunk had page_title "Jay Somnath" (the site-wide
    <title>). v2 derives a readable title from the page slug.
  * No overlap in the fallback splitter — v2 adds ~OVERLAP_WORDS overlap so
    facts spanning a chunk boundary aren't lost.

Chunking strategy per page:
  - Substantial sections (>= TINY_MAX chars) are kept whole (topical coherence),
    or split with overlap if they exceed HARD_MAX.
  - Tiny sections are accumulated and merged, flushing at ~TARGET chars, so a
    run of one-line entries becomes one readable chunk.

Run modes:
  python sync_trust_pages_v2.py --dry-run   # compute + print stats, NO db writes
  python sync_trust_pages_v2.py             # full rebuild: delete all chunks,
                                            # re-insert (then run embed_knowledge_chunks.py)

NOTE: a real run does a FULL REBUILD (delete_many({}) then insert) because the
chunk_id scheme changes — this clears stale chunks and their embeddings, so you
must re-run embed_knowledge_chunks.py afterwards.
"""
import argparse
import hashlib
import re
import statistics
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import get_clean_db, get_raw_db  # noqa: E402

# ── Tunables ──────────────────────────────────────────────────────────────────
BOILERPLATE_MIN_PAGES = 5      # a section repeated on more pages than this is nav/footer
MIN_CONTENT_CHARS = 10         # drop sections with less real text than this
TINY_MAX = 250                 # sections shorter than this are merged, not emitted alone
TARGET_CHARS = 700             # soft target size for a chunk
HARD_MAX_CHARS = 1300          # sections longer than this are split into sub-chunks
OVERLAP_WORDS = 40             # word overlap between split sub-chunks
FALLBACK_CHUNK_CHARS = 800     # target size for page_text fallback chunks

# Utility / low-value pages that add noise rather than answers.
DENYLIST_SLUGS = {
    "privacy_policy",
    "tenders",
    "reprint_pooja_receipt",
    "online_donation",
    "sitemap",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def section_signature(section):
    key = (section.get("heading"), tuple(section.get("content") or []))
    return hashlib.md5(str(key).encode("utf-8")).hexdigest()


def page_slug(url):
    path = urlparse(url).path.strip("/")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", path).strip("_").lower()
    return slug or "home"


def humanize_slug(slug):
    """Readable page title derived from the slug (the raw <title> is useless)."""
    return re.sub(r"_+", " ", slug).strip().title() or None


def find_boilerplate(pages):
    counts, texts_by_sig = {}, {}
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


def chunk_text(text, size=TARGET_CHARS, overlap_words=OVERLAP_WORDS):
    """Split text into ~size-char chunks with word overlap between them."""
    words = text.split()
    if not words:
        return []
    chunks, i = [], 0
    while i < len(words):
        cur, cur_len, j = [], 0, i
        while j < len(words) and cur_len < size:
            cur.append(words[j])
            cur_len += len(words[j]) + 1
            j += 1
        chunks.append(" ".join(cur))
        if j >= len(words):
            break
        i = max(i + 1, j - overlap_words)   # advance, keeping overlap; never stall
    return chunks


def clean_section_content(section):
    content_list = section.get("content") or []
    return "\n".join(c.strip() for c in content_list if c and c.strip())


def build_page_chunks(sections, boilerplate_sigs):
    """Turn a page's sections into a list of (heading, content) chunks.

    Substantial sections are kept whole (or split if oversized); tiny sections
    are merged into ~TARGET-sized coherent chunks.
    """
    out = []
    tiny_buf, tiny_len = [], 0

    def flush_tiny():
        nonlocal tiny_buf, tiny_len
        if not tiny_buf:
            return
        if len(tiny_buf) == 1:
            h, c = tiny_buf[0]
            out.append((h, c))
        else:
            heading = tiny_buf[0][0]
            segs = [f"{h}: {c}" if h else c for h, c in tiny_buf]
            out.append((heading, "\n".join(segs)))
        tiny_buf, tiny_len = [], 0

    for section in sections:
        if section_signature(section) in boilerplate_sigs:
            continue
        content = clean_section_content(section)
        if len(content) < MIN_CONTENT_CHARS:
            continue
        heading = section.get("heading")

        if len(content) < TINY_MAX:
            tiny_buf.append((heading, content))
            tiny_len += len(content)
            if tiny_len >= TARGET_CHARS:
                flush_tiny()
            continue

        # Substantial section — flush pending tiny fragments first.
        flush_tiny()
        if len(content) > HARD_MAX_CHARS:
            for sub in chunk_text(content):
                out.append((heading, sub))
        else:
            out.append((heading, content))

    flush_tiny()
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def run(dry_run=False):
    raw_db = get_raw_db()
    clean_db = get_clean_db()

    pages = list(raw_db["trust_pages"].find({}))
    boilerplate_sigs, boilerplate_texts = find_boilerplate(pages)

    all_chunks = []          # dicts ready to insert
    skipped_denylist = 0
    fallback_pages = 0

    for page in pages:
        url = page.get("source_url") or page.get("canonical_url")
        slug = page_slug(url)
        if slug in DENYLIST_SLUGS:
            skipped_denylist += 1
            continue

        page_title = humanize_slug(slug)
        sections = page.get("sections") or []
        chunks = build_page_chunks(sections, boilerplate_sigs)

        if not chunks:
            # Fallback: no usable sections — chunk the raw page_text with overlap.
            fallback_pages += 1
            cleaned = strip_boilerplate_text(page.get("page_text"), boilerplate_texts)
            for idx, piece in enumerate(chunk_text(cleaned, FALLBACK_CHUNK_CHARS)):
                if len(piece) < MIN_CONTENT_CHARS:
                    continue
                all_chunks.append({
                    "chunk_id": f"{slug}#text{idx}",
                    "page_url": url, "page_slug": slug, "page_title": page_title,
                    "heading": page_title, "heading_level": None, "position": idx,
                    "content": piece, "word_count": len(piece.split()),
                    "scraped_at": page.get("scraped_at"),
                    "source": "page_text_fallback", "verified": False, "schema_version": 2,
                })
            continue

        for idx, (heading, content) in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{slug}#c{idx}",
                "page_url": url, "page_slug": slug, "page_title": page_title,
                "heading": heading, "heading_level": None, "position": idx,
                "content": content, "word_count": len(content.split()),
                "scraped_at": page.get("scraped_at"),
                "source": "sections", "verified": False, "schema_version": 2,
            })

    _report(all_chunks, len(pages), skipped_denylist, fallback_pages)

    if dry_run:
        print("\n[dry-run] no database writes performed.")
        return

    # Full rebuild — chunk_id scheme changed, so clear stale chunks + embeddings.
    result = clean_db["knowledge_chunks"].delete_many({})
    if all_chunks:
        clean_db["knowledge_chunks"].insert_many(all_chunks)
    print(f"\ndeleted {result.deleted_count} old chunks, inserted {len(all_chunks)} new chunks.")
    print("NEXT: run embed_knowledge_chunks.py to embed the new chunks, then hit /retriever/reload.")


def _report(chunks, n_pages, skipped_denylist, fallback_pages):
    lengths = sorted(len(c["content"]) for c in chunks)
    print(f"pages processed        : {n_pages}  (denylisted {skipped_denylist}, fallback {fallback_pages})")
    print(f"chunks produced        : {len(chunks)}")
    if not lengths:
        return
    def pct(p): return lengths[min(len(lengths) - 1, int(p * len(lengths)))]
    print(f"content length (chars) : min {lengths[0]} | median {int(statistics.median(lengths))} "
          f"| p90 {pct(.9)} | max {lengths[-1]} | mean {int(statistics.mean(lengths))}")
    b = lengths
    def cnt(lo, hi): return sum(1 for x in b if lo <= x < hi)
    print(f"size buckets           : <100:{cnt(0,100)}  100-500:{cnt(100,500)}  "
          f"500-1500:{cnt(500,1500)}  1500-4000:{cnt(1500,4000)}  >4000:{sum(1 for x in b if x>=4000)}")
    print("\nsample merged/first chunks:")
    for c in chunks[:5]:
        print(f"  [{c['source']}] {str(c['heading'])[:40]!r} ({len(c['content'])} chars): "
              f"{c['content'][:120].replace(chr(10),' ')!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="compute + print stats, no DB writes")
    args = ap.parse_args()
    run(dry_run=args.dry_run)
