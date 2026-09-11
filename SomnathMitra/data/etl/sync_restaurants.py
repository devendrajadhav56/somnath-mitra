"""
Idempotent sync: raw.restaurants -> clean.restaurants (+ restaurants_excluded)

- geo-filters to within ~60km of the temple; anything farther goes to
  restaurants_excluded instead of clean.restaurants (the raw data reaches
  as far as Ahmedabad, ~400km away)
- nulls out `category` when it's just scraped hours/plus-code noise, kept
  verbatim in `category_raw`
- dedups by place_id/cid, keeping the doc with the most reviews
- leaves rating/reviews as null rather than defaulting to 0
Safe to re-run any time raw.restaurants changes; upserts on source_id.
"""
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TEMPLE_LAT, TEMPLE_LON  # noqa: E402
from db import get_clean_db, get_raw_db  # noqa: E402
from schemas.restaurant import Restaurant  # noqa: E402

MAX_DISTANCE_KM = 60

PLUS_CODE_RE = re.compile(r"\b[23456789CFGHJMPQRVWX]{4,8}\+[23456789CFGHJMPQRVWX0-9]{2,3}\b")
NOISE_PREFIXES = ("closes", "closed", "open", "opens")


def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def clean_category(raw_category):
    if not raw_category:
        return None
    low = raw_category.strip().lower()
    if low.startswith(NOISE_PREFIXES):
        return None
    if PLUS_CODE_RE.search(raw_category):
        return None
    return raw_category


def dedup_by_place(docs):
    """Group by place_id/cid (the real Google identifier); keep the doc with most reviews."""
    by_place = {}
    for doc in docs:
        key = doc.get("place_id") or doc.get("cid") or str(doc["_id"])
        by_place.setdefault(key, []).append(doc)

    for group in by_place.values():
        group.sort(key=lambda d: (-(d.get("reviews") or 0), str(d["_id"])))
        primary, *dupes = group
        yield primary, dupes


def run():
    raw_db = get_raw_db()
    clean_db = get_clean_db()

    raw_docs = list(raw_db["restaurants"].find({}))

    kept, excluded = 0, 0
    for primary, dupes in dedup_by_place(raw_docs):
        lon, lat = primary["location"]["coordinates"]
        distance_km = haversine_km(TEMPLE_LON, TEMPLE_LAT, lon, lat)

        clean_doc = {
            "source_id": str(primary["_id"]),
            "name": primary.get("name"),
            "category": clean_category(primary.get("category")),
            "category_raw": primary.get("category"),
            "rating": primary.get("rating"),
            "reviews": primary.get("reviews"),
            "has_website": primary.get("has_website"),
            "place_id": primary.get("place_id"),
            "cid": primary.get("cid"),
            "google_maps_url": primary.get("google_maps_url"),
            "location": primary["location"],
            "scraped_at": primary.get("scraped_at"),
            "merged_from": [str(d["_id"]) for d in dupes],
            "distance_from_temple_km": round(distance_km, 2),
            "schema_version": 1,
        }

        validated = Restaurant.model_validate(clean_doc).model_dump()
        target = "restaurants" if distance_km <= MAX_DISTANCE_KM else "restaurants_excluded"
        clean_db[target].replace_one({"source_id": validated["source_id"]}, validated, upsert=True)
        if target == "restaurants":
            kept += 1
        else:
            excluded += 1

    print(
        f"synced {kept} clean restaurants, excluded {excluded} beyond "
        f"{MAX_DISTANCE_KM}km ({len(raw_docs)} raw docs total)"
    )


if __name__ == "__main__":
    run()
