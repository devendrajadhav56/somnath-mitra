"""
Load GMS (Google Maps Scraper) data from restaurant_hospital_pharmacy_data.zip
into clean DB collections: restaurants, hospitals, pharmacies.

Dedup strategy: upsert on source_id = CID extracted from Google Maps URL.
  - Existing restaurants are UPDATED (better category, phone, hours added).
  - New restaurants/hospitals/pharmacies are INSERTED.
  - Records outside MAX_DISTANCE_KM are skipped.

Safe to re-run any time — fully idempotent.

Usage:
    python load_gms_data.py [path/to/zip]
    # defaults to ../../data/restaurant_hospital_pharmacy_data.zip
"""
from __future__ import annotations

import json
import math
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import TEMPLE_LAT, TEMPLE_LON
from db import get_clean_db

MAX_DISTANCE_KM = 30

# Map GMS file name prefix → target clean collection
FILE_TO_COLLECTION = {
    "restaurant": "restaurants",
    "hospital":   "hospitals",
    "pharmacy":   "pharmacies",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _parse_url(url: str | None) -> dict:
    """Extract cid, lat, lon from a Google Maps URL."""
    if not url:
        return {}
    cid   = re.search(r"!1s(0x[0-9a-f]+:0x[0-9a-f]+)", url)
    lat   = re.search(r"!3d([-\d.]+)", url)
    lon   = re.search(r"!4d([-\d.]+)", url)
    return {
        "cid": cid.group(1) if cid else None,
        "lat": float(lat.group(1)) if lat else None,
        "lon": float(lon.group(1)) if lon else None,
    }


def _parse_rating(raw: str | None) -> float | None:
    try:
        return float(raw) if raw else None
    except (ValueError, TypeError):
        return None


def _parse_reviews(raw: str | None) -> int | None:
    if not raw:
        return None
    digits = re.sub(r"[^\d]", "", str(raw))
    return int(digits) if digits else None


def _to_doc(record: dict, collection: str) -> dict | None:
    """Convert a raw GMS record to a clean document. Returns None if unusable."""
    url_info = _parse_url(record.get("Google Maps URL"))
    cid = url_info.get("cid")
    lat = url_info.get("lat")
    lon = url_info.get("lon")

    if not cid or lat is None or lon is None:
        return None

    distance_km = _haversine_km(TEMPLE_LAT, TEMPLE_LON, lat, lon)
    if distance_km > MAX_DISTANCE_KM:
        return None

    return {
        "source_id":             cid,
        "cid":                   cid,
        "name":                  record.get("Name") or "",
        "category":              record.get("Category") or None,
        "phone":                 record.get("Phone") or None,
        "address":               record.get("Address") or None,
        "website":               record.get("Website") or None,
        "business_status":       record.get("Business Status") or None,
        "hours_raw":             record.get("Hours") or None,
        "rating":                _parse_rating(record.get("Rating")),
        "reviews":               _parse_reviews(record.get("Total Reviews")),
        "google_maps_url":       record.get("Google Maps URL") or None,
        "location": {
            "type": "Point",
            "coordinates": [lon, lat],
        },
        "distance_from_temple_km": round(distance_km, 2),
        "data_source":           "gms",
        "schema_version":        2,
    }


def _ensure_indexes(col) -> None:
    existing = col.index_information()
    if "location_2dsphere" not in existing:
        col.create_index([("location", "2dsphere")], name="location_2dsphere")
    if "source_id_1" not in existing:
        col.create_index("source_id", unique=True, name="source_id_1")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(zip_path: Path) -> None:
    clean_db = get_clean_db()
    counters: dict[str, dict] = {c: {"upserted": 0, "skipped": 0} for c in FILE_TO_COLLECTION.values()}

    with zipfile.ZipFile(zip_path) as zf:
        for entry in zf.namelist():
            if not entry.endswith(".json"):
                continue

            # Determine target collection from file name prefix
            name_lower = entry.lower()
            collection = next((c for prefix, c in FILE_TO_COLLECTION.items() if prefix in name_lower), None)
            if not collection:
                print(f"  Skipping unrecognised file: {entry}")
                continue

            print(f"Processing {entry} → {collection} ...")
            with zf.open(entry) as f:
                records = json.load(f)

            col = clean_db[collection]
            _ensure_indexes(col)

            for record in records:
                doc = _to_doc(record, collection)
                if doc is None:
                    counters[collection]["skipped"] += 1
                    continue
                col.update_one(
                    {"source_id": doc["source_id"]},
                    {"$set": doc},
                    upsert=True,
                )
                counters[collection]["upserted"] += 1

    print("\n── Results ──────────────────────────────────────────")
    for collection, stats in counters.items():
        print(f"  {collection:15s}  upserted={stats['upserted']}  skipped(out of range/no coords)={stats['skipped']}")


if __name__ == "__main__":
    default_zip = Path(__file__).resolve().parents[1] / "restaurant_hospital_pharmacy_data.zip"
    zip_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_zip
    if not zip_path.exists():
        print(f"Zip not found: {zip_path}")
        sys.exit(1)
    run(zip_path)
