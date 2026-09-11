"""
Idempotent sync: raw.pois -> clean.pois

- normalizes blank strings to null
- dedups by name + proximity (<=50m), keeping the most complete doc
- drops/flags docs outside the Somnath region bounding box
Safe to re-run any time raw.pois changes; upserts on source_id.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import get_clean_db, get_raw_db  # noqa: E402
from schemas.poi import Poi  # noqa: E402

# Loose bounding box around the Somnath / Veraval / Junagadh region.
BBOX = {"min_lat": 20.6, "max_lat": 21.4, "min_lon": 69.9, "max_lon": 70.9}
DEDUP_RADIUS_M = 50


def haversine_m(lon1, lat1, lon2, lat2):
    r = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def in_bbox(lon, lat):
    return BBOX["min_lat"] <= lat <= BBOX["max_lat"] and BBOX["min_lon"] <= lon <= BBOX["max_lon"]


def completeness(doc):
    return sum(1 for f in ("address", "phone", "website", "opening_hours") if doc.get(f))


def blank_to_none(v):
    return v if v not in ("", None) else None


def dedup_groups(docs):
    """Group docs sharing a name, then cluster by proximity within each group."""
    by_name = {}
    for doc in docs:
        by_name.setdefault(doc.get("name"), []).append(doc)

    clusters = []
    for name, group in by_name.items():
        remaining = list(group)
        while remaining:
            seed = remaining.pop(0)
            seed_coords = seed["location"]["coordinates"]
            cluster = [seed]
            still_remaining = []
            for other in remaining:
                other_coords = other["location"]["coordinates"]
                if haversine_m(*seed_coords, *other_coords) <= DEDUP_RADIUS_M:
                    cluster.append(other)
                else:
                    still_remaining.append(other)
            remaining = still_remaining
            clusters.append(cluster)
    return clusters


def run():
    raw_db = get_raw_db()
    clean_db = get_clean_db()

    raw_docs = list(raw_db["pois"].find({}))
    excluded = [d for d in raw_docs if not in_bbox(*d["location"]["coordinates"])]
    in_region = [d for d in raw_docs if in_bbox(*d["location"]["coordinates"])]

    if excluded:
        print(f"excluding {len(excluded)} pois outside the Somnath region bbox:")
        for d in excluded:
            print(f"  {d.get('name')} ({d['_id']}) at {d['location']['coordinates']}")

    clusters = dedup_groups(in_region)
    upserted = 0
    for cluster in clusters:
        cluster.sort(key=lambda d: (-completeness(d), str(d["_id"])))
        primary, *dupes = cluster

        clean_doc = {
            "source_id": str(primary["_id"]),
            "name": primary.get("name"),
            "category": blank_to_none(primary.get("category")),
            "subtype": blank_to_none(primary.get("subtype")),
            "address": blank_to_none(primary.get("address")),
            "phone": blank_to_none(primary.get("phone")),
            "website": blank_to_none(primary.get("website")),
            "opening_hours": blank_to_none(primary.get("opening_hours")),
            "location": primary["location"],
            "merged_from": [str(d["_id"]) for d in dupes],
            "schema_version": 1,
        }
        validated = Poi.model_validate(clean_doc).model_dump()
        clean_db["pois"].replace_one(
            {"source_id": validated["source_id"]}, validated, upsert=True
        )
        upserted += 1

    print(f"synced {upserted} clean pois ({len(in_region)} raw docs, "
          f"{len(in_region) - upserted} merged as duplicates, {len(excluded)} excluded)")


if __name__ == "__main__":
    run()
