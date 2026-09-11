"""
Load bus route data into clean.bus_routes.

Reads somnath_bus_routes.jsonl, strips the heavy per-trip detail (trips array),
retains route-level summary fields, and upserts into MongoDB.

Run with:
    ../backend/venv/bin/python etl/load_bus_routes.py
    ../backend/venv/bin/python etl/load_bus_routes.py --drop
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import get_clean_db  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "train_flight_bus_route_data"
BUS_FILE = DATA_DIR / "somnath_bus_routes.jsonl"


def _flatten(route: dict) -> dict:
    return {
        "route_id": route["route_id"],
        "origin": route["origin"],
        "destination": route["destination"],
        "distance_km": route.get("distance_km"),
        "typical_duration_min": route.get("typical_duration_min"),
        "operator_count": route.get("operator_count"),
        "trip_count": route.get("trip_count"),
        "first_departure": route.get("first_departure"),
        "last_departure": route.get("last_departure"),
        "fare_range_inr": route.get("fare_range_inr"),
        "boarding_point_names": [
            p["name"] for p in route.get("boarding_points", []) if p.get("name")
        ],
        "dropping_point_names": [
            p["name"] for p in route.get("dropping_points", []) if p.get("name")
        ],
    }


def run(drop: bool = False) -> None:
    db = get_clean_db()
    col = db["bus_routes"]

    if drop:
        col.drop()
        print("dropped bus_routes collection")

    routes = []
    with open(BUS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                routes.append(_flatten(json.loads(line)))

    print(f"loaded {len(routes)} bus routes")

    upserted = updated = 0
    for doc in routes:
        result = col.update_one(
            {"route_id": doc["route_id"]},
            {"$set": doc},
            upsert=True,
        )
        if result.upserted_id:
            upserted += 1
        elif result.modified_count:
            updated += 1

    col.create_index("route_id", unique=True)
    col.create_index("origin.name")
    col.create_index("destination.name")

    print(f"done — inserted {upserted}, updated {updated}")
    print(f"total in collection: {col.count_documents({})}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", action="store_true", help="drop collection before loading")
    args = parser.parse_args()
    run(drop=args.drop)
