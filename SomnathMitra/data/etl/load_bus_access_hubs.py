"""
Build bus_access_hubs collection from two sources:
  - somnath_bus_routes.jsonl          (Gujarat, aggregated route-level data)
  - direct_private_buses_to_somnath_india.json  (outside Gujarat, per-operator trips)

Each hub doc = one origin city with a `buses` array of services.
Gujarat hubs carry aggregated stats; outside-Gujarat hubs carry per-operator trips.

Run with:
    ../backend/venv/bin/python etl/load_bus_access_hubs.py
    ../backend/venv/bin/python etl/load_bus_access_hubs.py --drop
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db import get_clean_db  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "train_flight_bus_route_data"
GUJARAT_FILE = DATA_DIR / "somnath_bus_routes.jsonl"
OUTSIDE_FILE = DATA_DIR / "direct_private_buses_to_somnath_india.json"

# Coordinates for outside-Gujarat origin cities (no lat/lon in that dataset)
CITY_COORDS: dict[str, tuple[float, float]] = {
    "Mumbai":      (19.0760,  72.8777),
    "Thane":       (19.2183,  72.9781),
    "Navi Mumbai": (19.0368,  73.0158),
    "Pune":        (18.5204,  73.8567),
    "Kalyan":      (19.2437,  73.1355),
    "Panvel":      (18.9894,  73.1175),
    "Vasai":       (19.3919,  72.8397),
    "Udaipur":     (24.5854,  73.7125),
    "Nathdwara":   (24.9333,  73.8167),
    "Bhilwara":    (25.3467,  74.6350),
    "Sirohi":      (24.8878,  72.8635),
    "Abu Road":    (24.4800,  72.7833),
    "Mount Abu":   (24.5926,  72.7156),
    "Sumerpur":    (25.1572,  73.0841),
    "Indore":      (22.7196,  75.8577),
    "Ujjain":      (23.1793,  75.7849),
    "Dhar":        (22.5989,  75.3007),
    "Ratlam":      (23.3315,  74.9455),
    "Diu":         (20.7144,  70.9874),
    "Silvassa":    (20.2734,  73.0169),
    "Ghoghla":     (20.7014,  70.9640),
    "Daman":       (20.3974,  72.8328),
}


def _hub_id(city: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", city.lower()).strip("-")


def _parse_duration_min(duration_str: str) -> int | None:
    """Parse '17h 00m' or '16h 15m' → minutes."""
    m = re.match(r"(\d+)h\s*(\d+)m", duration_str or "")
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    m = re.match(r"(\d+)h", duration_str or "")
    if m:
        return int(m.group(1)) * 60
    return None


def _load_gujarat() -> dict[str, dict]:
    """One hub per Gujarat origin city. Bus entry is aggregated route stats."""
    hubs: dict[str, dict] = {}
    with open(GUJARAT_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            origin = r["origin"]
            city = origin["name"]
            lat = origin.get("lat")
            lng = origin.get("lng")
            if lat is None or lng is None:
                continue

            hid = _hub_id(city)
            fare = r.get("fare_range_inr") or []
            bus_entry = {
                "kind":           "aggregated",
                "route_id":       r["route_id"],
                "operator_count": r.get("operator_count"),
                "trip_count":     r.get("trip_count"),
                "first_departure": r.get("first_departure"),
                "last_departure":  r.get("last_departure"),
                "duration_min":   r.get("typical_duration_min"),
                "fare_min":       fare[0] if len(fare) > 0 else None,
                "fare_max":       fare[1] if len(fare) > 1 else None,
            }

            hubs[hid] = {
                "hub_id":  hid,
                "city":    city,
                "state":   origin.get("state", "Gujarat"),
                "lat":     lat,
                "lon":     lng,
                "location": {"type": "Point", "coordinates": [lng, lat]},
                "buses":   [bus_entry],
            }
    return hubs


def _load_outside_gujarat() -> dict[str, dict]:
    """Group per-operator routes by origin city. Each bus entry is one operator/trip."""
    with open(OUTSIDE_FILE) as f:
        data = json.load(f)

    city_routes: dict[str, list[dict]] = defaultdict(list)
    for r in data["routes"]:
        city_routes[r["origin"]["city"]].append(r)

    hubs: dict[str, dict] = {}
    missing_coords: list[str] = []

    for city, routes in city_routes.items():
        coords = CITY_COORDS.get(city)
        if coords is None:
            missing_coords.append(city)
            continue

        lat, lon = coords
        hid = _hub_id(city)
        state = routes[0]["origin"]["state"]

        buses = []
        for r in routes:
            b = r["bus"]
            buses.append({
                "kind":           "operator",
                "route_id":       r["route_id"],
                "operator":       r.get("operator"),
                "bus_type":       b.get("bus_type"),
                "departure_time": b.get("departure_time"),
                "arrival_time":   b.get("arrival_time"),
                "duration_min":   _parse_duration_min(b.get("duration", "")),
                "fare_min":       b["fare_inr"].get("minimum") if b.get("fare_inr") else None,
                "fare_max":       b["fare_inr"].get("maximum") if b.get("fare_inr") else None,
            })

        hubs[hid] = {
            "hub_id":  hid,
            "city":    city,
            "state":   state,
            "lat":     lat,
            "lon":     lon,
            "location": {"type": "Point", "coordinates": [lon, lat]},
            "buses":   buses,
        }

    if missing_coords:
        print(f"WARNING: no coords for cities: {missing_coords}")

    return hubs


def run(drop: bool = False) -> None:
    db = get_clean_db()
    col = db["bus_access_hubs"]

    if drop:
        col.drop()
        print("dropped bus_access_hubs")

    gujarat_hubs = _load_gujarat()
    outside_hubs = _load_outside_gujarat()

    print(f"Gujarat hubs:        {len(gujarat_hubs)}")
    print(f"Outside-Gujarat hubs: {len(outside_hubs)}")

    # Gujarat first, outside-Gujarat wins on conflict (richer per-operator data)
    all_hubs = {**gujarat_hubs, **outside_hubs}
    print(f"Total unique hubs:   {len(all_hubs)}")

    upserted = updated = 0
    for doc in all_hubs.values():
        result = col.update_one(
            {"hub_id": doc["hub_id"]},
            {"$set": doc},
            upsert=True,
        )
        if result.upserted_id:
            upserted += 1
        elif result.modified_count:
            updated += 1

    col.create_index([("location", "2dsphere")])
    col.create_index("hub_id", unique=True)
    col.create_index("city")
    col.create_index("state")

    print(f"Done — inserted {upserted}, updated {updated}")
    print(f"Total in collection: {col.count_documents({})}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", action="store_true", help="drop collection before loading")
    args = parser.parse_args()
    run(drop=args.drop)
