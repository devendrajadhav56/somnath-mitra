"""
Build flight_access_hubs collection from the weekly timetable in the flights JSON.

Each hub = one origin airport city (e.g. Mumbai/BOM) with a `flights` array
of arrivals at the four airports near Somnath (HSR, DIU, IXK, PBD).

Only inbound timetable entries (direction == "arrival") are used.
Flights with the same flight_number + destination airport are merged across
day-groups so each service appears once with its combined days_of_week.

Run with:
    ../backend/venv/bin/python etl/load_flight_access_hubs.py
    ../backend/venv/bin/python etl/load_flight_access_hubs.py --drop
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db import get_clean_db  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "train_flight_bus_route_data"
FLIGHTS_FILE = DATA_DIR / "flights_20260909_163102.json"

# Origin airports we build hubs for. IXK is excluded — it's a near-Somnath
# destination airport, not a meaningful pilgrim origin.
ORIGIN_AIRPORTS: dict[str, dict] = {
    "AMD": {
        "city": "Ahmedabad",
        "airport_name": "Sardar Vallabhbhai Patel International Airport",
        "lat": 23.0734, "lon": 72.6347,
    },
    "BLR": {
        "city": "Bengaluru",
        "airport_name": "Kempegowda International Airport",
        "lat": 13.1986, "lon": 77.7066,
    },
    "BOM": {
        "city": "Mumbai",
        "airport_name": "Chhatrapati Shivaji Maharaj International Airport",
        "lat": 19.0897, "lon": 72.8679,
    },
    "DEL": {
        "city": "Delhi",
        "airport_name": "Indira Gandhi International Airport",
        "lat": 28.5562, "lon": 77.1000,
    },
    "GOX": {
        "city": "Goa",
        "airport_name": "Goa Manohar International Airport",
        "lat": 15.3826, "lon": 73.8314,
    },
    "HYD": {
        "city": "Hyderabad",
        "airport_name": "Rajiv Gandhi International Airport",
        "lat": 17.2403, "lon": 78.4294,
    },
    "NMI": {
        "city": "Navi Mumbai",
        "airport_name": "Navi Mumbai International Airport",
        "lat": 18.9824, "lon": 73.1178,
    },
    "PNQ": {
        "city": "Pune",
        "airport_name": "Pune International Airport",
        "lat": 18.5822, "lon": 73.9197,
    },
    "STV": {
        "city": "Surat",
        "airport_name": "Surat International Airport",
        "lat": 21.1141, "lon": 72.7418,
    },
}

DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _parse_onward(note: str) -> tuple[int | None, int | None]:
    """Parse '~55 km / 1.5 h by road' or '~190 km from Somnath, ~4 h by road'."""
    km_match = re.search(r"~?(\d+)\s*km", note)
    h_match = re.search(r"~?(\d+(?:\.\d+)?)\s*h", note)
    km = int(km_match.group(1)) if km_match else None
    hours = float(h_match.group(1)) if h_match else None
    minutes = int(hours * 60) if hours is not None else None
    return km, minutes


def _hub_id(iata: str) -> str:
    return iata.lower()


def run(drop: bool = False) -> None:
    with open(FLIGHTS_FILE) as f:
        data = json.load(f)

    # Build destination airport metadata (onward distance/time from the airports key)
    dest_airports: dict[str, dict] = {}
    for iata, info in data["airports"].items():
        km, minutes = _parse_onward(info.get("note", ""))
        dest_airports[iata] = {
            "iata":       iata,
            "name":       info["name"],
            "city":       info["city"],
            "onward_km":  km,
            "onward_min": minutes,
        }

    # Filter to inbound timetable entries from known origin airports
    arrivals = [
        f for f in data["weekly_timetable"]
        if f["direction"] == "arrival" and f["other_endpoint"] in ORIGIN_AIRPORTS
    ]

    # Deduplicate: same flight_number + focus_airport may appear across day-groups
    # with slightly different scheduled times. Merge days, keep first arrival time seen.
    merged: dict[tuple[str, str], dict] = {}
    for f in arrivals:
        key = (f["flight_number"], f["focus_airport"])
        if key not in merged:
            merged[key] = {
                "flight_number":        f["flight_number"],
                "airline":              f["airline_name"],
                "airline_iata":         f["airline_iata"],
                "origin_iata":          f["other_endpoint"],
                "destination_iata":     f["focus_airport"],
                "arrival_time":         f["scheduled_local_time"],
                "days_of_week":         set(f["days_of_week"]),
                "via_stops":            f.get("stops", []),
            }
        else:
            merged[key]["days_of_week"].update(f["days_of_week"])

    # Sort days and compute frequency
    flight_list: list[dict] = []
    for entry in merged.values():
        days = sorted(entry["days_of_week"], key=lambda d: DAY_ORDER.index(d))
        dest = dest_airports.get(entry["destination_iata"], {})
        flight_list.append({
            "flight_number":    entry["flight_number"],
            "airline":          entry["airline"],
            "airline_iata":     entry["airline_iata"],
            "origin_iata":      entry["origin_iata"],
            "destination_iata": entry["destination_iata"],
            "destination_name": dest.get("name", entry["destination_iata"]),
            "destination_city": dest.get("city", entry["destination_iata"]),
            "arrival_time":     entry["arrival_time"],
            "days_of_week":     days,
            "frequency_per_week": len(days),
            "onward_km":        dest.get("onward_km"),
            "onward_min":       dest.get("onward_min"),
            "via_stops":        entry["via_stops"],
        })

    # Group flights by origin airport → build hub docs
    from collections import defaultdict
    by_origin: dict[str, list[dict]] = defaultdict(list)
    for f in flight_list:
        by_origin[f["origin_iata"]].append(f)

    db = get_clean_db()
    col = db["flight_access_hubs"]

    if drop:
        col.drop()
        print("dropped flight_access_hubs")

    upserted = updated = 0
    for iata, flights in by_origin.items():
        origin = ORIGIN_AIRPORTS[iata]
        lat, lon = origin["lat"], origin["lon"]
        doc = {
            "hub_id":       _hub_id(iata),
            "airport_iata": iata,
            "airport_name": origin["airport_name"],
            "city":         origin["city"],
            "lat":          lat,
            "lon":          lon,
            "location":     {"type": "Point", "coordinates": [lon, lat]},
            "flights":      sorted(flights, key=lambda f: (-f["frequency_per_week"], f["arrival_time"])),
        }
        result = col.update_one({"hub_id": doc["hub_id"]}, {"$set": doc}, upsert=True)
        if result.upserted_id:
            upserted += 1
        elif result.modified_count:
            updated += 1
        print(f"  {iata} ({origin['city']}): {len(flights)} flight(s)")

    col.create_index([("location", "2dsphere")])
    col.create_index("hub_id", unique=True)
    col.create_index("airport_iata")

    print(f"\nDone — inserted {upserted}, updated {updated}")
    print(f"Total hubs: {col.count_documents({})}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", action="store_true")
    args = parser.parse_args()
    run(drop=args.drop)
