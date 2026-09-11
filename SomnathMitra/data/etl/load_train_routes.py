"""
Load train timetable data into clean.train_routes.

Reads veraval_arrivals.json and veraval_departures.json, deduplicates by
train_number (halting trains appear in both files), strips the heavy
stop-by-stop route array, and upserts into MongoDB.

Run with:
    ../backend/venv/bin/python etl/load_train_routes.py
    ../backend/venv/bin/python etl/load_train_routes.py --drop
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import get_clean_db  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "train_flight_bus_route_data"
ARRIVALS_FILE = DATA_DIR / "veraval_arrivals.json"
DEPARTURES_FILE = DATA_DIR / "veraval_departures.json"

KEEP_FIELDS = {
    "train_number", "return_train_number", "train_name", "train_type",
    "train_category", "service_type",
    "source_station", "source_station_code",
    "destination_station", "destination_station_code",
    "running_days", "running_days_text", "frequency_per_week",
    "serves_veraval", "serves_somnath",
    "terminates_at_veraval", "originates_at_veraval",
    "terminates_at_somnath", "originates_at_somnath",
    "stops_at_somnath",
    "veraval", "somnath",
}

# Map full station names to their common city name for easier searching
STATION_TO_CITY: dict[str, str] = {
    "Bandra Terminus": "Mumbai",
    "Mumbai Bandra Terminus": "Mumbai",
    "Bhavnagar Terminus": "Bhavnagar",
    "Gandhinagar Capital": "Gandhinagar",
    "Haridwar Junction": "Haridwar",
    "Indore Junction": "Indore",
    "Jabalpur Junction": "Jabalpur",
    "Junagadh Junction": "Junagadh",
    "Prayagraj Junction": "Prayagraj",
    "Pune Junction": "Pune",
    "Rajkot Junction": "Rajkot",
    "Sabarmati Junction": "Ahmedabad",
    "Thiruvananthapuram Central": "Thiruvananthapuram",
    "Veraval Junction": "Veraval",
    # raw uppercase forms found in route arrays
    "Delhi": "Delhi",
    "New Delhi": "Delhi",
    "Hazrat Nizamuddin": "Delhi",
}

# Abbreviations used in raw route station names
_ABBREV = {" Jn": " Junction", " Jct": " Junction", " Rd": " Road"}


def _city(station: str) -> str:
    """Normalise a station name (title-cased) to a city name."""
    for abbr, full in _ABBREV.items():
        station = station.replace(abbr, full)
    if station in STATION_TO_CITY:
        return STATION_TO_CITY[station]
    for suffix in (" Junction", " Terminus", " Central", " Capital", " Road"):
        if station.endswith(suffix):
            return station[: -len(suffix)]
    return station


def _extract_stops(train: dict) -> tuple[list[str], list[dict]]:
    """
    Extract intermediate stops from the full route array.
    Returns (stop_cities, compact_stops).
    stop_cities  — deduplicated list of city names for fast DB queries.
    compact_stops — [{city, station_name, station_code, arrival, departure, day}]
                    excluding the origin and terminus stops.
    """
    src_code = train.get("source_station_code", "")
    dst_code = train.get("destination_station_code", "")
    route = train.get("route", [])

    city_set: set[str] = set()
    compact: list[dict] = []

    for stop in route:
        code = stop.get("station_code", "")
        if code in (src_code, dst_code):
            continue
        raw_name = stop.get("station_name", "").title()
        city = _city(raw_name)
        city_set.add(city)
        compact.append({
            "city": city,
            "station_name": raw_name,
            "station_code": code,
            "arrival": stop.get("arrival"),
            "departure": stop.get("departure"),
            "day": stop.get("day"),
        })

    return sorted(city_set), compact


def _flatten(train: dict) -> dict:
    doc = {k: v for k, v in train.items() if k in KEEP_FIELDS}
    doc["source_city"] = _city(train["source_station"])
    doc["destination_city"] = _city(train["destination_station"])
    stop_cities, compact_stops = _extract_stops(train)
    doc["stop_cities"] = stop_cities
    doc["intermediate_stops"] = compact_stops
    return doc


def run(drop: bool = False) -> None:
    db = get_clean_db()
    col = db["train_routes"]

    if drop:
        col.drop()
        print("dropped train_routes collection")

    trains: dict[str, dict] = {}
    for path in (ARRIVALS_FILE, DEPARTURES_FILE):
        with open(path) as f:
            data = json.load(f)
        for t in data["trains"]:
            trains[t["train_number"]] = _flatten(t)

    print(f"loaded {len(trains)} unique trains")

    upserted = updated = 0
    for doc in trains.values():
        result = col.update_one(
            {"train_number": doc["train_number"]},
            {"$set": doc},
            upsert=True,
        )
        if result.upserted_id:
            upserted += 1
        elif result.modified_count:
            updated += 1

    col.create_index("train_number", unique=True)
    col.create_index("source_station")
    col.create_index("destination_station")
    col.create_index("source_city")
    col.create_index("destination_city")
    col.create_index("stop_cities")
    col.create_index("serves_somnath")
    col.create_index("running_days.monday")
    col.create_index("running_days.tuesday")
    col.create_index("running_days.wednesday")
    col.create_index("running_days.thursday")
    col.create_index("running_days.friday")
    col.create_index("running_days.saturday")
    col.create_index("running_days.sunday")

    print(f"done — inserted {upserted}, updated {updated}")
    print(f"total in collection: {col.count_documents({})}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", action="store_true", help="drop collection before loading")
    args = parser.parse_args()
    run(drop=args.drop)
