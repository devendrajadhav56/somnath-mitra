"""
Derive a static airport/flight summary from the raw FlightRadar24 snapshot
and seed it into clean.temple_info as key = "how_to_reach_by_air".

The raw data is a dated snapshot (not a live schedule), so we only extract
stable facts: which airports are near Somnath, which airlines operate routes
to those airports, and from which cities.

Run with:
    ../backend/venv/bin/python etl/seed_flight_info.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import get_clean_db  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "train_flight_bus_route_data"
FLIGHTS_FILE = next(DATA_DIR.glob("flights_*.json"))

# Distance order for display
AIRPORT_ORDER = ["IXK", "DIU", "PBD", "HSR"]


def run() -> None:
    with open(FLIGHTS_FILE) as f:
        data = json.load(f)

    airports = data["airports"]
    flights = data["flights"]

    # For each airport: which airlines fly IN from which cities?
    # (arrivals = someone coming TO the area to visit Somnath)
    airport_services: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for fl in flights:
        if fl["direction"] == "arrival" and fl.get("airline_name") and fl.get("origin_city"):
            airport_services[fl["focus_airport"]][fl["airline_name"]].add(fl["origin_city"])

    items = []
    for iata in AIRPORT_ORDER:
        if iata not in airports:
            continue
        info = airports[iata]
        services = airport_services.get(iata, {})
        if services:
            routes_str = "; ".join(
                f"{airline}: {', '.join(sorted(cities))}"
                for airline, cities in sorted(services.items())
            )
            note = f"{info['note']} | Airlines (sample routes): {routes_str}"
        else:
            note = f"{info['note']} | Check airline websites for current schedules"
        items.append({
            "name": f"{info['name']} ({iata}) — {info['city']}",
            "note": note,
        })

    doc = {
        "key": "how_to_reach_by_air",
        "title": "Reaching Somnath by Air",
        "content": (
            "There are no direct flights to Somnath. The nearest airports are listed below. "
            "From any of them, hire a taxi or take a bus/shared cab to reach Somnath temple. "
            "Keshod (IXK) is the closest but has limited routes; Rajkot (HSR) has the widest "
            "choice of flights from major Indian cities."
        ),
        "items": items,
        "caveats": [
            "Flight schedules change seasonally — verify on airline websites before booking.",
            "Road transfers: Keshod ~1.5h, Diu ~2h, Porbandar ~3h, Rajkot ~4h.",
        ],
    }

    col = get_clean_db()["temple_info"]
    result = col.update_one({"key": "how_to_reach_by_air"}, {"$set": doc}, upsert=True)
    if result.upserted_id:
        print("inserted how_to_reach_by_air into temple_info")
    else:
        print("updated how_to_reach_by_air in temple_info")


if __name__ == "__main__":
    run()
