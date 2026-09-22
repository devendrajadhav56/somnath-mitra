"""
Load Veraval-bound train network from the indianrailways-gtfs GTFS snapshot.

Produces two MongoDB collections in clean DB:
  veraval_trains  — one doc per inbound trip (→ Veraval), full stop list
  access_hubs     — one doc per unique station, pre-joined with its Veraval trains

Run with:
    ../backend/venv/bin/python etl/load_veraval_network.py
    ../backend/venv/bin/python etl/load_veraval_network.py --drop
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db import get_clean_db  # noqa: E402

GTFS_ZIP = (
    Path(__file__).resolve().parent.parent
    / "train_flight_bus_route_data"
    / "indianrailways_gtfs.zip"
)

VERAVAL_STOP_ID = "VRL"
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
DAY_ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_csv(zf: zipfile.ZipFile, name: str) -> list[dict]:
    with zf.open(name) as f:
        return list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))


def _parse_time(t: str) -> tuple[int, str | None]:
    """GTFS times can exceed 24:00 for overnight trips. Returns (day_offset, 'HH:MM')."""
    if not t or t.strip() == "":
        return 0, None
    parts = t.strip().split(":")
    h = int(parts[0])
    m = parts[1] if len(parts) > 1 else "00"
    return h // 24, f"{h % 24:02d}:{m}"


def _running_days_text(days: dict) -> str:
    active = [DAY_ABBR[i] for i, name in enumerate(DAYS) if days.get(name)]
    if len(active) == 7:
        return "Daily"
    if not active:
        return "No service"
    return ", ".join(active)


# ── Main ETL ──────────────────────────────────────────────────────────────────

def run(drop: bool = False) -> None:
    db = get_clean_db()
    col_trains = db["veraval_trains"]
    col_hubs = db["access_hubs"]

    if drop:
        col_trains.drop()
        col_hubs.drop()
        print("dropped veraval_trains and access_hubs")

    with zipfile.ZipFile(GTFS_ZIP) as zf:
        stops_rows = _read_csv(zf, "stops.txt")
        trips_rows = _read_csv(zf, "trips.txt")
        calendar_rows = _read_csv(zf, "calendar.txt")
        stop_times_rows = _read_csv(zf, "stop_times.txt")
        routes_rows = _read_csv(zf, "routes.txt")

    # Index: stop_id → stop metadata
    stops: dict[str, dict] = {s["stop_id"]: s for s in stops_rows}

    # Index: route_id → route metadata
    routes: dict[str, dict] = {r["route_id"]: r for r in routes_rows}

    # Index: service_id → running_days dict
    calendar: dict[str, dict] = {}
    for row in calendar_rows:
        sid = row["service_id"]
        # Multiple entries can share the same service_id pattern (different date ranges).
        # We use the one with the latest end_date (most current).
        if sid not in calendar or row["end_date"] > calendar[sid]["end_date"]:
            calendar[sid] = row

    # Index: trip_id → trip metadata
    trips: dict[str, dict] = {t["trip_id"]: t for t in trips_rows}

    # Step 1: find all trip_ids that stop at Veraval
    veraval_trip_ids: set[str] = set()
    for row in stop_times_rows:
        if row["stop_id"] == VERAVAL_STOP_ID:
            veraval_trip_ids.add(row["trip_id"])

    # Step 2: keep only inbound trips (headsign is VERAVAL)
    inbound_trip_ids = {
        t for t in veraval_trip_ids
        if "VERAVAL" in trips.get(t, {}).get("trip_headsign", "").upper()
    }
    print(f"Total trips at Veraval: {len(veraval_trip_ids)}  |  Inbound: {len(inbound_trip_ids)}")

    # Step 3: collect all stop_times for inbound trips
    trip_stop_times: dict[str, list[dict]] = defaultdict(list)
    for row in stop_times_rows:
        if row["trip_id"] in inbound_trip_ids:
            trip_stop_times[row["trip_id"]].append(row)

    # Step 4: build train documents
    train_docs: list[dict] = []
    hub_data: dict[str, dict] = {}  # station_code → hub doc being built

    for trip_id in sorted(inbound_trip_ids):
        trip = trips.get(trip_id, {})
        route = routes.get(trip.get("route_id", ""), {})
        svc_id = trip.get("service_id", "")
        # Exact lookup first; fall back to matching on day-pattern prefix in case
        # the date-range suffix in trips.txt differs by a day from calendar.txt.
        svc = calendar.get(svc_id) or next(
            (v for k, v in calendar.items() if k.split("_")[0] == svc_id.split("_")[0]),
            {},
        )

        running_days = {d: svc.get(d, "0") == "1" for d in DAYS}

        # Sort stops by sequence
        raw_stops = sorted(trip_stop_times[trip_id], key=lambda x: int(x["stop_sequence"]))

        # Build clean stop list
        clean_stops = []
        veraval_arrival = None
        veraval_day = 0

        for st in raw_stops:
            sid = st["stop_id"]
            stop = stops.get(sid, {})
            lat_s = stop.get("stop_lat", "")
            lon_s = stop.get("stop_lon", "")
            lat = float(lat_s) if lat_s else None
            lon = float(lon_s) if lon_s else None

            arr_day, arr_time = _parse_time(st.get("arrival_time", ""))
            dep_day, dep_time = _parse_time(st.get("departure_time", ""))
            day_offset = arr_day or dep_day

            if sid == VERAVAL_STOP_ID:
                veraval_arrival = arr_time
                veraval_day = day_offset

            clean_stops.append({
                "sequence": int(st["stop_sequence"]),
                "code": sid,
                "name": stop.get("stop_name", sid),
                "lat": lat,
                "lon": lon,
                "arrival": arr_time,
                "departure": dep_time,
                "day": day_offset,
            })

        train_name = route.get("route_long_name") or trip.get("trip_headsign", trip_id)
        train_doc = {
            "trip_id": trip_id,
            "train_name": train_name,
            "running_days": running_days,
            "running_days_text": _running_days_text(running_days),
            "service_id": svc_id,
            "start_date": svc.get("start_date"),
            "end_date": svc.get("end_date"),
            "veraval_arrival": veraval_arrival,
            "veraval_day": veraval_day,
            "stops": clean_stops,
        }
        train_docs.append(train_doc)

        # Accumulate hub data for every stop in this train (except Veraval itself)
        for st in clean_stops:
            code = st["code"]
            if code == VERAVAL_STOP_ID or st["lat"] is None:
                continue

            if code not in hub_data:
                hub_data[code] = {
                    "station_code": code,
                    "station_name": st["name"],
                    "lat": st["lat"],
                    "lon": st["lon"],
                    "location": {
                        "type": "Point",
                        "coordinates": [st["lon"], st["lat"]],
                    },
                    "trains": [],
                }

            hub_data[code]["trains"].append({
                "trip_id": trip_id,
                "train_name": train_name,
                "departure": st["departure"],
                "departure_day": st["day"],
                "veraval_arrival": veraval_arrival,
                "veraval_day": veraval_day,
                "running_days": running_days,
                "running_days_text": _running_days_text(running_days),
            })

    # Step 5: upsert trains
    upserted = updated = 0
    for doc in train_docs:
        res = col_trains.update_one(
            {"trip_id": doc["trip_id"]},
            {"$set": doc},
            upsert=True,
        )
        if res.upserted_id:
            upserted += 1
        elif res.modified_count:
            updated += 1

    col_trains.create_index("trip_id", unique=True)
    print(f"veraval_trains: inserted {upserted}, updated {updated}  (total: {col_trains.count_documents({})})")

    # Step 6: upsert hubs — add train_count
    upserted = updated = 0
    for code, doc in hub_data.items():
        doc["train_count"] = len(doc["trains"])
        res = col_hubs.update_one(
            {"station_code": code},
            {"$set": doc},
            upsert=True,
        )
        if res.upserted_id:
            upserted += 1
        elif res.modified_count:
            updated += 1

    col_hubs.create_index([("location", "2dsphere")])
    col_hubs.create_index("station_code", unique=True)
    col_hubs.create_index("train_count")
    print(f"access_hubs:    inserted {upserted}, updated {updated}  (total: {col_hubs.count_documents({})})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", action="store_true")
    args = parser.parse_args()
    run(drop=args.drop)
