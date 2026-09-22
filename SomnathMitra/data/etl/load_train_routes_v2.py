"""
Load train timetable data into clean.train_routes from Kunal's scraped dataset.

Source: Train_data_latest_kunal_scraped.zip
  schedules.jsonl — one JSON object per line, each with full stop list

Filters to trains that serve Veraval Junction (VRL) station only.
Replaces the old veraval_arrivals/veraval_departures JSON pipeline.

Run with:
    ../backend/venv/bin/python etl/load_train_routes_v2.py
    ../backend/venv/bin/python etl/load_train_routes_v2.py --drop
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db import get_clean_db  # noqa: E402

VERAVAL_CODE = "VRL"
DATA_ZIP = Path(__file__).resolve().parent.parent / "Train_data_latest_kunal_scraped.zip"

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_DAY_ABBR_TO_FULL = {
    "Mon": "monday", "Tue": "tuesday", "Wed": "wednesday", "Thu": "thursday",
    "Fri": "friday", "Sat": "saturday", "Sun": "sunday",
}
_DAY_DISPLAY = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

_STATION_CITY: dict[str, str] = {
    "SABARMATI BG": "Ahmedabad",
    "GANDHINAGAR CAPITAL": "Gandhinagar",
    "BANDRA TERMINUS": "Mumbai",
    "MUMBAI CENTRAL": "Mumbai",
    "CSMT": "Mumbai",
    "LOKMANYATILAK T": "Mumbai",
    "THIRUVANANTHAPURAM CENTRAL": "Thiruvananthapuram",
    "KOLKATTA TERMINAL": "Kolkata",
    "HOWRAH JN": "Kolkata",
    "DELHI SAFDARJUNG": "Delhi",
    "HAZRAT NIZAMUDDIN": "Delhi",
    "NEW DELHI": "Delhi",
    "DELHI": "Delhi",
    "BHAVNAGAR TERMINUS": "Bhavnagar",
    "PRAYAGRAJ JN": "Prayagraj",
    "HARIDWAR JN": "Haridwar",
    "INDORE JN": "Indore",
    "JUNAGADH JN": "Junagadh",
    "RAJKOT": "Rajkot",
    "VERAVAL": "Veraval",
    "BANARAS": "Varanasi",
    "OKHA": "Okha",
    "DWARKA": "Dwarka",
    "DELVADA": "Delvada",
    "PUNE JN": "Pune",
    "DIVA": "Diva",
    "DWARKA JN": "Dwarka",
}

_STRIP_SUFFIXES = (
    " JUNCTION", " TERMINUS", " CENTRAL", " CAPITAL", " ROAD", " BG",
    " JN", " JCT", " RD",
)


def _city(raw: str) -> str:
    upper = raw.strip().upper()
    if upper in _STATION_CITY:
        return _STATION_CITY[upper]
    title = raw.strip().title()
    upper_title = title.upper()
    for sfx in _STRIP_SUFFIXES:
        if upper_title.endswith(sfx):
            return title[: -len(sfx)].strip()
    return title.strip()


def _parse_running_days(runs_days: str) -> dict:
    s = runs_days.strip()
    if s.lower() == "daily":
        return {d: True for d in DAYS}
    active = {
        _DAY_ABBR_TO_FULL[p.strip()]
        for p in s.split(",")
        if p.strip() in _DAY_ABBR_TO_FULL
    }
    return {d: (d in active) for d in DAYS}


def _running_days_text(rd: dict) -> str:
    active = [_DAY_DISPLAY[i] for i, d in enumerate(DAYS) if rd.get(d)]
    if len(active) == 7:
        return "Daily"
    if not active:
        return "No service"
    return "Runs on " + ", ".join(active)


def _or_none(val: str | None) -> str | None:
    if val is None:
        return None
    return val.strip() or None


def _build_doc(train: dict) -> dict:
    stops = train.get("stops", [])

    running_days = _parse_running_days(train.get("runs_days", ""))
    freq = sum(1 for v in running_days.values() if v)

    src_stop = stops[0] if stops else {}
    dst_stop = stops[-1] if stops else {}
    source_raw = train.get("source") or src_stop.get("name", "")
    dest_raw = train.get("destination") or dst_stop.get("name", "")
    source_code = src_stop.get("code", "")
    dest_code = dst_stop.get("code", "")

    vrl_stop = next((s for s in stops if s.get("code") == VERAVAL_CODE), None)
    terminates_at_veraval = dest_code == VERAVAL_CODE
    originates_at_veraval = source_code == VERAVAL_CODE

    veraval_info = {
        "arrival": _or_none(vrl_stop.get("arr")) if vrl_stop else None,
        "departure": _or_none(vrl_stop.get("dep")) if vrl_stop else None,
        "halt_minutes": vrl_stop.get("halt_min") if vrl_stop else None,
        "day": vrl_stop.get("day") if vrl_stop else None,
        "distance_from_origin_km": vrl_stop.get("distance_km") if vrl_stop else None,
        "terminates_here": terminates_at_veraval,
        "originates_here": originates_at_veraval,
    }

    city_set: set[str] = set()
    intermediate: list[dict] = []
    for s in stops:
        code = s.get("code", "")
        if code in (source_code, dest_code):
            continue
        name_raw = s.get("name", "")
        city = _city(name_raw)
        city_set.add(city)
        intermediate.append({
            "city": city,
            "station_name": name_raw.title(),
            "station_code": code,
            "arrival": _or_none(s.get("arr")),
            "departure": _or_none(s.get("dep")),
            "day": s.get("day"),
        })

    return {
        "train_number": train["number"],
        "train_name": train.get("name", "").title(),
        "train_type": train.get("type_label", ""),
        "train_category": train.get("type", ""),
        "service_type": None,
        "source_station": source_raw.title(),
        "source_station_code": source_code,
        "destination_station": dest_raw.title(),
        "destination_station_code": dest_code,
        "source_city": _city(source_raw),
        "destination_city": _city(dest_raw),
        "running_days": running_days,
        "running_days_text": _running_days_text(running_days),
        "frequency_per_week": freq,
        "serves_veraval": True,
        "serves_somnath": False,
        "terminates_at_veraval": terminates_at_veraval,
        "originates_at_veraval": originates_at_veraval,
        "terminates_at_somnath": False,
        "originates_at_somnath": False,
        "stops_at_somnath": False,
        "veraval": veraval_info,
        "somnath": {
            "arrival": None,
            "departure": None,
            "halt_minutes": None,
            "day": None,
            "distance_from_origin_km": None,
            "terminates_here": False,
            "originates_here": False,
        },
        "stop_cities": sorted(city_set),
        "intermediate_stops": intermediate,
        "return_train_number": None,
    }


def run(drop: bool = False) -> None:
    if not DATA_ZIP.exists():
        print(f"ERROR: zip not found at {DATA_ZIP}")
        sys.exit(1)

    db = get_clean_db()
    col = db["train_routes"]

    if drop:
        col.drop()
        print("dropped train_routes collection")

    print(f"reading {DATA_ZIP.name} ...")
    veraval_train_numbers: set[str] = set()

    with zipfile.ZipFile(DATA_ZIP) as zf:
        # Pass 1: find all train numbers that stop at VRL (exact match)
        with zf.open("stops.csv") as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            for row in reader:
                if row["station_code"] == VERAVAL_CODE:
                    veraval_train_numbers.add(row["train_number"])

        print(f"trains serving Veraval (VRL): {len(veraval_train_numbers)}")

        # Pass 2: read schedules.jsonl for those trains only
        all_trains: dict[str, dict] = {}
        with zf.open("schedules.jsonl") as f:
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                train = json.loads(line)
                if train.get("number") in veraval_train_numbers:
                    all_trains[train["number"]] = _build_doc(train)

    print(f"built {len(all_trains)} train documents")

    upserted = updated = skipped = 0
    for doc in all_trains.values():
        result = col.update_one(
            {"train_number": doc["train_number"]},
            {"$set": doc},
            upsert=True,
        )
        if result.upserted_id:
            upserted += 1
        elif result.modified_count:
            updated += 1
        else:
            skipped += 1

    col.create_index("train_number", unique=True)
    col.create_index("source_city")
    col.create_index("destination_city")
    col.create_index("stop_cities")
    col.create_index("serves_somnath")
    for day in DAYS:
        col.create_index(f"running_days.{day}")

    print(f"done — inserted {upserted}, updated {updated}, unchanged {skipped}")
    print(f"total in collection: {col.count_documents({})}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", action="store_true", help="drop collection before loading")
    args = parser.parse_args()
    run(drop=args.drop)
