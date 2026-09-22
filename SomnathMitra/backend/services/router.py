"""Hierarchical route planner: anywhere in India → Somnath.

Pipeline:
  1. find_nearest_hubs / find_nearest_bus_hubs — MongoDB $nearSphere
  2. build_journeys / build_bus_journeys       — candidate journey objects
  3. rank_journeys / rank_bus_journeys         — score by distance + frequency
  4. format_context(origin, train_journeys, bus_journeys) — LLM context string

Public API:
  plan_route(origin_location: dict) -> str
    origin_location is the dict returned by services.location.resolve_origin()
"""
from __future__ import annotations

import math

import config
import db

VERAVAL_LAT = 20.913
VERAVAL_LON = 70.370
SOMNATH_LAT = config.TEMPLE_LAT
SOMNATH_LON = config.TEMPLE_LON
VERAVAL_TO_SOMNATH_KM = 6.0
VERAVAL_TO_SOMNATH_MIN = 20  # taxi/auto

MAX_HUB_DISTANCE_KM = 500  # don't suggest a hub that's further than this
MAX_HUBS = 4               # top N train hubs to consider
MAX_BUS_HUBS = 4           # top N bus hubs to consider
MAX_FLIGHT_HUBS = 2        # top N flight hubs to consider
MAX_JOURNEYS = 5           # top N train journeys to return
MAX_BUS_JOURNEYS = 3       # top N bus journeys to return
MAX_FLIGHT_JOURNEYS = 3    # top N flight journeys to return
HUB_TRAVEL_THRESHOLD_KM = 5  # below this, skip "travel to hub" step
NEARBY_HUB_RADIUS_KM = 10   # show all boarding stops within this radius for the same train


# ── Geo helper ────────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ── Hub finders ───────────────────────────────────────────────────────────────

def find_nearest_hubs(lat: float, lon: float, max_km: float = MAX_HUB_DISTANCE_KM, n: int = MAX_HUBS) -> list[dict]:
    col = db.get_clean_db()["access_hubs"]
    results = list(col.find(
        {
            "location": {
                "$nearSphere": {
                    "$geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "$maxDistance": int(max_km * 1000),
                }
            }
        },
        {"_id": 0},
    ).limit(n))

    for hub in results:
        hub["distance_km"] = round(_haversine_km(lat, lon, hub["lat"], hub["lon"]), 1)

    return results


def find_nearest_flight_hubs(lat: float, lon: float, max_km: float = MAX_HUB_DISTANCE_KM, n: int = MAX_FLIGHT_HUBS) -> list[dict]:
    col = db.get_clean_db()["flight_access_hubs"]
    results = list(col.find(
        {
            "location": {
                "$nearSphere": {
                    "$geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "$maxDistance": int(max_km * 1000),
                }
            }
        },
        {"_id": 0},
    ).limit(n))

    for hub in results:
        hub["distance_km"] = round(_haversine_km(lat, lon, hub["lat"], hub["lon"]), 1)

    return results


def find_nearest_bus_hubs(lat: float, lon: float, max_km: float = MAX_HUB_DISTANCE_KM, n: int = MAX_BUS_HUBS) -> list[dict]:
    col = db.get_clean_db()["bus_access_hubs"]
    results = list(col.find(
        {
            "location": {
                "$nearSphere": {
                    "$geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "$maxDistance": int(max_km * 1000),
                }
            }
        },
        {"_id": 0},
    ).limit(n))

    for hub in results:
        hub["distance_km"] = round(_haversine_km(lat, lon, hub["lat"], hub["lon"]), 1)

    return results


# ── Train journey builder ─────────────────────────────────────────────────────

def _score_train(journey: dict) -> float:
    score = journey["hub_distance_km"] * 2.0
    freq = journey["trains_per_week"]
    score += max(0, (7 - freq)) * 30
    return score


def build_journeys(origin: dict, hubs: list[dict]) -> list[dict]:
    nearby = [h for h in hubs if h["distance_km"] <= NEARBY_HUB_RADIUS_KM]
    further = [h for h in hubs if h["distance_km"] > NEARBY_HUB_RADIUS_KM]

    # For nearby hubs: collect ALL boarding stops per train so the LLM can
    # tell the user "you can board at Sabarmati OR Gandhigram".
    nearby_by_trip: dict[str, dict] = {}
    for hub in sorted(nearby, key=lambda h: h["distance_km"]):
        for train in hub.get("trains", []):
            trip_id = train["trip_id"]
            option = {
                "hub_station_code": hub["station_code"],
                "hub_station_name": hub["station_name"],
                "hub_distance_km":  hub["distance_km"],
                "departure":        train.get("departure"),
            }
            if trip_id not in nearby_by_trip:
                nearby_by_trip[trip_id] = {"train": train, "options": [option]}
            else:
                nearby_by_trip[trip_id]["options"].append(option)

    journeys = []
    seen: set[str] = set()

    for trip_id, entry in nearby_by_trip.items():
        seen.add(trip_id)
        train = entry["train"]
        # Sort boarding options by departure time (earliest first), then distance
        options = sorted(
            entry["options"],
            key=lambda o: (o["departure"] or "99:99", o["hub_distance_km"]),
        )
        nearest = min(options, key=lambda o: o["hub_distance_km"])

        days = train.get("running_days", {})
        journeys.append({
            "hub_station_code":  nearest["hub_station_code"],
            "hub_station_name":  nearest["hub_station_name"],
            "hub_distance_km":   nearest["hub_distance_km"],
            "trip_id":           trip_id,
            "train_name":        train["train_name"],
            "departure_from_hub": nearest["departure"],
            "departure_day":     train.get("departure_day", 0),
            "veraval_arrival":   train.get("veraval_arrival"),
            "veraval_day":       train.get("veraval_day", 0),
            "running_days":      days,
            "running_days_text": train.get("running_days_text", ""),
            "trains_per_week":   sum(1 for v in days.values() if v),
            "boarding_options":  options if len(options) > 1 else [],
        })

    for hub in further:
        for train in hub.get("trains", []):
            trip_id = train["trip_id"]
            if trip_id in seen:
                continue
            seen.add(trip_id)
            days = train.get("running_days", {})
            journeys.append({
                "hub_station_code":  hub["station_code"],
                "hub_station_name":  hub["station_name"],
                "hub_distance_km":   hub["distance_km"],
                "trip_id":           trip_id,
                "train_name":        train["train_name"],
                "departure_from_hub": train.get("departure"),
                "departure_day":     train.get("departure_day", 0),
                "veraval_arrival":   train.get("veraval_arrival"),
                "veraval_day":       train.get("veraval_day", 0),
                "running_days":      days,
                "running_days_text": train.get("running_days_text", ""),
                "trains_per_week":   sum(1 for v in days.values() if v),
                "boarding_options":  [],
            })

    return journeys


def rank_journeys(journeys: list[dict]) -> list[dict]:
    return sorted(journeys, key=_score_train)[:MAX_JOURNEYS]


# ── Bus journey builder ───────────────────────────────────────────────────────

def _score_bus(journey: dict) -> float:
    score = journey["hub_distance_km"] * 2.0
    if journey["kind"] == "aggregated":
        # trip_count is per day; cap at 7 so penalty bottoms out at 0
        trips_per_week = min((journey.get("trip_count") or 1) * 7, 7)
    else:
        trips_per_week = 7  # private operators typically run daily
    score += max(0, (7 - trips_per_week)) * 10
    return score


def build_bus_journeys(bus_hubs: list[dict]) -> list[dict]:
    """One journey candidate per bus service (per operator for outside-Gujarat,
    one aggregated entry for Gujarat hubs)."""
    journeys = []
    seen: set[str] = set()

    for hub in bus_hubs:
        for bus in hub.get("buses", []):
            route_id = bus["route_id"]
            if route_id in seen:
                continue
            seen.add(route_id)

            journeys.append({
                "hub_id":          hub["hub_id"],
                "city":            hub["city"],
                "state":           hub["state"],
                "hub_distance_km": hub["distance_km"],
                "kind":            bus["kind"],
                # aggregated fields
                "operator_count":  bus.get("operator_count"),
                "trip_count":      bus.get("trip_count"),
                "first_departure": bus.get("first_departure"),
                "last_departure":  bus.get("last_departure"),
                # operator fields
                "operator":        bus.get("operator"),
                "bus_type":        bus.get("bus_type"),
                "departure_time":  bus.get("departure_time"),
                "arrival_time":    bus.get("arrival_time"),
                # shared
                "duration_min":    bus.get("duration_min"),
                "fare_min":        bus.get("fare_min"),
                "fare_max":        bus.get("fare_max"),
                "route_id":        route_id,
            })

    return journeys


def rank_bus_journeys(journeys: list[dict]) -> list[dict]:
    return sorted(journeys, key=_score_bus)[:MAX_BUS_JOURNEYS]


# ── Flight journey builder ────────────────────────────────────────────────────

def _score_flight(journey: dict) -> float:
    score = journey["hub_distance_km"] * 2.0
    freq = journey["frequency_per_week"]
    score += max(0, (7 - freq)) * 15
    # Prefer airports closer to Somnath (lower onward time)
    score += (journey.get("onward_min") or 240) / 20
    return score


def build_flight_journeys(flight_hubs: list[dict]) -> list[dict]:
    """One journey candidate per flight service."""
    journeys = []
    seen: set[str] = set()

    for hub in flight_hubs:
        for flight in hub.get("flights", []):
            key = f"{flight['flight_number']}_{flight['destination_iata']}"
            if key in seen:
                continue
            seen.add(key)

            journeys.append({
                "hub_id":           hub["hub_id"],
                "origin_city":      hub["city"],
                "origin_iata":      hub["airport_iata"],
                "hub_distance_km":  hub["distance_km"],
                "flight_number":    flight["flight_number"],
                "airline":          flight["airline"],
                "destination_iata": flight["destination_iata"],
                "destination_name": flight["destination_name"],
                "destination_city": flight["destination_city"],
                "arrival_time":     flight["arrival_time"],
                "days_of_week":     flight["days_of_week"],
                "frequency_per_week": flight["frequency_per_week"],
                "onward_km":        flight.get("onward_km"),
                "onward_min":       flight.get("onward_min"),
                "via_stops":        flight.get("via_stops", []),
            })

    return journeys


def rank_flight_journeys(journeys: list[dict]) -> list[dict]:
    return sorted(journeys, key=_score_flight)[:MAX_FLIGHT_JOURNEYS]


# ── Formatters ────────────────────────────────────────────────────────────────

def _fmt_duration(minutes: int | None) -> str:
    if not minutes:
        return ""
    h, m = divmod(int(minutes), 60)
    return f"~{h}h {m}min" if h else f"~{m}min"


def _fmt_fare(fare_min: float | None, fare_max: float | None) -> str:
    if fare_min and fare_max:
        return f"₹{int(fare_min)}–₹{int(fare_max)}"
    return ""


def _fmt_train_journey(j: dict, idx: int) -> str:
    lines = [f"Train Option {idx}: {j['trip_id']} — {j['train_name']}"]

    boarding_options = j.get("boarding_options") or []
    if boarding_options:
        lines.append("  Boarding stations near you (choose whichever is convenient):")
        for opt in boarding_options:
            name = opt["hub_station_name"].title()
            dist = opt["hub_distance_km"]
            dep = opt["departure"] or "?"
            lines.append(f"    - {name} ({dist} km away) — departs {dep}")
    else:
        hub = j["hub_station_name"].title()
        dist = j["hub_distance_km"]
        dep = j["departure_from_hub"] or "?"
        lines.append(f"  Step 1: Travel to {hub} (~{dist} km from your location by road/bus)")
        lines.append(f"  Step 2: Board train at {hub}, departs {dep}")

    lines.append(f"  Runs: {j['running_days_text']}")
    arr = j["veraval_arrival"] or "?"
    day_note = " (next day)" if (j.get("veraval_day", 0) or 0) > (j.get("departure_day", 0) or 0) else ""
    lines.append(f"  Arrives Veraval: {arr}{day_note}")
    lines.append(f"  Step 3: From Veraval, take a taxi/auto-rickshaw to Somnath temple (~{VERAVAL_TO_SOMNATH_KM} km, ~{VERAVAL_TO_SOMNATH_MIN} min)")
    return "\n".join(lines)


def _fmt_bus_journey(j: dict, idx: int) -> str:
    city = j["city"]
    dist = j["hub_distance_km"]
    dur = _fmt_duration(j.get("duration_min"))
    fare = _fmt_fare(j.get("fare_min"), j.get("fare_max"))

    if j["kind"] == "aggregated":
        lines = [f"Bus Option {idx}: Direct buses from {city} → Somnath"]
        if dist > HUB_TRAVEL_THRESHOLD_KM:
            lines.append(f"  Step 1: Travel to {city} (~{dist} km from your location)")
        parts = []
        if j.get("operator_count"):
            parts.append(f"{j['operator_count']} operators")
        if j.get("trip_count"):
            parts.append(f"{j['trip_count']} buses/day")
        if j.get("first_departure") and j.get("last_departure"):
            parts.append(f"departs {j['first_departure']}–{j['last_departure']}")
        if dur:
            parts.append(dur)
        if fare:
            parts.append(fare)
        lines.append("  " + " | ".join(parts))
    else:
        lines = [f"Bus Option {idx}: {j['operator']} ({city} → Somnath)"]
        if dist > HUB_TRAVEL_THRESHOLD_KM:
            lines.append(f"  Step 1: Travel to {city} (~{dist} km from your location)")
        if j.get("bus_type"):
            lines.append(f"  {j['bus_type']}")
        dep = j.get("departure_time", "?")
        arr = j.get("arrival_time", "?")
        detail_parts = [f"Departs: {dep}", f"Arrives Somnath: {arr}"]
        if dur:
            detail_parts.append(dur)
        if fare:
            detail_parts.append(fare)
        lines.append("  " + " | ".join(detail_parts))

    return "\n".join(lines)


def _fmt_flight_journey(j: dict, idx: int) -> str:
    origin_city = j["origin_city"]
    dist = j["hub_distance_km"]
    dest_name = j["destination_name"]
    dest_city = j["destination_city"]
    days = ", ".join(j["days_of_week"])
    freq = j["frequency_per_week"]
    via = f" via {', '.join(j['via_stops'])}" if j.get("via_stops") else ""

    lines = [f"Flight Option {idx}: {j['airline']} {j['flight_number']} ({origin_city} → {dest_city}{via})"]
    if dist > HUB_TRAVEL_THRESHOLD_KM:
        lines.append(f"  Step 1: Travel to {origin_city} airport (~{dist} km from your location)")
    lines.append(f"  Arrives {dest_name}: {j['arrival_time']} | {days} ({freq}×/week)")
    onward_h = int((j["onward_min"] or 0) // 60)
    onward_m = int((j["onward_min"] or 0) % 60)
    onward_dur = f"{onward_h}h {onward_m}min" if onward_h else f"{onward_m}min"
    lines.append(f"  Then: road from {dest_city} to Somnath ~{j['onward_km']} km (~{onward_dur} by taxi/cab)")
    return "\n".join(lines)


def format_context(origin: dict, train_journeys: list[dict], bus_journeys: list[dict], flight_journeys: list[dict] | None = None) -> str:
    name = origin.get("display_name") or origin.get("name") or "your location"
    flight_journeys = flight_journeys or []
    has_trains = bool(train_journeys)
    has_buses = bool(bus_journeys)
    has_flights = bool(flight_journeys)

    if not has_trains and not has_buses and not has_flights:
        return (
            f"[Route to Somnath from {name}]\n"
            f"No direct train, bus, or flight connections found within {MAX_HUB_DISTANCE_KM} km."
        )

    parts = [f"[Route to Somnath from {name}]"]

    if has_trains:
        parts.append("── By Train ──")
        for i, j in enumerate(train_journeys, 1):
            parts.append(_fmt_train_journey(j, i))
        parts.append(
            f"Note: Trains terminate at Veraval Junction (VRL), {VERAVAL_TO_SOMNATH_KM} km from Somnath. "
            f"Taxi/auto-rickshaw takes ~{VERAVAL_TO_SOMNATH_MIN} min."
        )

    if has_buses:
        parts.append("── By Bus ──")
        for i, j in enumerate(bus_journeys, 1):
            parts.append(_fmt_bus_journey(j, i))

    if has_flights:
        parts.append("── By Flight ──")
        for i, j in enumerate(flight_journeys, 1):
            parts.append(_fmt_flight_journey(j, i))

    return "\n\n".join(parts)


# ── Public API ────────────────────────────────────────────────────────────────

def plan_route(origin: dict) -> str:
    """
    Entry point. origin is the dict from services.location.resolve_origin().
    Returns a formatted context string ready to inject into the LLM.
    """
    lat = origin.get("lat")
    lon = origin.get("lon")
    if lat is None or lon is None:
        return ""

    train_hubs = find_nearest_hubs(lat, lon)
    bus_hubs = find_nearest_bus_hubs(lat, lon)
    flight_hubs = find_nearest_flight_hubs(lat, lon)

    train_journeys = rank_journeys(build_journeys(origin, train_hubs))
    bus_journeys = rank_bus_journeys(build_bus_journeys(bus_hubs))
    flight_journeys = rank_flight_journeys(build_flight_journeys(flight_hubs))

    return format_context(origin, train_journeys, bus_journeys, flight_journeys)
