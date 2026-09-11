"""Hierarchical route planner: anywhere in India → Somnath.

Pipeline:
  1. find_nearest_hubs(lat, lon)  — MongoDB $nearSphere over access_hubs
  2. build_journeys(origin, hubs) — construct candidate journey objects
  3. rank_journeys(journeys)      — score by transfers, travel time, frequency
  4. format_context(origin, journeys) — structured string for the LLM

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

MAX_HUB_DISTANCE_KM = 150  # don't suggest a hub that's further than this
MAX_HUBS = 4               # top N hubs to consider
MAX_JOURNEYS = 5           # top N journeys to return


# ── Geo helper ────────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ── Hub finder ────────────────────────────────────────────────────────────────

def find_nearest_hubs(lat: float, lon: float, max_km: float = MAX_HUB_DISTANCE_KM, n: int = MAX_HUBS) -> list[dict]:
    """
    Return up to n access hubs sorted by distance from (lat, lon).
    Each hub doc includes its pre-joined trains list.
    """
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


# ── Journey builder ───────────────────────────────────────────────────────────

def _score(journey: dict) -> float:
    """Lower is better."""
    score = 0.0
    # Penalise hub distance (ground travel to reach the station)
    score += journey["hub_distance_km"] * 2.0
    # Penalise low-frequency trains (weekly << daily)
    freq = journey["trains_per_week"]
    score += max(0, (7 - freq)) * 30
    return score


def build_journeys(origin: dict, hubs: list[dict]) -> list[dict]:
    """
    For each hub, construct one journey candidate per unique train.
    Returns a flat list of journey dicts.
    """
    journeys = []
    seen = set()  # avoid duplicate trip_ids across hubs

    for hub in hubs:
        for train in hub.get("trains", []):
            trip_id = train["trip_id"]
            if trip_id in seen:
                continue
            seen.add(trip_id)

            days = train.get("running_days", {})
            trains_per_week = sum(1 for v in days.values() if v)

            journeys.append({
                "hub_station_code": hub["station_code"],
                "hub_station_name": hub["station_name"],
                "hub_distance_km": hub["distance_km"],
                "trip_id": trip_id,
                "train_name": train["train_name"],
                "departure_from_hub": train.get("departure"),
                "departure_day": train.get("departure_day", 0),
                "veraval_arrival": train.get("veraval_arrival"),
                "veraval_day": train.get("veraval_day", 0),
                "running_days": days,
                "running_days_text": train.get("running_days_text", ""),
                "trains_per_week": trains_per_week,
            })

    return journeys


def rank_journeys(journeys: list[dict]) -> list[dict]:
    return sorted(journeys, key=_score)[:MAX_JOURNEYS]


# ── Formatter ─────────────────────────────────────────────────────────────────

def _fmt_journey(j: dict, idx: int) -> str:
    lines = [f"Option {idx}: Train {j['trip_id']} — {j['train_name']}"]
    hub = j["hub_station_name"].title()
    dist = j["hub_distance_km"]
    lines.append(f"  Step 1: Travel to {hub} (~{dist} km from your location by road/bus)")
    dep = j["departure_from_hub"] or "?"
    lines.append(f"  Step 2: Board train at {hub}, departs {dep}")
    lines.append(f"  Runs: {j['running_days_text']}")
    arr = j["veraval_arrival"] or "?"
    day_note = f" (next day)" if (j.get("veraval_day", 0) or 0) > (j.get("departure_day", 0) or 0) else ""
    lines.append(f"  Arrives Veraval: {arr}{day_note}")
    lines.append(f"  Step 3: From Veraval, take a taxi/auto-rickshaw to Somnath temple (~{VERAVAL_TO_SOMNATH_KM} km, ~{VERAVAL_TO_SOMNATH_MIN} min)")
    return "\n".join(lines)


def format_context(origin: dict, journeys: list[dict]) -> str:
    name = origin.get("display_name") or origin.get("name") or "your location"
    if not journeys:
        return f"[Route to Somnath from {name}]\nNo direct train connections found within {MAX_HUB_DISTANCE_KM} km. Suggest checking bus options or a broader search."

    parts = [f"[Route to Somnath from {name}]"]
    for i, j in enumerate(journeys, 1):
        parts.append(_fmt_journey(j, i))
    parts.append(
        f"\nNote: All trains terminate at Veraval Junction (VRL), 6 km from Somnath temple. "
        f"Taxi/auto-rickshaw from Veraval to the temple takes ~{VERAVAL_TO_SOMNATH_MIN} minutes."
    )
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

    hubs = find_nearest_hubs(lat, lon)
    if not hubs:
        return format_context(origin, [])

    journeys = build_journeys(origin, hubs)
    ranked = rank_journeys(journeys)
    return format_context(origin, ranked)
