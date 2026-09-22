"""Tool execution — takes a list of tool calls from the intent detector and
returns a formatted string ready to inject into the LLM context."""
from __future__ import annotations

import config
import db
from services import router as _router
from services.applog import get_logger, log_debug, log_step

log = get_logger(__name__)

TEMPLE_LAT = config.TEMPLE_LAT
TEMPLE_LON = config.TEMPLE_LON


# ── Formatters ────────────────────────────────────────────────────────────────

def _fmt_restaurant(r: dict) -> str:
    parts = [r.get("name", "Unknown")]
    if r.get("category"):
        parts.append(r["category"])
    if r.get("rating") is not None:
        rev = f" ({r['reviews']} reviews)" if r.get("reviews") else ""
        parts.append(f"rating {r['rating']}{rev}")
    if r.get("distance_from_temple_km") is not None:
        parts.append(f"{r['distance_from_temple_km']} km from temple")
    if r.get("google_maps_url"):
        parts.append(f"maps: {r['google_maps_url']}")
    return " | ".join(parts)


def _fmt_poi(p: dict) -> str:
    parts = [p.get("name", "Unknown")]
    if p.get("category"):
        parts.append(p["category"])
    if p.get("subtype"):
        parts.append(p["subtype"])
    if p.get("address"):
        parts.append(p["address"])
    if p.get("phone"):
        parts.append(f"phone: {p['phone']}")
    return " | ".join(parts)


def _fmt_train(t: dict, highlight_city: str | None = None) -> str:
    parts = [f"[{t['train_number']}] {t['train_name']} ({t.get('train_type', '')})"]
    parts.append(f"{t['source_station']} → {t['destination_station']}")
    if t.get("running_days_text"):
        parts.append(t["running_days_text"])
    v = t.get("veraval") or {}
    if v.get("arrival"):
        suffix = " (terminates)" if t.get("terminates_at_veraval") else ""
        parts.append(f"Veraval arr: {v['arrival']}{suffix}")
    if v.get("departure"):
        suffix = " (originates)" if t.get("originates_at_veraval") else ""
        parts.append(f"Veraval dep: {v['departure']}{suffix}")
    s = t.get("somnath") or {}
    if s.get("arrival"):
        parts.append(f"Somnath arr: {s['arrival']}")
    if s.get("departure"):
        parts.append(f"Somnath dep: {s['departure']}")
    # If the matched city is an intermediate stop, show its timing
    if highlight_city:
        for stop in t.get("intermediate_stops", []):
            if stop["city"].lower() == highlight_city.lower():
                arr = stop.get("arrival") or "—"
                dep = stop.get("departure") or "—"
                day = f" day {stop['day']}" if stop.get("day") and stop["day"] > 1 else ""
                parts.append(f"[passes through {stop['station_name']} arr:{arr} dep:{dep}{day}]")
                break
    return " | ".join(parts)


def _fmt_bus(r: dict) -> str:
    origin = r.get("origin", {}).get("name", "?")
    destination = r.get("destination", {}).get("name", "?")
    parts = [f"{origin} → {destination}"]
    if r.get("distance_km"):
        parts.append(f"{r['distance_km']} km")
    if r.get("typical_duration_min"):
        h, m = divmod(int(r["typical_duration_min"]), 60)
        parts.append(f"~{h}h {m}min" if h else f"~{m}min")
    if r.get("operator_count"):
        trips = f", {r['trip_count']} trips/day" if r.get("trip_count") else ""
        parts.append(f"{r['operator_count']} operators{trips}")
    if r.get("first_departure") and r.get("last_departure"):
        parts.append(f"Departs: {r['first_departure']}–{r['last_departure']}")
    if r.get("fare_range_inr"):
        lo, hi = r["fare_range_inr"]
        parts.append(f"Fare: ₹{int(lo)}–₹{int(hi)}")
    return " | ".join(parts)


def _fmt_temple_info(doc: dict) -> str:
    lines = [f"[{doc['title']}]", doc.get("content", "")]
    if doc.get("caveats"):
        lines.append("Note: " + "; ".join(doc["caveats"]))
    if doc.get("items"):
        for item in doc["items"]:
            label = item.get("name") or item.get("dept") or item.get("q", "")
            rest = []
            for f in ("note", "distance_km", "when", "phone", "email", "a"):
                if item.get(f) is not None:
                    rest.append(f"{f}: {item[f]}")
            lines.append("  • " + label + (" | " + " | ".join(rest) if rest else ""))
    return "\n".join(lines)


# ── Individual tool runners ───────────────────────────────────────────────────

def _run_search_restaurants(params: dict, lat: float, lon: float) -> str:
    radius_m = params.get("radius_m", 2000)
    min_rating = params.get("min_rating")
    limit = params.get("limit", 10)

    query: dict = {
        "location": {
            "$nearSphere": {
                "$geometry": {"type": "Point", "coordinates": [lon, lat]},
                "$maxDistance": radius_m,
            }
        }
    }
    if min_rating is not None:
        query["rating"] = {"$gte": float(min_rating)}

    docs = list(db.get_clean_db()["restaurants"].find(query, {"_id": 0}).limit(limit))
    if not docs:
        return f"No restaurants found within {radius_m}m of the temple."
    lines = [f"[Restaurants within {radius_m}m of Somnath temple — {len(docs)} results]"]
    lines += [f"- {_fmt_restaurant(r)}" for r in docs]
    return "\n".join(lines)


def _fmt_medical(r: dict) -> str:
    parts = [r.get("name", "Unknown")]
    if r.get("category"):
        parts.append(r["category"])
    if r.get("rating") is not None:
        rev = f" ({r['reviews']} reviews)" if r.get("reviews") else ""
        parts.append(f"rating {r['rating']}{rev}")
    if r.get("phone"):
        parts.append(f"phone: {r['phone']}")
    if r.get("business_status"):
        parts.append(r["business_status"])
    if r.get("distance_from_temple_km") is not None:
        parts.append(f"{r['distance_from_temple_km']} km from temple")
    if r.get("google_maps_url"):
        parts.append(f"maps: {r['google_maps_url']}")
    return " | ".join(parts)


def _run_search_hospitals(params: dict, lat: float, lon: float) -> str:
    radius_m = params.get("radius_m", 5000)
    limit = params.get("limit", 15)
    query: dict = {
        "location": {
            "$nearSphere": {
                "$geometry": {"type": "Point", "coordinates": [lon, lat]},
                "$maxDistance": radius_m,
            }
        }
    }
    docs = list(db.get_clean_db()["hospitals"].find(query, {"_id": 0}).limit(limit))
    if not docs:
        return f"No hospitals found within {radius_m}m of the temple."
    lines = [f"[Hospitals within {radius_m}m of Somnath temple — {len(docs)} results]"]
    lines += [f"- {_fmt_medical(r)}" for r in docs]
    return "\n".join(lines)


def _run_search_pharmacies(params: dict, lat: float, lon: float) -> str:
    radius_m = params.get("radius_m", 3000)
    limit = params.get("limit", 15)
    query: dict = {
        "location": {
            "$nearSphere": {
                "$geometry": {"type": "Point", "coordinates": [lon, lat]},
                "$maxDistance": radius_m,
            }
        }
    }
    docs = list(db.get_clean_db()["pharmacies"].find(query, {"_id": 0}).limit(limit))
    if not docs:
        return f"No pharmacies found within {radius_m}m of the temple."
    lines = [f"[Pharmacies within {radius_m}m of Somnath temple — {len(docs)} results]"]
    lines += [f"- {_fmt_medical(r)}" for r in docs]
    return "\n".join(lines)


def _run_search_pois(params: dict, lat: float, lon: float) -> str:
    radius_m = params.get("radius_m", 5000)
    category = params.get("category")
    limit = params.get("limit", 15)

    query: dict = {
        "location": {
            "$nearSphere": {
                "$geometry": {"type": "Point", "coordinates": [lon, lat]},
                "$maxDistance": radius_m,
            }
        }
    }
    if category:
        query["category"] = category

    docs = list(db.get_clean_db()["pois"].find(query, {"_id": 0}).limit(limit))
    if not docs:
        return f"No points of interest found within {radius_m}m of the temple."
    lines = [f"[Points of interest within {radius_m}m of Somnath temple — {len(docs)} results]"]
    lines += [f"- {_fmt_poi(p)}" for p in docs]
    return "\n".join(lines)


def _run_search_trains(params: dict) -> str:
    origin_city = params.get("origin_city")
    day = params.get("day", "").lower() if params.get("day") else None
    limit = params.get("limit", 10)

    query: dict = {}
    if origin_city:
        pat = {"$regex": origin_city, "$options": "i"}
        query["$or"] = [
            {"source_city": pat},
            {"destination_city": pat},
            {"source_station": pat},
            {"destination_station": pat},
            {"stop_cities": pat},          # trains that pass through the city
        ]
    if day in {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}:
        query[f"running_days.{day}"] = True

    docs = list(db.get_clean_db()["train_routes"].find(query, {"_id": 0}).limit(limit))
    if not docs:
        desc = f" from/to or passing through {origin_city}" if origin_city else ""
        return f"No train services found{desc}."

    # Separate direct trains from pass-through trains for clarity
    direct, via = [], []
    if origin_city:
        pat_lower = origin_city.lower()
        for t in docs:
            if (pat_lower in (t.get("source_city") or "").lower() or
                    pat_lower in (t.get("destination_city") or "").lower() or
                    pat_lower in (t.get("source_station") or "").lower() or
                    pat_lower in (t.get("destination_station") or "").lower()):
                direct.append((t, None))
            else:
                via.append((t, origin_city))
    else:
        direct = [(t, None) for t in docs]

    lines = [f"[Train services at Veraval/Somnath — {len(docs)} results]"]
    if direct:
        lines.append("Direct (originates/terminates):")
        lines += [f"- {_fmt_train(t, city)}" for t, city in direct]
    if via:
        lines.append(f"Also passes through {origin_city}:")
        lines += [f"- {_fmt_train(t, city)}" for t, city in via]
    return "\n".join(lines)


def _run_search_buses(params: dict) -> str:
    origin_city = params.get("origin_city")
    destination_city = params.get("destination_city")
    limit = params.get("limit", 10)

    query: dict = {}
    if origin_city:
        query["origin.name"] = {"$regex": origin_city, "$options": "i"}
    if destination_city:
        query["destination.name"] = {"$regex": destination_city, "$options": "i"}

    docs = list(db.get_clean_db()["bus_routes"].find(query, {"_id": 0}).limit(limit))
    if not docs:
        desc = f" from {origin_city}" if origin_city else ""
        return f"No bus routes found{desc}."
    lines = [f"[Bus routes — {len(docs)} results]"]
    lines += [f"- {_fmt_bus(r)}" for r in docs]
    return "\n".join(lines)


def _run_get_temple_info(params: dict) -> str:
    key = params.get("key")
    doc = db.get_clean_db()["temple_info"].find_one({"key": key}, {"_id": 0})
    if not doc:
        return f"No temple info found for key: {key}"
    return _fmt_temple_info(doc)


def _run_search_shop(params: dict) -> tuple[str, list[dict]]:
    category = params.get("category")
    min_price = params.get("min_price")
    max_price = params.get("max_price")
    limit = params.get("limit", 8)

    query: dict = {}
    if category:
        query["category"] = {"$regex": category, "$options": "i"}
    price_filter: dict = {}
    if min_price is not None:
        price_filter["$gte"] = float(min_price)
    if max_price is not None:
        price_filter["$lte"] = float(max_price)
    if price_filter:
        query["price"] = price_filter

    docs = list(db.get_clean_db()["shop_products"].find(query, {"_id": 0}).limit(limit))
    if not docs:
        desc = f" in category '{category}'" if category else ""
        return f"No shop products found{desc}.", []

    lines = [f"[Somnath Temple Shop — {len(docs)} items]"]
    for d in docs:
        parts = [d["name"], d["category"], f"₹{int(d['price'])}"]
        if d.get("product_url"):
            parts.append(d["product_url"])
        lines.append("- " + " | ".join(parts))

    products = [
        {
            "name":        d["name"],
            "category":    d.get("category"),
            "price":       d.get("price"),
            "image_url":   d.get("image_url"),
            "product_url": d.get("product_url"),
        }
        for d in docs
    ]
    return "\n".join(lines), products


# ── Public API ────────────────────────────────────────────────────────────────

def execute_tools(
    tool_calls: list[dict],
    user_lat: float | None = None,
    user_lon: float | None = None,
) -> tuple[str, list[dict]]:
    """
    Execute a list of tool calls as decided by the intent detector.
    Returns (structured_text, products) where structured_text is injected into
    the LLM context and products is a list of shop product dicts for the frontend.
    """
    if not tool_calls:
        return "", []

    lat = user_lat or TEMPLE_LAT
    lon = user_lon or TEMPLE_LON
    sections: list[str] = []
    products: list[dict] = []

    for call in tool_calls:
        name = call.get("name")
        params = call.get("params", {})
        section = ""

        if name == "search_restaurants":
            section = _run_search_restaurants(params, lat, lon)
        elif name == "search_pois":
            section = _run_search_pois(params, lat, lon)
        elif name == "get_temple_info":
            section = _run_get_temple_info(params)
        elif name == "plan_route_to_somnath":
            origin = params.get("origin")
            if origin:
                section = _router.plan_route(origin)
        elif name == "search_hospitals":
            section = _run_search_hospitals(params, lat, lon)
        elif name == "search_pharmacies":
            section = _run_search_pharmacies(params, lat, lon)
        elif name == "search_shop":
            section, prods = _run_search_shop(params)
            products.extend(prods)
        else:
            log.warning("unknown tool requested: name=%s params=%s", name, params)
            continue

        if not section:
            continue

        sections.append(section)
        # Runners signal "nothing found" with a message starting "No ..."
        miss = section.lstrip().startswith("No ")
        if miss:
            log.warning("tool returned no results: name=%s params=%s", name, params)
        log_step("tool", name=name, chars=len(section), miss=miss)
        log_debug("tool_result", name=name, result=section)

    return "\n\n".join(sections), products
