from fastapi import APIRouter, Query

import db

router = APIRouter(prefix="/pois", tags=["pois"])


@router.get("/nearby")
def nearby_pois(
    lat: float = Query(..., description="Latitude of the user's location"),
    lon: float = Query(..., description="Longitude of the user's location"),
    radius_m: int = Query(2000, description="Search radius in metres", ge=1, le=50000),
    limit: int = Query(20, ge=1, le=100),
    category: str | None = Query(None, description="Filter by category"),
):
    clean = db.get_clean_db()
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

    docs = list(
        clean["pois"]
        .find(query, {"_id": 0, "embedding": 0})
        .limit(limit)
    )
    return {"count": len(docs), "pois": docs}


@router.get("/categories")
def list_categories():
    clean = db.get_clean_db()
    cats = clean["pois"].distinct("category")
    return {"categories": sorted(c for c in cats if c)}
