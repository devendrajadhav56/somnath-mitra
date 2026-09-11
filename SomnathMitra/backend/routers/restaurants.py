from fastapi import APIRouter, Query

import db

router = APIRouter(prefix="/restaurants", tags=["restaurants"])


@router.get("/nearby")
def nearby_restaurants(
    lat: float = Query(..., description="Latitude of the user's location"),
    lon: float = Query(..., description="Longitude of the user's location"),
    radius_m: int = Query(3000, description="Search radius in metres", ge=1, le=60000),
    limit: int = Query(20, ge=1, le=100),
    min_rating: float | None = Query(None, ge=1.0, le=5.0),
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
    if min_rating is not None:
        query["rating"] = {"$gte": min_rating}

    docs = list(
        clean["restaurants"]
        .find(query, {"_id": 0})
        .limit(limit)
    )
    return {"count": len(docs), "restaurants": docs}
