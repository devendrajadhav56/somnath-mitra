"""
Read-only audit of the raw somnath_chatbot database.
Run with: python audit/audit_raw_data.py
Reprints the data-quality findings that shaped the cleaning rules in etl/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import get_raw_db  # noqa: E402


def audit_collections(db):
    print("=== Collections & counts ===")
    for name in db.list_collection_names():
        print(f"  {name}: {db[name].count_documents({})} docs")


def audit_pois(db):
    coll = db["pois"]
    print("\n=== pois ===")
    print("distinct category:", coll.distinct("category"))
    print("distinct subtype:", coll.distinct("subtype"))
    print("missing name:", coll.count_documents({"$or": [{"name": None}, {"name": ""}]}))
    print("empty address:", coll.count_documents({"address": ""}))
    print("empty phone:", coll.count_documents({"phone": ""}))

    dups = list(
        coll.aggregate(
            [
                {"$group": {"_id": "$name", "count": {"$sum": 1}}},
                {"$match": {"count": {"$gt": 1}}},
                {"$sort": {"count": -1}},
            ]
        )
    )
    print(f"duplicate names ({len(dups)} groups):", dups[:10])

    bbox = list(
        coll.aggregate(
            [
                {
                    "$project": {
                        "lon": {"$arrayElemAt": ["$location.coordinates", 0]},
                        "lat": {"$arrayElemAt": ["$location.coordinates", 1]},
                    }
                },
                {
                    "$group": {
                        "_id": None,
                        "minLat": {"$min": "$lat"},
                        "maxLat": {"$max": "$lat"},
                        "minLon": {"$min": "$lon"},
                        "maxLon": {"$max": "$lon"},
                    }
                },
            ]
        )
    )
    print("bounding box:", bbox[0] if bbox else None)


def audit_restaurants(db):
    coll = db["restaurants"]
    print("\n=== restaurants ===")
    print("distinct status:", coll.distinct("status"))
    print("distinct score:", coll.distinct("score"))
    print("missing/zero rating:", coll.count_documents({"$or": [{"rating": {"$exists": False}}, {"rating": None}]}))

    dups = list(
        coll.aggregate(
            [
                {"$group": {"_id": "$name", "count": {"$sum": 1}}},
                {"$match": {"count": {"$gt": 1}}},
                {"$sort": {"count": -1}},
            ]
        )
    )
    print(f"duplicate names ({len(dups)} groups):", dups[:10])

    bbox = list(
        coll.aggregate(
            [
                {
                    "$group": {
                        "_id": None,
                        "minLat": {"$min": "$lat"},
                        "maxLat": {"$max": "$lat"},
                        "minLon": {"$min": "$lon"},
                        "maxLon": {"$max": "$lon"},
                    }
                }
            ]
        )
    )
    print("bounding box:", bbox[0] if bbox else None)

    sample_categories = coll.distinct("category")[:20]
    print("sample distinct category values:", sample_categories)


def main():
    db = get_raw_db()
    audit_collections(db)
    audit_pois(db)
    audit_restaurants(db)


if __name__ == "__main__":
    main()
