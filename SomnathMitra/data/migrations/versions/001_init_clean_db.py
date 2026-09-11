"""Create somnath_clean collections and their indexes."""

VERSION = "001_init_clean_db"


def apply(clean_db):
    # pois: geo queries + dedup-safe upsert key
    clean_db["pois"].create_index([("location", "2dsphere")])
    clean_db["pois"].create_index("source_id", unique=True)

    # restaurants: same pattern, plus the geo-excluded overflow bucket
    clean_db["restaurants"].create_index([("location", "2dsphere")])
    clean_db["restaurants"].create_index("source_id", unique=True)
    clean_db["restaurants_excluded"].create_index([("location", "2dsphere")])
    clean_db["restaurants_excluded"].create_index("source_id", unique=True)

    # scaffolded for later phases: no data yet, but shape is ready
    clean_db["shops"].create_index([("location", "2dsphere")])
    clean_db["shops"].create_index("source_id", unique=True)
    clean_db["temple_info"].create_index("key", unique=True)
    clean_db["contacts"].create_index("created_at")
