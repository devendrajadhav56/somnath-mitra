from pymongo import MongoClient
from pymongo.database import Database

import config

_client: MongoClient | None = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(config.MONGODB_URI)
    return _client


def get_clean_db() -> Database:
    return get_client()[config.CLEAN_DB_NAME]
