import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

MONGODB_URI = os.environ["MONGODB_URI"]
RAW_DB_NAME = os.environ.get("RAW_DB_NAME", "somnath_chatbot")
CLEAN_DB_NAME = os.environ.get("CLEAN_DB_NAME", "somnath_clean")

# Somnath temple coordinates, used as the center point for geo-cleaning rules.
TEMPLE_LAT = 20.888
TEMPLE_LON = 70.401
