import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

MONGODB_URI = os.environ["MONGODB_URI"]
RAW_DB_NAME = os.environ.get("RAW_DB_NAME", "somnath_chatbot")
CLEAN_DB_NAME = os.environ.get("CLEAN_DB_NAME", "somnath_clean")

TEMPLE_LAT = 20.888
TEMPLE_LON = 70.401

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma4:12b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "ollama")

# Separate lighter model for intent classification (runs on every request)
INTENT_MODEL = os.environ.get("INTENT_MODEL", "qwen3.5:9b")

EMBED_BASE_URL = os.environ.get("EMBED_BASE_URL", "http://localhost:11434/v1")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")

TOP_K_CHUNKS = int(os.environ.get("TOP_K_CHUNKS", "5"))

# Minimum cosine similarity for a RAG chunk to be injected into context.
# Chunks below this are dropped; if all are below it, no context is injected.
RAG_MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.5"))

# Max tokens for the main chat LLM response
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "2048"))

# Logging verbosity: INFO for normal operation, DEBUG to also capture full
# payloads (retrieved chunks, tool output, LLM prompt, full replies).
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

NOMINATIM_URL = os.environ.get("NOMINATIM_URL", "https://nominatim.openstreetmap.org")
NOMINATIM_USER_AGENT = os.environ.get("NOMINATIM_USER_AGENT", "SomnathMitra/0.1")
