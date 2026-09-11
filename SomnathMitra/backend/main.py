from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services import embedder, retriever
from routers import chat, pois, restaurants, qa


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Verify embedding endpoint is reachable
    embedder.warmup()

    # Load knowledge_chunks embeddings into memory
    n = retriever.reload()
    print(f"[startup] retriever loaded {n} knowledge chunks")
    if n == 0:
        print("[startup] WARNING: no embedded chunks found — run data/etl/embed_knowledge_chunks.py first")

    yield


app = FastAPI(
    title="Somnath Mitra API",
    description="Pilgrim assistant for Somnath Jyotirlinga temple",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(pois.router)
app.include_router(restaurants.router)
app.include_router(qa.router)


@app.get("/health")
def health():
    from services import retriever as r
    return {
        "status": "ok",
        "chunks_indexed": len(r._chunks),
    }


@app.post("/retriever/reload")
def reload_retriever():
    """Hot-reload embeddings from MongoDB without restarting the server."""
    n = retriever.reload()
    return {"chunks_indexed": n}
