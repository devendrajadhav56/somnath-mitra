from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from services import embedder, retriever
from services.applog import setup_logging, new_req_id, set_req_id, log_step
from routers import chat, pois, restaurants, qa


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()

    embedder.warmup()

    n = retriever.reload()
    log_step("startup", chunks_indexed=n)
    if n == 0:
        log_step("startup", warning="no embedded chunks — run data/etl/embed_knowledge_chunks.py")

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


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    req_id = new_req_id()
    set_req_id(req_id)
    log_step("request_in", method=request.method, path=request.url.path)
    response = await call_next(request)
    log_step("request_out", status=response.status_code)
    return response

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
