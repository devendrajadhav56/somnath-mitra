import asyncio

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services import retriever, llm
from services.intent import detect_intent
from services.location import resolve_origin
from services.tools import execute_tools

router = APIRouter(prefix="/chat", tags=["chat"])


class Message(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []
    stream: bool = True
    user_lat: float | None = None
    user_lon: float | None = None


class ChatResponse(BaseModel):
    reply: str
    intent: dict = {}
    sources: list[dict] = []
    origin: dict = {}


@router.post("", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    history = [m.model_dump() for m in req.history]

    # Step 1 — resolve origin and detect intent in parallel (independent tasks)
    origin, intent = await asyncio.gather(
        resolve_origin(req.message, history),
        detect_intent(req.message, history),
    )
    origin = origin or {}

    # Step 2 — inject resolved origin into plan_route_to_somnath tool params
    for tool_call in intent.get("tools", []):
        if tool_call.get("name") == "plan_route_to_somnath":
            tool_call["params"]["origin"] = origin

    # Step 4 — fetch structured data for the decided tools
    structured = execute_tools(intent["tools"], req.user_lat, req.user_lon)

    # Step 5 — RAG retrieval only if intent says so
    chunks = retriever.search(req.message) if intent["use_rag"] else []

    # Step 6 — LLM call
    if req.stream:
        async def token_stream():
            async for token in llm.chat_stream(req.message, history, chunks, structured):
                yield token
        return StreamingResponse(token_stream(), media_type="text/plain")

    reply = await llm.chat(req.message, history, chunks, structured)
    sources = [
        {
            "chunk_id": c["chunk_id"],
            "heading": c.get("heading"),
            "page_url": c.get("page_url"),
            "score": c["score"],
        }
        for c in chunks
    ]
    return ChatResponse(reply=reply, intent=intent, sources=sources, origin=origin)
