"""Create the knowledge_chunks collection (RAG corpus from trust_pages) and its indexes."""

VERSION = "002_add_knowledge_chunks"


def apply(clean_db):
    clean_db["knowledge_chunks"].create_index("chunk_id", unique=True)
    clean_db["knowledge_chunks"].create_index("page_url")
    clean_db["knowledge_chunks"].create_index([("heading", "text"), ("content", "text")])
