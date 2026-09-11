"""Add a sparse index on knowledge_chunks.embedding to support efficient queries for un-embedded docs."""

VERSION = "003_add_embedding_index"


def apply(clean_db):
    clean_db["knowledge_chunks"].create_index("embedding", sparse=True)
