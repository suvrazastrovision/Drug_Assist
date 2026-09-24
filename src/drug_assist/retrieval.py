"""Small helpers for resumable indexing, review flags, and local retrieval."""

import os
from pathlib import Path

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
COLLECTION_NAME = "drug_documents"


def database_path():
    """Use the same persistent location as the original Windows notebook."""
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share"))
    return base / "Drug_Assist" / "chroma"


def openai_client(project_root):
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(project_root / ".env", override=True)
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise ValueError("Set OPENAI_API_KEY in the project .env file.")
    return OpenAI(api_key=key, timeout=30.0, max_retries=0)


def pending_documents(collection, documents):
    """Reuse saved records; reject positional ID collisions rather than overwrite."""
    saved = collection.get(include=["documents"])
    existing = dict(zip(saved["ids"], saved["documents"]))
    conflicts = [
        doc["id"] for doc in documents
        if doc["id"] in existing and existing[doc["id"]] != doc["document_text"]
    ]
    if conflicts:
        raise ValueError(
            f"Existing IDs refer to different text: {conflicts[:5]}. "
            "Use a new collection for a changed dataset; do not overwrite this one."
        )
    return [doc for doc in documents if doc["id"] not in existing]


def embed_texts(client, texts):
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    ordered = sorted(response.data, key=lambda item: item.index)
    if [item.index for item in ordered] != list(range(len(texts))):
        raise ValueError("Embedding response indices do not match the input batch.")
    vectors = [item.embedding for item in ordered]
    if any(len(vector) != EMBEDDING_DIMENSIONS for vector in vectors):
        raise ValueError("Unexpected embedding dimensionality.")
    return vectors


def index_missing(collection, documents, client, batch_size=100):
    """Save every successful batch immediately; retain existing metadata and flags."""
    pending = pending_documents(collection, documents)
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        vectors = embed_texts(client, [doc["document_text"] for doc in batch])
        collection.add(
            ids=[doc["id"] for doc in batch],
            documents=[doc["document_text"] for doc in batch],
            metadatas=[doc["metadata"] for doc in batch],
            embeddings=vectors,
        )
        print(f"Saved {min(start + batch_size, len(pending))}/{len(pending)} new records", flush=True)


def flag_for_review(collection, record_id, reason):
    """Use a reviewed record ID, never a result's changing rank position."""
    if not reason.strip():
        raise ValueError("Provide a review reason.")
    record = collection.get(ids=[record_id], include=["metadatas"])
    if not record["ids"]:
        raise ValueError(f"Unknown record: {record_id}")
    metadata = dict(record["metadatas"][0] or {})
    metadata.update(needs_review=True, review_reason=reason)
    collection.update(ids=[record_id], metadatas=[metadata])


def search(collection, query_vector, k=3, condition=None):
    """Include legacy records without a flag, but exclude every explicitly flagged ID."""
    if k < 1:
        raise ValueError("k must be positive.")
    arguments = {"include": ["metadatas"]}
    if condition is not None:
        arguments["where"] = {"condition": condition}
    records = collection.get(**arguments)
    eligible = [
        record_id for record_id, metadata in zip(records["ids"], records["metadatas"])
        if not (metadata or {}).get("needs_review", False)
    ]
    if not eligible:
        return []
    result = collection.query(
        ids=eligible, query_embeddings=[query_vector], n_results=min(k, len(eligible)),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"id": record_id, **metadata, "distance": distance, "document_text": text}
        for record_id, metadata, distance, text in zip(
            result["ids"][0], result["metadatas"][0],
            result["distances"][0], result["documents"][0],
        )
    ]
