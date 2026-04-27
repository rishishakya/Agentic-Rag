"""
Retriever Agent — searches ChromaDB vector store for relevant documents.
Ingests local PDFs/text files on first run, then queries via embeddings.
"""

import os
import hashlib
from pathlib import Path
from typing import List

import chromadb
from chromadb.utils import embedding_functions


# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_PATH   = "./chroma_db"
COLLECTION    = "research_docs"
DOCS_FOLDER   = "./sample_docs"
CHUNK_SIZE    = 800    # characters per chunk
CHUNK_OVERLAP = 100


def _get_collection():
    """Get or create the ChromaDB collection with sentence-transformer embeddings."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Use local sentence-transformers (free, no API key)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def _chunk_text(text: str, source: str) -> List[dict]:
    """Split text into overlapping chunks."""
    chunks = []
    start  = 0
    idx    = 0

    while start < len(text):
        end   = start + CHUNK_SIZE
        chunk = text[start:end]
        chunk_id = hashlib.md5(f"{source}_{idx}".encode()).hexdigest()

        chunks.append({
            "id":       chunk_id,
            "text":     chunk,
            "source":   source,
            "chunk_idx": idx
        })

        start += CHUNK_SIZE - CHUNK_OVERLAP
        idx   += 1

    return chunks


def ingest_documents(folder: str = DOCS_FOLDER) -> int:
    """
    Ingest all .txt and .pdf files from a folder into ChromaDB.
    Returns number of chunks ingested.
    """
    collection = _get_collection()
    folder_path = Path(folder)

    if not folder_path.exists():
        folder_path.mkdir(parents=True)
        return 0

    total_chunks = 0

    for file_path in folder_path.iterdir():
        text = ""

        if file_path.suffix == ".txt":
            text = file_path.read_text(encoding="utf-8", errors="ignore")

        elif file_path.suffix == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(str(file_path))
                text   = "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                print(f"⚠️  pypdf not installed — skipping {file_path.name}")
                continue
            except Exception as e:
                print(f"⚠️  Could not read PDF {file_path.name}: {e}")
                continue

        if not text.strip():
            continue

        chunks = _chunk_text(text, source=file_path.name)

        # Only add chunks that aren't already stored
        existing_ids = set(collection.get(ids=[c["id"] for c in chunks])["ids"])
        new_chunks   = [c for c in chunks if c["id"] not in existing_ids]

        if new_chunks:
            collection.add(
                ids       = [c["id"]   for c in new_chunks],
                documents = [c["text"] for c in new_chunks],
                metadatas = [{"source": c["source"], "chunk": c["chunk_idx"]} for c in new_chunks]
            )
            total_chunks += len(new_chunks)

    return total_chunks


def retrieve_from_vectordb(query: str, k: int = 5) -> List[dict]:
    """
    Retrieve top-k relevant chunks from ChromaDB for a query.

    Returns list of dicts with 'text', 'source', 'score'.
    """
    # Auto-ingest docs if collection is empty
    collection = _get_collection()
    if collection.count() == 0:
        ingested = ingest_documents()
        if ingested == 0:
            return []  # no local docs yet

    results = collection.query(
        query_texts=[query],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    docs = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        docs.append({
            "text":      text,
            "source":    meta.get("source", "local"),
            "score":     round(1 - dist, 3),   # cosine similarity
            "type":      "local"
        })

    return docs
