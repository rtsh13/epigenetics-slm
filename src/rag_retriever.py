"""
Week 4: Bio-RAG Retrieval Module

Wraps a ChromaDB collection with a clean .search() API for querying the
curated epigenetics knowledge base. Accepts an injected client for testing
(EphemeralClient) or uses a persistent local client in production.

Usage (from project root, venv activated):
    from src.rag_retriever import BioRAG

    # First time: build the DB from knowledge files
    rag = BioRAG.from_knowledge_dir("knowledge")

    # Query
    results = rag.search("low RMSSD 55-year-old male HRV aging")
    for r in results:
        print(r["category"], r["title"])
        print(r["context"][:200])
        print(r["source"])
"""

import logging
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from vector_db_builder import BioRAGBuilder

log = logging.getLogger(__name__)

VALID_CATEGORIES = frozenset(
    ["Aging", "Stress", "Metabolism", "Inflammation", "Sleep", "Environmental"]
)
DEFAULT_CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "bio_rag"


class BioRAG:
    """Query interface for the Bio-RAG ChromaDB knowledge base.

    Args:
        chroma_dir: Path to ChromaDB persistent storage (used in production).
        client: Injected ChromaDB client (use EphemeralClient for tests).
        collection_name: Name of the ChromaDB collection to query.
    """

    def __init__(
        self,
        chroma_dir: str = DEFAULT_CHROMA_DIR,
        client: Optional[chromadb.api.ClientAPI] = None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            self._client = chromadb.PersistentClient(path=chroma_dir)
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    @classmethod
    def from_knowledge_dir(
        cls,
        knowledge_dir: str,
        chroma_dir: str = DEFAULT_CHROMA_DIR,
        client: Optional[chromadb.api.ClientAPI] = None,
        collection_name: str = COLLECTION_NAME,
    ) -> "BioRAG":
        """Build DB from knowledge_dir and return a ready-to-query BioRAG.

        Raises:
            FileNotFoundError: If knowledge_dir does not exist.
        """
        rag = cls(chroma_dir=chroma_dir, client=client, collection_name=collection_name)
        builder = BioRAGBuilder.from_knowledge_dir(
            knowledge_dir=knowledge_dir,
            chroma_client=rag._client,
            collection_name=collection_name,
        )
        builder.build()
        return rag

    def search(
        self,
        query: str,
        n_results: int = 3,
        category: Optional[str] = None,
    ) -> list[dict]:
        """Retrieve the most relevant knowledge chunks for a query.

        Args:
            query: Natural language query string.
            n_results: Maximum number of results to return (default 3).
            category: Optional filter to restrict results to one category.
                Must be one of: Aging, Stress, Metabolism, Inflammation, Sleep, Environmental.

        Returns:
            List of dicts, each with keys:
                context  — The retrieved text chunk
                source   — Citation string
                type     — entry type
                category — biomarker category
                title    — Human-readable entry title

        Raises:
            ValueError: If query is empty, category is not a valid value, or n_results < 1.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        if category is not None and category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{category}'. Must be one of: {sorted(VALID_CATEGORIES)}"
            )

        if n_results < 1:
            raise ValueError("n_results must be >= 1")

        query_embedding = self._model.encode([query], show_progress_bar=False).tolist()

        where_filter = {"category": {"$eq": category}} if category is not None else None

        total_docs = self._collection.count()
        if total_docs == 0:
            log.warning("search() called on empty collection '%s'", self._collection_name)
            return []
        effective_n = min(n_results, total_docs)

        query_kwargs: dict = {
            "query_embeddings": query_embedding,
            "n_results": effective_n,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter is not None:
            query_kwargs["where"] = where_filter

        raw = self._collection.query(**query_kwargs)

        results: list[dict] = []
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]

        for doc, meta in zip(docs, metas):
            results.append(
                {
                    "context": doc,
                    "source": meta.get("source", ""),
                    "type": meta.get("type", ""),
                    "category": meta.get("category", ""),
                    "title": meta.get("title", ""),
                }
            )

        return results
