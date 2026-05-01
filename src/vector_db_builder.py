"""
Week 4: Bio-RAG Vector Database Builder

Loads curated knowledge JSON files, splits long entries into ~400-word chunks,
embeds each chunk with all-MiniLM-L6-v2, and stores them in a ChromaDB
collection. Accepts an injected ChromaDB client so tests can use EphemeralClient.

Usage (from project root, venv activated):
    from src.vector_db_builder import BioRAGBuilder
    import chromadb

    client = chromadb.PersistentClient(path="data/chroma_db")
    builder = BioRAGBuilder.from_knowledge_dir("knowledge", chroma_client=client)
    builder.build()
"""

import json
import logging
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

VALID_CATEGORIES = frozenset(
    ["Aging", "Stress", "Metabolism", "Inflammation", "Sleep", "Environmental"]
)
REQUIRED_FIELDS = ("id", "category", "type", "title", "content", "source")
# all-MiniLM-L6-v2 max_seq_length = 256 tokens; ~150 words leaves headroom for title prefix
MAX_WORDS_PER_CHUNK = 150
COLLECTION_NAME = "bio_rag"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class BioRAGBuilder:
    """Populates a ChromaDB collection from curated knowledge entries.

    Args:
        knowledge_entries: List of dicts, each with keys:
            id, category, type, title, content, source.
        chroma_client: Any ChromaDB client (PersistentClient or EphemeralClient).
        collection_name: Name of the ChromaDB collection to create/update.
        embedding_model: Sentence-transformers model name.
    """

    def __init__(
        self,
        knowledge_entries: list[dict],
        chroma_client: chromadb.api.ClientAPI,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        self._entries = knowledge_entries
        self._client = chroma_client
        self._collection_name = collection_name
        self._model = SentenceTransformer(embedding_model)

    @classmethod
    def from_knowledge_dir(
        cls,
        knowledge_dir: str,
        chroma_client: chromadb.api.ClientAPI,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> "BioRAGBuilder":
        """Load all .json files from knowledge_dir and return a builder.

        Raises:
            FileNotFoundError: If knowledge_dir does not exist.
        """
        p = Path(knowledge_dir)
        if not p.exists():
            raise FileNotFoundError(f"Knowledge directory not found: {p}")

        entries: list[dict] = []
        for json_file in sorted(p.glob("*.json")):
            with open(json_file) as f:
                data = json.load(f)
            if isinstance(data, list):
                entries.extend(data)
            else:
                entries.append(data)
        log.info("Loaded %d knowledge entries from %s", len(entries), knowledge_dir)
        return cls(entries, chroma_client, collection_name, embedding_model)

    def build(self) -> None:
        """Validate, chunk, embed, and upsert all entries into ChromaDB.

        Idempotent: uses upsert so calling build() twice does not duplicate documents.

        Raises:
            ValueError: If any entry is missing required fields or has invalid category.
        """
        chunks = self._prepare_chunks()

        if not chunks:
            log.warning("No chunks to embed — knowledge_entries produced no content.")
            return

        collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [c["id"] for c in chunks]

        embeddings = self._model.encode(documents, show_progress_bar=False).tolist()

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        log.info(
            "Upserted %d chunks into collection '%s'", len(chunks), self._collection_name
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_chunks(self) -> list[dict]:
        """Validate entries and split long ones into chunks.

        Returns list of dicts with keys: id, text, metadata.
        """
        all_chunks: list[dict] = []
        for entry in self._entries:
            self._validate_entry(entry)
            entry_chunks = self._chunk_entry(entry)
            all_chunks.extend(entry_chunks)
        return all_chunks

    def _validate_entry(self, entry: dict) -> None:
        for field in REQUIRED_FIELDS:
            if field not in entry:
                raise ValueError(
                    f"Knowledge entry missing required field '{field}': {entry.get('id', '(no id)')}"
                )
        if entry["category"] not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{entry['category']}' for entry '{entry['id']}'. "
                f"Must be one of: {sorted(VALID_CATEGORIES)}"
            )

    def _chunk_entry(self, entry: dict) -> list[dict]:
        """Split content into <=MAX_WORDS_PER_CHUNK word chunks.

        Prepends title to each chunk so that semantic search works even on
        mid-entry chunks. Each chunk gets a unique id: {entry_id}_{chunk_index}.
        """
        words = entry["content"].split()
        chunk_texts: list[str] = []

        if len(words) <= MAX_WORDS_PER_CHUNK:
            chunk_texts.append(entry["content"])
        else:
            for start in range(0, len(words), MAX_WORDS_PER_CHUNK):
                chunk_words = words[start : start + MAX_WORDS_PER_CHUNK]
                chunk_texts.append(" ".join(chunk_words))

        chunks: list[dict] = []
        for i, chunk_text in enumerate(chunk_texts):
            full_text = f"{entry['title']}\n\n{chunk_text}"
            chunks.append(
                {
                    "id": f"{entry['id']}_{i}",
                    "text": full_text,
                    "metadata": {
                        "category": entry["category"],
                        "type": entry["type"],
                        "title": entry["title"],
                        "source": entry["source"],
                        "entry_id": entry["id"],
                        "chunk_index": i,
                    },
                }
            )
        return chunks
