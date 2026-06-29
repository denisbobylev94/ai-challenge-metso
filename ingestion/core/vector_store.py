"""
Thin wrapper around ChromaDB for persistent vector storage.

All embedding is done externally; this class only stores and retrieves
pre-computed vectors so the embedding model is not coupled to storage.
"""

import chromadb
import config


class VectorStore:
    """Persistent ChromaDB collection with cosine-similarity search.

    One instance maps to one named collection. The collection is created on
    first access and reused on subsequent runs.
    """

    def __init__(self, collection_name: str, db_path: str = config.CHROMA_DB_PATH) -> None:
        """Open or create a named collection in the ChromaDB store at db_path."""
        self._client = chromadb.PersistentClient(path=db_path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Insert a batch of pre-embedded documents into the collection."""
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        """Return the top-n most similar documents to query_embedding.

        Args:
            query_embedding: Pre-computed embedding vector for the query.
            n_results: Maximum number of results to return.
            where: Optional ChromaDB metadata filter (e.g. {"region": "Europe"}).

        Returns:
            List of dicts with keys: id, text, metadata, distance.
        """
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, self._collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        result = self._collection.query(**kwargs)
        return [
            {
                "id": doc_id,
                "text": result["documents"][0][i],
                "metadata": result["metadatas"][0][i],
                "distance": result["distances"][0][i],
            }
            for i, doc_id in enumerate(result["ids"][0])
        ]

    def get_all(self) -> list[dict]:
        """Return every document in the collection (used to build the BM25 index)."""
        result = self._collection.get(include=["documents", "metadatas"])
        return [
            {"id": id_, "text": doc, "metadata": meta}
            for id_, doc, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        ]

    def count(self) -> int:
        """Return the number of documents currently stored in the collection."""
        return self._collection.count()

    def collection_is_populated(self) -> bool:
        """Return True if the collection contains at least one document."""
        return self.count() > 0

    def delete_collection(self) -> None:
        """Permanently delete the collection and all its data from ChromaDB."""
        self._client.delete_collection(self._collection.name)
