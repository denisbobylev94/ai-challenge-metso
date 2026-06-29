import re
from rank_bm25 import BM25Okapi
from ingestion.core.vector_store import VectorStore


class HybridSearch:
    def __init__(self, vector_store: VectorStore):
        self._vs = vector_store
        self._corpus: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._id_to_doc: dict[str, dict] = {}
        self._fit()

    def _fit(self) -> None:
        self._corpus = self._vs.get_all()
        self._id_to_doc = {d["id"]: d for d in self._corpus}
        tokenised = [self._tokenise(d["text"]) for d in self._corpus]
        if tokenised:
            self._bm25 = BM25Okapi(tokenised)
        else:
            self._bm25 = None

    def refresh(self) -> None:
        self._fit()

    def search(
        self,
        query: str,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        if not self._corpus or self._bm25 is None:
            return []

        # BM25 branch
        tokens = self._tokenise(query)
        bm25_scores = self._bm25.get_scores(tokens)
        bm25_ranked = sorted(
            range(len(self._corpus)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:20]
        bm25_ids = [self._corpus[i]["id"] for i in bm25_ranked]

        # Dense branch
        dense_results = self._vs.query(query_embedding, n_results=20, where=where)
        dense_ids = [r["id"] for r in dense_results]

        # RRF fusion
        fused = self._rrf([bm25_ids, dense_ids])

        # Return top n_results
        ranked_ids = sorted(fused, key=lambda k: fused[k], reverse=True)[:n_results]
        results = []
        for rid in ranked_ids:
            doc = self._id_to_doc.get(rid, {})
            results.append({
                "id": rid,
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata", {}),
                "score": fused[rid],
            })
        return results

    def _tokenise(self, text: str) -> list[str]:
        return re.split(r"[^a-zA-Z0-9]+", text.lower())

    def _rrf(self, rankings: list[list[str]], k: int = 60) -> dict[str, float]:
        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return scores
