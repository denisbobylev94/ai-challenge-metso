"""
Product expert — product brochure retrieval and question answering.

Hybrid (vector + keyword) search over the product brochure corpus, with query
routing that decides how widely to fan out across brochures:

  - Suitability question ("does X work for Y") → anchor + second search on the
    application alone, so competing products can surface alongside the named one.
  - Discovery query ("what options for...", "compare...") → keep the top brochures.
  - Everything else (a specific product question) → narrow to the single best
    brochure so the answer stays focused on one product.

The LLM never sees raw brochures — it only narrates the passages returned by
search_product_brochures().

Public API:
    search_product_brochures(query)  ->  dict   agent tool entry point
"""

import re
import config
from ingestion.core.vector_store import VectorStore
from ingestion.core.hybrid_search import HybridSearch

# Module-level cache — the HybridSearch index is built once and reused per query.
_hs: HybridSearch | None = None

# Words that signal a broad discovery query needing results across multiple brochures.
_DISCOVERY_SIGNALS = {
    "option", "options", "alternative", "alternatives",
    "compare", "comparison", "versus", " vs ",
    "recommend", "suggest",
    "suitable for", "work for", "works for", "best for", "good for", "used for",
    "what can", "what do you have", "what products", "which products",
    "what equipment", "which equipment", "what do we have",
}

# Extracts the use-case/application from suitability questions so it can be
# searched independently of the product name.
# "does the Orion pump work for mill discharge duty" → "mill discharge duty"
_SUITABILITY_RE = re.compile(
    r"(?:work(?:s)?\s+for|suitable\s+for|good\s+for|used\s+for|designed\s+for|"
    r"built\s+for|fit\s+for|right\s+for|intended\s+for)\s+(.+?)(?:\?|\.|\s*$)",
    re.IGNORECASE,
)


def _get_hybrid_search() -> HybridSearch:
    """Return the cached HybridSearch index, building it on first use."""
    global _hs
    if _hs is None:
        _hs = HybridSearch(VectorStore("brochures"))
    return _hs


def _is_discovery_query(query: str) -> bool:
    """True if the query asks to explore/compare options across products."""
    q = query.lower()
    return any(signal in q for signal in _DISCOVERY_SIGNALS)


def _extract_application(query: str) -> str | None:
    """Return the application/use-case phrase from a suitability question, or None."""
    m = _SUITABILITY_RE.search(query)
    if m:
        app = m.group(1).strip().rstrip("?. ")
        return app if len(app.split()) >= 2 else None
    return None


def _scores_by_source(results: list[dict]) -> dict[str, float]:
    """Sum each passage's relevance score by its source brochure."""
    totals: dict[str, float] = {}
    for r in results:
        source = (r.get("metadata", {}).get("source", "") or "").strip() or "unknown"
        totals[source] = totals.get(source, 0.0) + float(r.get("score", 0.0))
    return totals


def _pick_primary_source(results: list[dict]) -> str:
    """Return the single brochure with the highest total relevance score."""
    scores = _scores_by_source(results)
    return max(scores, key=scores.get)


def _multi_source_results(results: list[dict], max_sources: int = 3) -> list[dict]:
    """Keep the top max_sources brochures, up to 3 passages each."""
    scores = _scores_by_source(results)
    top_sources = set(sorted(scores, key=scores.get, reverse=True)[:max_sources])
    seen: dict[str, int] = {}
    kept = []
    for r in results:
        source = (r.get("metadata", {}).get("source", "") or "").strip()
        if source not in top_sources:
            continue
        if seen.get(source, 0) >= 3:
            continue
        seen[source] = seen.get(source, 0) + 1
        kept.append(r)
    return kept


def _build_passages(results: list[dict]) -> list[dict]:
    """Shape raw search hits into the passage dicts the agent narrates."""
    return [
        {
            "text":            r["text"],
            "source":          (r.get("metadata", {}).get("source", "") or "").strip(),
            "section":         r["metadata"].get("section", ""),
            "relevance_score": round(r.get("score", 0.0), 4),
        }
        for r in results
    ]


def _embed(text: str) -> list[float]:
    """Embed a single string in the same space as the indexed brochures."""
    response = config.client.embeddings.create(
        model=config.OPENAI_EMBEDDING_MODEL,
        input=[text],
    )
    return response.data[0].embedding


def search_product_brochures(query: str) -> dict:
    """Agent tool entry point: retrieve the brochure passages that answer a query.

    Runs a hybrid search, then routes by query shape (suitability / discovery /
    specific) to decide how many brochures to keep. Returns the kept passages,
    the query actually searched, and the list of source documents for citation.
    Returns found=False with empty passages when nothing matches.
    """
    hs = _get_hybrid_search()

    results = hs.search(query, _embed(query), n_results=10)

    if not results:
        return {
            "found":            False,
            "passages":         [],
            "query_used":       query,
            "source_documents": [],
        }

    application = _extract_application(query)

    if application:
        # Suitability question ("does X work for Y"): the main search is anchored to
        # the product name and won't surface alternatives. Run a second search on just
        # the application so competing products can appear.
        app_results = hs.search(application, _embed(application), n_results=10)
        seen_ids = {r["id"] for r in results}
        merged   = list(results) + [r for r in app_results if r["id"] not in seen_ids]
        kept     = _multi_source_results(merged, max_sources=3)

    elif _is_discovery_query(query):
        kept = _multi_source_results(results, max_sources=3)

    else:
        primary = _pick_primary_source(results)
        kept    = [
            r for r in results
            if (r.get("metadata", {}).get("source", "") or "").strip() == primary
        ]

    passages = _build_passages(kept)
    sources  = list(dict.fromkeys(p["source"] for p in passages))

    return {
        "found":            True,
        "passages":         passages,
        "query_used":       query,
        "source_documents": sources,
    }
