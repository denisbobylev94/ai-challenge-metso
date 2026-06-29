"""
Service cost estimation: vector search → SQL aggregation.

Flow:
  1. Embed the query in the same format as stored records.
  2. Vector search returns the top semantically similar FSR IDs.
  3. SQL filters those IDs to actual invoiced records and aggregates costs.

The LLM never sees raw records and never computes numbers — it only narrates
the pre-computed result returned by estimate_service_cost().
"""

import json
import sqlite3
import statistics

import config
from ingestion.core.vector_store import VectorStore

# Cosine distance threshold for the vector search pre-filter.
# Records above this threshold are too dissimilar to be useful candidates.
# Kept deliberately relaxed because SQL equipment filtering is the precision gate.
_MAX_VECTOR_DISTANCE = 0.55

# Confidence thresholds — how many invoiced records back the estimate.
_MIN_RECORDS_HIGH_CONFIDENCE   = 5
_MIN_RECORDS_MEDIUM_CONFIDENCE = 3


# ---------------------------------------------------------------------------
# Public tools (called by the agent via DISPATCH table)
# ---------------------------------------------------------------------------

def estimate_service_cost(
    service_type: str,
    user_query: str,
    equipment_model: str,
) -> dict:
    """Estimate the cost of a field service job from historical invoiced records.

    Returns a cost range per currency, confidence level, and the source records
    used. Returns found=False with a known_equipment list when no data exists,
    so the agent can suggest alternatives.
    """
    vs = VectorStore("service_records")
    db_path = config.SERVICE_DB_PATH

    # Load known equipment names upfront — used for token matching and fallback suggestions.
    known_equipment = _get_known_equipment(db_path)

    # Format the query to match the stored embedding schema:
    # "{service_types} on {equipment_model}. {work_summary}"
    # This keeps query and index in the same semantic space so cosine distance
    # measures job similarity, not just surface phrasing similarity.
    query_text = f"{service_type} on {equipment_model}. {user_query}".strip()
    response = config.client.embeddings.create(
        model=config.OPENAI_EMBEDDING_MODEL,
        input=[query_text],
    )
    query_embedding = response.data[0].embedding

    # Stage 1: semantic search — find the top-10 most similar past jobs.
    candidates = vs.query(query_embedding, n_results=10)
    if not candidates:
        return {
            "found": False,
            "reason": "No similar service records found",
            "known_equipment": known_equipment,
        }

    # Drop candidates that are too semantically distant.
    # SQL equipment filtering below acts as the precision gate, so this
    # threshold is intentionally relaxed to avoid discarding valid records.
    candidates = [r for r in candidates if r["distance"] <= _MAX_VECTOR_DISTANCE]
    if not candidates:
        return {
            "found": False,
            "reason": "No sufficiently similar service records found",
            "known_equipment": known_equipment,
        }

    candidate_ids = [r["id"] for r in candidates]

    # Stage 2: SQL aggregation — filter candidates to invoiced records only
    # and apply the equipment LIKE filter for precision.
    actual_rows = _query_actual_costs(db_path, candidate_ids, equipment_model, known_equipment)

    # Group costs by currency — never mix EUR and USD into one range.
    costs_by_currency: dict[str, list[float]] = {}
    for row in actual_rows:
        currency = row.get("currency") or "UNKNOWN"
        costs_by_currency.setdefault(currency, []).append(row["total_cost"])

    actual_count = sum(len(v) for v in costs_by_currency.values())

    if actual_count == 0:
        return {
            "found": False,
            "reason": (
                "No invoiced cost data available for this service type. "
                "Please contact the service team for a manual quote."
            ),
            "known_equipment": known_equipment,
        }

    # Confidence reflects how many invoiced records back the estimate.
    if actual_count >= _MIN_RECORDS_HIGH_CONFIDENCE:
        confidence = "high"
    elif actual_count >= _MIN_RECORDS_MEDIUM_CONFIDENCE:
        confidence = "medium"
    else:
        confidence = "low"

    # Compute min / max / median per currency in Python — not by the LLM.
    ranges_by_currency = {
        currency: {
            "min_cost": round(min(vals), 2),
            "max_cost": round(max(vals), 2),
            "median_cost": round(statistics.median(vals), 2),
            "record_count": len(vals),
        }
        for currency, vals in costs_by_currency.items()
    }

    # Build per-job detail so the agent can narrate country context.
    similar_jobs = [
        {
            "id": row["id"],
            "summary": row.get("work_summary") or "",
            "cost": row["total_cost"],
            "currency": row.get("currency") or "",
            "country": row.get("country") or "",
            "source_document": row.get("filename") or f"{row['id']}.txt",
        }
        for row in actual_rows
    ]

    # Deduplicated list of source FSR filenames for citation.
    seen: set[str] = set()
    source_documents = []
    for job in similar_jobs:
        doc = job.get("source_document")
        if doc and doc not in seen:
            source_documents.append(doc)
            seen.add(doc)

    # Caveats surface limitations the agent should communicate to the rep.
    caveats = []
    if len(costs_by_currency) > 1:
        caveats.append(
            f"Records span {len(costs_by_currency)} currencies "
            f"({', '.join(costs_by_currency)}); ranges shown per currency"
        )
    if actual_count < _MIN_RECORDS_HIGH_CONFIDENCE:
        caveats.append(
            f"Based on {actual_count} invoiced record(s) — "
            "range will narrow as more data is added"
        )

    return {
        "found": True,
        "estimate": {
            "ranges_by_currency": ranges_by_currency,
            "confidence": confidence,
            "actual_records_used": actual_count,
            "equipment_model_searched": equipment_model,
        },
        "source_record_ids": candidate_ids,
        "source_documents": source_documents,
        "similar_jobs": similar_jobs,
        "caveats": caveats,
    }


def list_equipment_services(equipment_model: str) -> dict:
    """Return the service types recorded for a given equipment model.

    Used as Step 1 of the cost estimation flow to:
    - Confirm the equipment is in our records (equipment_matched shows full names).
    - Surface available service types so the agent can validate the user's request
      and pass the exact stored label to estimate_service_cost().
    """
    db_path = config.SERVICE_DB_PATH
    known_equipment = _get_known_equipment(db_path)

    # Extract the most useful search token from the user's equipment description.
    token = _equipment_token(equipment_model, known_equipment)
    if not token:
        return {"found": False, "reason": "Could not identify equipment model."}

    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT equipment_model, service_types FROM service_records "
                "WHERE equipment_model LIKE ?",
                (f"%{token}%",),
            ).fetchall()
    except Exception as exc:
        return {"found": False, "reason": str(exc)}

    # Count how often each service type appears across matched records.
    counts: dict[str, int] = {}
    matched_models: set[str] = set()
    for equipment, raw_service_types in rows:
        if equipment:
            matched_models.add(equipment)
        for service_type in json.loads(raw_service_types or "[]"):
            counts[service_type] = counts.get(service_type, 0) + 1

    if not counts:
        return {
            "found": False,
            "reason": f"No service records found for equipment matching '{token}'.",
            "known_equipment": known_equipment,
        }

    # Return services sorted by frequency so the most common appear first.
    services = sorted(counts.items(), key=lambda item: -item[1])
    return {
        "found": True,
        "equipment_matched": sorted(matched_models),   # full DB model names, not just the token
        "services": [{"service_type": s, "record_count": c} for s, c in services],
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_known_equipment(db_path: str) -> list[str]:
    """Return all distinct non-null equipment model names from the database.

    Used to suggest alternatives when a query finds no matching equipment,
    and to guide the token extraction in _equipment_token().
    """
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT DISTINCT equipment_model FROM service_records "
            "WHERE equipment_model IS NOT NULL AND equipment_model != '' "
            "ORDER BY equipment_model"
        ).fetchall()
        conn.close()
        return [row[0] for row in rows]
    except Exception:
        return []


def _equipment_token(equipment_model: str, known_equipment: list[str] | None = None) -> str:
    """Extract the most useful search token from a free-text equipment description.

    Scans each word in the description and returns the first that appears as a
    substring of any known equipment name in the database. Falls back to the
    first word if no match is found.

    Example: "Metso Concorde Cell Flotation Unit"
      → "Metso" not in any known model
      → "Concorde" found in "Concorde Cell"
      → returns "Concorde"  →  SQL uses LIKE '%Concorde%'

    This prevents generic brand names or adjectives at the start of a
    description from producing empty SQL matches.
    """
    if not equipment_model:
        return ""

    words = equipment_model.split()

    if known_equipment:
        known_lower = [name.lower() for name in known_equipment]
        for word in words:
            if any(word.lower() in known_name for known_name in known_lower):
                return word

    return words[0]


def _query_actual_costs(
    db_path: str,
    ids: list[str],
    equipment_model: str,
    known_equipment: list[str] | None = None,
) -> list[dict]:
    """Fetch invoiced cost records for the given candidate IDs, filtered by equipment model.

    Equipment identity is confirmed upstream by list_equipment_services, so if
    no records match here the correct response is found=False rather than
    falling back to records from a different machine.
    """
    if not ids:
        return []

    placeholders = ",".join("?" * len(ids))
    token = _equipment_token(equipment_model, known_equipment)
    equip_pattern = f"%{token}%"

    # Use explicit close rather than context manager — on Windows, sqlite3's
    # context manager only commits/rolls back but does not close the connection,
    # leaving the file locked until GC runs.
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(f"""
            SELECT id, filename, total_cost, currency, country, work_summary, service_types
            FROM service_records
            WHERE id IN ({placeholders})
              AND cost_source = 'actual_recorded'
              AND total_cost IS NOT NULL
              AND equipment_model LIKE ?
        """, [*ids, equip_pattern]).fetchall()

        return [dict(row) for row in rows]
    finally:
        conn.close()
