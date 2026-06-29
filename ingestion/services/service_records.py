"""
Ingestion pipeline for Field Service Reports (FSRs).

Flow: raw .txt files → LLM structured extraction → Pydantic validation
      → SQLite (structured query store) + ChromaDB (vector similarity store).

Run via `python -m ingestion.run_all` or directly as a script.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import time
from pathlib import Path

import config
from agent.prompts import EXTRACTION_PROMPT
from ingestion.core.vector_store import VectorStore
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

# Maximum cost value accepted during extraction. Rejects obvious LLM hallucinations
# (e.g. annual revenue figures mistaken for a service invoice).
_MAX_PLAUSIBLE_COST = 500_000


class ServiceRecord(BaseModel):
    """Structured representation of one field service report.

    All fields are optional because FSR quality varies widely — some reports
    omit cost, model, or location entirely. The LLM is instructed to return
    null rather than guess any missing value.
    """

    service_types: list[str] = []
    equipment_model: str | None = None
    country: str | None = None
    total_cost: float | None = None
    currency: str | None = None
    work_summary: str | None = None
    cost_notes: str | None = None

    @field_validator("total_cost", mode="before")
    @classmethod
    def _coerce_cost(cls, value: object) -> float | None:
        """Normalise cost strings and reject implausible values.

        Strips thousands-separators (e.g. "1,500" → 1500.0).
        Rejects zero, negative, and values above _MAX_PLAUSIBLE_COST.
        """
        if value is None:
            return None
        try:
            parsed = float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return None
        return parsed if 0 < parsed < _MAX_PLAUSIBLE_COST else None

    @property
    def has_pricing_signal(self) -> bool:
        """True if the record contains a cost figure worth storing."""
        return self.total_cost is not None

    @property
    def cost_source(self) -> str:
        """Classify the cost quality.

        'actual_recorded' — explicit invoice total with currency present.
                            Used in cost range calculations.
        'no_invoice'      — cost or currency is missing.
                            Excluded from estimates to avoid speculative figures.
        """
        if self.total_cost is not None and self.currency:
            return "actual_recorded"
        return "no_invoice"

    def to_embedding_text(self) -> str:
        """Build the text that gets embedded in the vector store.

        Format mirrors the query format used at search time so that cosine
        similarity compares equivalent representations.
        Format: "{service_types} on {equipment_model} in {country}. {work_summary}"
        """
        service_types = ", ".join(self.service_types) if self.service_types else "service"
        equipment_model = self.equipment_model or "equipment"
        country = self.country or "unknown country"
        summary = self.work_summary or ""
        return f"{service_types} on {equipment_model} in {country}. {summary}".strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ingest_service_records(services_dir: str = config.DATA_SERVICES_DIR) -> int:
    """Ingest all .txt FSRs from `services_dir` into SQLite and ChromaDB.

    Wipes existing data first so re-running produces a clean, reproducible
    state. Returns the number of records successfully inserted.
    """
    # Reset the vector store so stale embeddings don't persist across runs.
    vector_store = VectorStore("service_records")
    if vector_store.collection_is_populated():
        vector_store.delete_collection()
        vector_store = VectorStore("service_records")

    # Reset the SQLite database for the same reason.
    db_path = config.SERVICE_DB_PATH
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)
    _init_sqlite_db(db_path)

    txt_files = sorted(Path(services_dir).glob("*.txt"))
    if not txt_files:
        logger.warning("No .txt files found in %s", services_dir)
        return 0

    inserted = 0
    skipped: list[str] = []

    for txt_path in txt_files:
        try:
            raw_text = txt_path.read_text(encoding="utf-8", errors="replace")

            # Ask the LLM to extract structured fields from free-form technician text.
            record = _extract_structured_fields(raw_text, txt_path.name)

            # Skip records with no cost — they cannot contribute to estimates.
            if not record.has_pricing_signal:
                skipped.append(txt_path.name)
                continue

            record_id = txt_path.stem

            # Persist structured fields to SQLite for deterministic SQL aggregation.
            _insert_sqlite(
                db_path=db_path,
                record=record,
                file_id=record_id,
                filename=txt_path.name,
                raw_text=raw_text,
            )

            # Persist embedding to ChromaDB for semantic similarity search.
            embed_text = record.to_embedding_text()
            embedding = _embed_text(embed_text)
            vector_store.add(
                ids=[record_id],
                embeddings=[embedding],
                documents=[embed_text],
                metadatas=[
                    {
                        "service_types": ",".join(record.service_types),
                        "equipment_model": record.equipment_model or "",
                        "country": record.country or "",
                        "cost_source": record.cost_source,
                    }
                ],
            )

            inserted += 1
            logger.info("✓ %s [%s]", txt_path.name, record.cost_source)

        except Exception as exc:
            logger.error("✗ %s — %s", txt_path.name, exc)

    print(f"  {inserted} inserted, {len(skipped)} skipped (no pricing signal)")
    if skipped:
        print("  Skipped: " + ", ".join(skipped))

    return inserted


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_structured_fields(raw_text: str, filename: str) -> ServiceRecord:
    """Call the LLM to extract structured fields from raw FSR text.

    Uses exponential back-off (up to 5 attempts) to handle transient API
    errors. Returns an empty ServiceRecord if all attempts fail so the
    caller can gracefully skip the file.
    """
    prompt = f"{EXTRACTION_PROMPT}\n\nFilename: {filename}\n\nReport:\n{raw_text}"

    for attempt in range(5):
        try:
            response = config.client.chat.completions.create(
                model=config.EXTRACTION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_completion_tokens=512,
            )
            payload = json.loads(response.choices[0].message.content)
            return ServiceRecord.model_validate(payload)
        except Exception as exc:
            if attempt == 4:
                logger.error("Extraction failed for %s: %s", filename, exc)
                return ServiceRecord()
            time.sleep(2 ** attempt)

    return ServiceRecord()


def _embed_text(text: str) -> list[float]:
    """Return the embedding vector for a single text string."""
    response = config.client.embeddings.create(
        model=config.OPENAI_EMBEDDING_MODEL,
        input=[text],
    )
    return response.data[0].embedding


def _init_sqlite_db(db_path: str) -> None:
    """Create the service_records table and indexes."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS service_records (
                id                TEXT PRIMARY KEY,
                filename          TEXT,
                service_types     TEXT,   -- JSON array e.g. '["inspection","seal_replacement"]'
                equipment_model   TEXT,
                country           TEXT,
                total_cost        REAL,
                currency          TEXT,
                cost_source       TEXT,   -- 'actual_recorded' | 'no_invoice'
                cost_notes        TEXT,
                work_summary      TEXT,
                raw_text          TEXT
            )
        """)
        # Index on cost_source speeds up the frequent WHERE cost_source = 'actual_recorded' filter.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cost_source ON service_records(cost_source)"
        )


def _insert_sqlite(
    db_path: str,
    record: ServiceRecord,
    *,
    file_id: str,
    filename: str,
    raw_text: str,
) -> None:
    """Upsert one ServiceRecord into SQLite.

    Uses INSERT OR REPLACE so re-running ingestion on the same file
    updates the row rather than raising a duplicate key error.
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO service_records
                (id, filename, service_types, equipment_model, country,
                 total_cost, currency, cost_source, cost_notes, work_summary, raw_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                filename,
                json.dumps(record.service_types),
                record.equipment_model,
                record.country,
                record.total_cost,
                record.currency,
                record.cost_source,
                record.cost_notes,
                record.work_summary,
                raw_text,
            ),
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest FSR .txt files into SQLite + ChromaDB"
    )
    parser.add_argument(
        "--dir",
        default=config.DATA_SERVICES_DIR,
        help="Folder containing .txt service reports",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ingest_service_records(args.dir)
