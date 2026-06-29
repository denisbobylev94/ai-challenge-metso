"""
Brochure ingestion — PDF → chunks → ChromaDB.

Flow:
  1. Each PDF is extracted via pdf_extract.process_pdf() which returns per-page
     text (with header/footer stripped) and detected tables as Markdown.
  2. Each table becomes one chunk — splitting a table at a token boundary
     would create rows without column headers, making them meaningless.
  3. Page text is split into overlapping 512-token windows so context is
     preserved across chunk boundaries.
  4. All chunks for one PDF are embedded in a single batched API call,
     then written to the 'brochures' ChromaDB collection.

Run via: python -m ingestion.run_all
"""

import logging
import time
from pathlib import Path

import tiktoken

import config
from ingestion.products.pdf_extract import process_pdf
from ingestion.core.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Tokeniser must match the embedding model's vocabulary so chunk sizes are accurate.
_TOKENIZER = tiktoken.get_encoding("cl100k_base")

# Text chunking parameters — tables bypass these limits (one table = one chunk).
_CHUNK_MAX_TOKENS = 512  # maximum tokens per sliding text window
_CHUNK_OVERLAP    = 64   # tokens repeated at the start of each new window


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ingest_brochures(brochures_dir: str = config.DATA_BROCHURES_DIR) -> int:
    """Wipe and re-ingest all PDFs from brochures_dir into the 'brochures' collection.

    Deletes the existing collection first so stale chunks from renamed or
    deleted PDFs do not persist across runs. Returns total chunks indexed.
    """
    # Reset the collection so every ingestion run produces a clean, reproducible state.
    vs = VectorStore("brochures")
    if vs.collection_is_populated():
        vs.delete_collection()
        vs = VectorStore("brochures")

    pdf_dir = Path(brochures_dir)
    pdfs = list(pdf_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning("No PDF files found in %s", brochures_dir)
        return 0

    total_chunks = 0
    for pdf_path in pdfs:
        try:
            chunks = _extract_chunks(pdf_path)
            if not chunks:
                logger.warning("%s → no chunks produced", pdf_path.name)
                continue

            # Embed all chunks from this PDF in one batched call to minimise API round-trips.
            texts      = [c["text"] for c in chunks]
            embeddings = _embed_texts(texts)
            stem       = pdf_path.stem
            ids        = [f"{stem}_{c['metadata']['chunk_index']:04d}" for c in chunks]
            metadatas  = [c["metadata"] for c in chunks]

            vs.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

            # Print a per-file summary so extraction quality is visible at ingestion time.
            table_chunks = sum(
                1 for c in chunks
                if c["metadata"].get("has_table") and not c["metadata"].get("from_regex_fallback")
            )
            regex_chunks = sum(1 for c in chunks if c["metadata"].get("from_regex_fallback"))
            msg = f"  {pdf_path.name} → {len(chunks)} chunks ({table_chunks} table chunks"
            if regex_chunks:
                msg += f", {regex_chunks} from regex fallback"
            msg += ")"
            print(msg)
            total_chunks += len(chunks)

        except Exception as exc:
            logger.error("Failed to process %s: %s", pdf_path.name, exc)
            print(f"  WARNING: failed to process {pdf_path.name}: {exc}")

    return total_chunks


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_chunks(pdf_path: Path) -> list[dict]:
    """Extract all chunks from one PDF: one chunk per table, plus sliding text windows.

    Raises RuntimeError if pdf_extract reports a fatal extraction error.
    Logs non-fatal quality warnings (garbled fonts, sparse pages, etc.) so they
    are visible during ingestion without stopping the run.
    """
    result = process_pdf(pdf_path)
    if result["stats"].get("error"):
        raise RuntimeError(result["stats"]["error"])

    # Surface extraction quality issues (garbled text, sparse pages, etc.)
    # so they are visible at ingestion time rather than silently degrading retrieval.
    issues = result["stats"].get("issues", [])
    if issues:
        logger.warning("%s — extraction issues: %s", pdf_path.name, "; ".join(issues))

    chunks: list[dict] = []
    chunk_index = 0
    source_name = pdf_path.name

    for page in result["pages"]:
        section = f"Page {page['page_number']}"
        # Use header/footer-stripped text when available; fall back to raw page text.
        text = page.get("text_cleaned", page["text"])

        # Tables are indexed before text so their chunk_index always comes first per page.
        table_chunks, chunk_index = _table_chunks_from_page(page, source_name, section, chunk_index)
        chunks.extend(table_chunks)

        text_chunks, chunk_index = _text_chunks_from_page(text, source_name, section, chunk_index)
        chunks.extend(text_chunks)

    return chunks


def _table_chunks_from_page(
    page: dict, source: str, section: str, start_index: int
) -> tuple[list[dict], int]:
    """Return one chunk per table on the page (pdfplumber or regex fallback).

    Tables are never split across chunks — a data row only makes sense when
    the column-header row is present. Splitting at a token boundary would
    produce chunks of orphaned numbers with no context about what they represent.
    """
    chunks: list[dict] = []
    idx = start_index
    for tbl in page["tables"]:
        md = tbl["markdown"]
        if md:
            chunks.append({
                "text": md,
                "metadata": {
                    "source":              source,
                    "section":             section,
                    "chunk_index":         idx,
                    "has_table":           True,
                    "from_regex_fallback": tbl.get("from_regex_fallback", False),
                    "token_count":         len(_TOKENIZER.encode(md)),
                },
            })
            idx += 1
    return chunks, idx


def _text_chunks_from_page(
    text: str, source: str, section: str, start_index: int
) -> tuple[list[dict], int]:
    """Sliding-window token chunks from cleaned page text.

    The _CHUNK_OVERLAP token repetition at the start of each new window
    preserves sentence context at boundaries — a question about a topic
    mentioned at the end of one chunk will still find it in the next.

    If a partial window contains a Markdown table (detected by '|'), the window
    is extended to end-of-page rather than cut mid-table. This handles the rare
    case where pdfplumber embeds a table directly in the text stream instead of
    the separate tables list.
    """
    if not text.strip():
        return [], start_index

    chunks: list[dict] = []
    idx    = start_index
    tokens = _TOKENIZER.encode(text)
    start  = 0

    while start < len(tokens):
        end          = min(start + _CHUNK_MAX_TOKENS, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text   = _TOKENIZER.decode(chunk_tokens)

        # Extend to page end rather than cut a Markdown table mid-row.
        if "|" in chunk_text and end < len(tokens):
            end          = len(tokens)
            chunk_tokens = tokens[start:end]
            chunk_text   = _TOKENIZER.decode(chunk_tokens)

        chunks.append({
            "text": chunk_text,
            "metadata": {
                "source":              source,
                "section":             section,
                "chunk_index":         idx,
                "has_table":           "|" in chunk_text,
                "from_regex_fallback": False,
                "token_count":         len(chunk_tokens),
            },
        })
        idx += 1
        if end == len(tokens):
            break
        start = end - _CHUNK_OVERLAP

    return chunks, idx


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Return embeddings for a list of texts, batched to stay within API limits.

    Sends at most 100 texts per request — well below the OpenAI API limit and
    avoids payload size errors on long chunks. Each batch retries up to 5 times
    with exponential back-off before propagating the exception.
    """
    embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        for attempt in range(5):
            try:
                response = config.client.embeddings.create(
                    model=config.OPENAI_EMBEDDING_MODEL,
                    input=batch,
                )
                embeddings.extend([d.embedding for d in response.data])
                break
            except Exception as exc:
                if attempt == 4:
                    raise
                wait = 2 ** attempt
                logger.warning("Embedding retry %d after %ds: %s", attempt + 1, wait, exc)
                time.sleep(wait)
    return embeddings
