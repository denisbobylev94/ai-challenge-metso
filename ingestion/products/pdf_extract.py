"""
PDF extraction — text, tables, and image-presence detection from a single brochure.

Uses pdfplumber for text extraction with table bounding-box detection and cell
parsing. Image presence per page is detected via pdfplumber's page.images list
(used only to identify image-heavy pages with no text — image content is not indexed).

Returns a structured dict consumed by ingestion/products/brochures.py:
  {
    "file": str,
    "path": str,
    "pages": [
      {
        "page_number": int,
        "text": str,           # raw text after CID stripping
        "text_cleaned": str,   # after cross-page header/footer removal
        "tables": [
          {"table_id": str, "rows": list, "row_count": int,
           "col_count": int, "markdown": str, "from_regex_fallback": bool}
        ],
        "analysis": {
          "char_count": int, "word_count": int,
          "table_count": int, "table_rows": int,
          "is_sparse": bool, "has_table": bool, "has_image": bool
        }
      }
    ],
    "stats": {
      "page_count": int, "total_chars": int, "total_tables": int,
      "sparse_pages": int, "pages_with_images_only": int,
      "grade": str,        # "GOOD" | "ACCEPTABLE" | "POOR"
      "issues": [str],     # human-readable warnings surfaced by brochures.py
      "error": str | None  # set if the file could not be opened
    }
  }

Usage:
    from ingestion.products.pdf_extract import process_pdf
    result = process_pdf("./data/Products/brochure.pdf")
"""

import re
from collections import Counter
from pathlib import Path

import pdfplumber


# ---------------------------------------------------------------------------
# Text quality heuristics
# ---------------------------------------------------------------------------

# Detects single-letter words followed by short fragments — symptom of garbled
# custom-font encoding where glyph→unicode mapping is broken.
_GARBLED_RE = re.compile(r"\b[A-Za-z]\s+[a-z]{1,3}\s+[a-z]{1,4}\b")

# CID placeholders appear when pdfplumber cannot decode a glyph from a custom
# font (common in chart labels and some spec tables). Stripped because they add
# noise to embeddings without contributing meaningful text.
_CID_RE = re.compile(r"\(cid:\d+\)")

# A page with fewer than this many characters after extraction is likely
# dominated by images or diagrams rather than readable text.
_SPARSE_CHAR_THRESHOLD = 80

# Lines shorter than this are considered "short" — a high ratio suggests a
# multi-column layout that pdfplumber may read in the wrong order.
_SHORT_LINE_LEN        = 40
_SHORT_LINE_RATIO_WARN = 0.6  # warn if >60 % of lines are short


# ---------------------------------------------------------------------------
# Spec table regex fallback
# ---------------------------------------------------------------------------

# Some brochures use InDesign decorative vector graphics for table borders
# instead of real PDF line primitives. pdfplumber's table detector works by
# finding line primitives — it finds nothing in those PDFs.
#
# This regex matches model-spec lines in the extracted text stream so they can
# be reconstructed into a proper table. Example match:
#   "FFP1200  1200x1200  12.5  450  24"
#
# LIMITATION: hardcoded to four equipment families (FFP, VPA, MDM, MDR).
# New product lines will not be detected and their spec tables will be indexed
# as unstructured text instead.
_INLINE_SPEC_RE = re.compile(
    r"^(FFP\d{4}|VPA\s*\d{4}|MDM\d*|MDR\d*)\s+(.+)$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Private text helpers
# ---------------------------------------------------------------------------

def _strip_cid(text: str) -> str:
    """Remove CID placeholders and collapse resulting empty lines."""
    cleaned = _CID_RE.sub("", text)
    return "\n".join(ln for ln in cleaned.splitlines() if ln.strip())


def _table_to_markdown(table: list[list]) -> str:
    """Convert a pdfplumber table (list of rows) to a Markdown table string.

    CID noise in cell text is stripped before formatting. The first row is
    treated as the header; a separator row is inserted after it.
    """
    if not table:
        return ""
    rows = []
    for row in table:
        cells = [_strip_cid(str(c)).strip() if c is not None else "" for c in row]
        rows.append("| " + " | ".join(cells) + " |")
    col_count = len(table[0]) if table else 0
    sep = "| " + " | ".join(["---"] * col_count) + " |"
    return rows[0] + "\n" + sep + "\n" + "\n".join(rows[1:])


def _table_has_content(table: list[list]) -> bool:
    """Return True if at least one cell has non-empty content after CID stripping."""
    return any(
        cell and _strip_cid(str(cell)).strip()
        for row in table
        for cell in row
    )


def _strip_repeated_lines(pages_text: list[str]) -> list[str]:
    """Remove lines that appear on 3 or more pages and bare page-number lines.

    Headers and footers repeat verbatim across pages; content almost never does.
    Threshold of 3 avoids stripping lines that coincidentally repeat once or twice.
    """
    freq = Counter(
        ln.strip()
        for text in pages_text
        for ln in text.splitlines()
        if ln.strip()
    )
    repeated    = {line for line, cnt in freq.items() if cnt >= 3}
    page_num_re = re.compile(r"^-?\s*\d+\s*-?$")

    return [
        "\n".join(
            ln for ln in text.splitlines()
            if ln.strip() not in repeated and not page_num_re.match(ln.strip())
        )
        for text in pages_text
    ]


# ---------------------------------------------------------------------------
# Private spec-table regex fallback helpers
# ---------------------------------------------------------------------------

def _extract_inline_specs(text: str) -> list[list[str]] | None:
    """Parse model-spec lines from page text when pdfplumber found no table.

    Returns a list of rows compatible with _table_to_markdown, or None if
    fewer than 2 matching lines were found (not enough to form a table).
    """
    rows = []
    for line in text.splitlines():
        m = _INLINE_SPEC_RE.match(line.strip())
        if m:
            model = m.group(1).strip()
            cols  = re.split(r"\s{2,}", m.group(2).strip())
            rows.append([model] + cols)
    return rows if len(rows) >= 2 else None


def _inline_specs_to_table(rows: list[list[str]]) -> list[list[str]]:
    """Normalise ragged spec rows to a rectangular table by right-padding with ''.

    Adds a synthetic header row because the text stream does not carry column
    headers for these spec lines.
    """
    max_cols = max(len(r) for r in rows)
    header   = ["Model"] + [f"Col{i}" for i in range(1, max_cols)]
    padded   = [r + [""] * (max_cols - len(r)) for r in rows]
    return [header] + padded


# ---------------------------------------------------------------------------
# Private page-level extraction
# ---------------------------------------------------------------------------

def _empty_stats(error: str | None = None) -> dict:
    """Return a zeroed stats dict recording a fatal extraction error."""
    return {
        "page_count": 0, "total_chars": 0, "total_tables": 0,
        "sparse_pages": 0, "pages_with_images_only": 0,
        "grade": "POOR", "issues": [], "error": error,
    }


def _process_page(
    page_num: int,
    plumber_page: "pdfplumber.page.Page",
) -> tuple[dict, str]:
    """Extract text and tables from one page (page_num is 1-based).

    Returns:
        page_data: Structured dict (text, tables, analysis).
        raw_text:  Unstripped text, collected by the caller for cross-page
                   header/footer deduplication after all pages are processed.
    """
    page_data: dict = {
        "page_number": page_num,
        "text": "", "tables": [], "analysis": {},
    }

    # ── Text extraction ───────────────────────────────────────────────────────
    # Detect table bounding boxes first and exclude those regions from the text
    # extraction to prevent table cell text from appearing twice in chunks.
    found_tables = plumber_page.find_tables()
    table_bboxes = [t.bbox for t in found_tables]

    if table_bboxes:
        # Closure captures table_bboxes — pdfplumber.filter() calls this per character object.
        def _outside_tables(obj):
            for bbox in table_bboxes:
                if (obj.get("x0", 0) >= bbox[0] - 1
                        and obj.get("top", 0) >= bbox[1] - 1
                        and obj.get("x1", 0) <= bbox[2] + 1
                        and obj.get("bottom", 0) <= bbox[3] + 1):
                    return False
            return True
        text = plumber_page.filter(_outside_tables).extract_text() or ""
    else:
        text = plumber_page.extract_text() or ""

    text = _strip_cid(text)

    # ── Table extraction (pdfplumber) ─────────────────────────────────────────
    page_table_rows = 0
    for t_idx, table in enumerate(plumber_page.extract_tables() or []):
        if table and _table_has_content(table):
            page_table_rows += len(table)
            page_data["tables"].append({
                "table_id":            f"page_{page_num}_table_{t_idx + 1}",
                "rows":                table,
                "row_count":           len(table),
                "col_count":           len(table[0]) if table else 0,
                "markdown":            _table_to_markdown(table),
                "from_regex_fallback": False,
            })

    # ── Regex fallback for InDesign-bordered spec tables ──────────────────────
    # Only runs when pdfplumber found no tables — happens when table borders are
    # decorative vector graphics rather than real PDF line primitives.
    if not page_data["tables"]:
        inline_rows = _extract_inline_specs(text)
        if inline_rows:
            table_rows = _inline_specs_to_table(inline_rows)
            md = _table_to_markdown(table_rows)
            if md:
                page_data["tables"].append({
                    "table_id":            f"page_{page_num}_table_regex",
                    "rows":                table_rows,
                    "row_count":           len(table_rows),
                    "col_count":           len(table_rows[0]) if table_rows else 0,
                    "markdown":            md,
                    "from_regex_fallback": True,
                })
                page_table_rows += len(table_rows)

    # ── Per-page analysis ─────────────────────────────────────────────────────
    # has_image uses pdfplumber's image list — only boolean presence matters here;
    # detailed image metadata lives in EDA/pdf_quality_analysis.ipynb if needed.
    char_count = len(text.strip())
    page_data["text"] = text.strip()
    page_data["analysis"] = {
        "char_count":  char_count,
        "word_count":  len(text.split()),
        "table_count": len(page_data["tables"]),
        "table_rows":  page_table_rows,
        "is_sparse":   char_count < _SPARSE_CHAR_THRESHOLD,
        "has_table":   bool(page_data["tables"]),
        "has_image":   bool(plumber_page.images),
    }
    return page_data, text


# ---------------------------------------------------------------------------
# Private document-level quality assessment
# ---------------------------------------------------------------------------

def _compute_stats(pages: list[dict], raw_texts: list[str]) -> dict:
    """Aggregate document-level quality metrics and produce an issues list.

    stats["issues"] is surfaced by brochures.py via logger.warning() so
    extraction problems are visible at ingestion time without stopping the run.
    """
    full_text = "\n".join(raw_texts)
    non_empty = [ln for ln in full_text.splitlines() if ln.strip()]
    short_lines = [ln for ln in non_empty if len(ln.strip()) < _SHORT_LINE_LEN]
    short_ratio = len(short_lines) / max(len(non_empty), 1)
    garbled     = _GARBLED_RE.findall(full_text)
    cid_pages   = sum(1 for pg in pages if _CID_RE.search(pg["text"]))

    sparse_pages           = sum(1 for pg in pages if pg["analysis"]["is_sparse"])
    pages_with_images_only = sum(
        1 for pg in pages
        if pg["analysis"]["is_sparse"]
        and pg["analysis"]["has_image"]
        and not pg["analysis"]["has_table"]
    )

    issues = []
    if len(full_text.strip()) < 300:
        issues.append("very short extraction — may be image-heavy PDF")
    if garbled:
        issues.append(f"possible garbled font text ({len(garbled)} instances)")
    if cid_pages:
        issues.append(f"{cid_pages} page(s) had CID placeholders (chart labels stripped)")
    if short_ratio > _SHORT_LINE_RATIO_WARN:
        issues.append(f"{short_ratio:.0%} short lines — possible multi-column layout")
    if sparse_pages:
        issues.append(f"{sparse_pages} sparse page(s) — likely image/diagram only")
    if pages_with_images_only:
        issues.append(f"{pages_with_images_only} page(s) with images but no text")

    grade = "GOOD" if not issues else ("ACCEPTABLE" if len(issues) <= 2 else "POOR")

    return {
        "page_count":             len(pages),
        "total_chars":            len(full_text.strip()),
        "total_tables":           sum(pg["analysis"]["table_count"] for pg in pages),
        "sparse_pages":           sparse_pages,
        "pages_with_images_only": pages_with_images_only,
        "grade":                  grade,
        "issues":                 issues,
        "error":                  None,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process_pdf(pdf_path: str | Path) -> dict:
    """Extract text and tables from a single PDF.

    Processes each page with pdfplumber, strips repeated headers/footers across
    pages, and returns the full structured result described in this module's docstring.
    Returns a result with stats["error"] set (and empty pages) if the file
    cannot be opened.
    """
    pdf_path = Path(pdf_path)
    result: dict = {"file": pdf_path.name, "path": str(pdf_path), "pages": [], "stats": {}}

    try:
        plumber = pdfplumber.open(str(pdf_path))
    except Exception as e:
        result["stats"] = _empty_stats(error=str(e))
        return result

    # Process pages one at a time, collecting raw text for cross-page deduplication.
    raw_texts: list[str] = []
    for page_num in range(1, len(plumber.pages) + 1):
        page_data, raw_text = _process_page(page_num, plumber.pages[page_num - 1])
        result["pages"].append(page_data)
        raw_texts.append(raw_text)

    plumber.close()

    # Header/footer stripping needs all pages' text — must run after the loop.
    cleaned = _strip_repeated_lines(raw_texts)
    for pg, cleaned_text in zip(result["pages"], cleaned):
        pg["text_cleaned"] = cleaned_text.strip()

    result["stats"] = _compute_stats(result["pages"], raw_texts)
    return result
