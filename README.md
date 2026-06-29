# Metso AI Sales Agent

An AI agent that assists salespeople during live customer conversations about Metso industrial equipment and services.

## Architecture

The agent uses a **two-stage routing + worker** pattern.

**Stage 1 — Intent pre-classification:**
A lightweight classification call (`CLASSIFICATION_MODEL`, default `gpt-4.1`) classifies each message into one of four categories (`product`, `cost`, `benchmark`, `general`) before any tool is invoked. This ensures the right tool is always called first — product questions are forced through brochure retrieval, cost questions are forced through equipment confirmation, and general messages are answered directly without any tool overhead.

**Stage 2 — Tool dispatch + synthesis:**
Based on the classified intent, the synthesis model (`SYNTHESIS_MODEL`, default `gpt-4.1`) is called with the appropriate tool locked via `tool_choice`. The tool returns structured data; the LLM narrates it into a natural-language answer.

| Capability | Data Source | Strategy | Why |
|---|---|---|---|
| Product Q&A | PDF brochures | Hybrid RAG (dense + BM25) | Semantic + exact model-name matching |
| Service cost estimate | Technician `.txt` reports | Equipment lookup → SQL aggregation | Costs require aggregation, not retrieval |
| Process benchmark | `flotation_process_data.csv` | Pure pandas statistics | Numeric data — compute, don't search |

## Key Design Decisions

- **Intent pre-classifier before tool dispatch** — a cheap single-word classification call (`CLASSIFICATION_MODEL`) classifies each message first, so the tool choice is deterministic rather than prompt-dependent. Product questions always search brochures; cost questions always confirm equipment first.
- **Two-step cost flow** — `list_equipment_services` is always called before `estimate_service_cost`. This confirms the equipment exists in our records and surfaces the exact service type label before any number is computed.
- **Retrieval vs. computation are different problems** — product Q&A is retrieval (vector search → LLM synthesises from passages); cost and benchmark are computation (SQL aggregation / pandas stats → LLM narrates the result). The LLM never does arithmetic.
- **SQLite for service records** — `MIN/MAX/AVG WHERE` gives auditable, deterministic ranges traceable to record IDs.
- **Hybrid search (dense + BM25 + RRF)** — semantic similarity misses exact tokens like "MD-650"; BM25 catches them; Reciprocal Rank Fusion combines both lists without score-scale bias.
- **Invoiced costs only** — only actual invoiced totals from FSRs are used; records without an explicit invoice total are excluded to avoid speculative estimates.

## Service Data Extraction

Field service reports (`.txt` files) are processed at ingestion time in three stages:

1. **LLM extraction** — each report is passed to `EXTRACTION_MODEL` (default `gpt-4.1`) with a structured prompt (`EXTRACTION_PROMPT` in `agent/prompts.py`) that pulls out service types, equipment model, country, total cost, currency, and a work summary. The model returns `null` for any field it cannot determine with confidence; it never guesses numeric values.
2. **Pydantic validation** — the JSON output is validated by the `ServiceRecord` model (`ingestion/services/service_records.py`). Records with no total cost are dropped.
3. **Cost classification** — each record is tagged `cost_source = "actual_recorded"` if both a total cost and currency are present, or `"no_invoice"` otherwise. Only `actual_recorded` rows are used when computing cost ranges for quotations; unpriced records are excluded to avoid speculative figures.

Extracted records are stored in `processed/service_records.db` (SQLite, committed to the repo) so reviewers can inspect them without re-running ingestion.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Place data files (flotation_process_data.csv is already included)
# data/Products/           — product brochure PDFs
# data/HistoricalServices/ — FSR-XXXX.txt service reports

# 4. Run ingestion
python -m ingestion.run_all

# 5. Launch UI
streamlit run app.py
```

## Example Conversations

**Product question:**
> Sales rep: "Does the MD series pump work for mill discharge applications?"
>
> Agent: "The MD series is specifically designed for mill discharge duty — the brochure describes it as built for high-abrasion, variable-flow conditions typical in grinding mill circuits. The Orion series covers general mineral processing but is not optimised for mill discharge. I'd recommend presenting the MD series for this application. *(Source: leaflet-mdseries-4241-08-23-en-mng.pdf)*"

---

**Service cost estimate (two-step flow):**
> Sales rep: "How much does a filter cloth replacement cost for a Larox filter in Sweden?"
>
> Agent *(Step 1 — equipment confirmation)*: "I found these Larox models with filter cloth replacement records: Larox PF, Larox PF-DS, Larox PF Pressure Filter. Which model does your customer have?"
>
> Sales rep: "It's a Larox PF-DS."
>
> Agent *(Step 2 — estimate)*: "Based on 2 invoiced EUR records for Larox PF-DS filter cloth replacement, I'd estimate **EUR 1,235–2,580** (median EUR 1,908). Confidence: low — only 2 records. Sources: FSR-0027 (Sweden), FSR-0031 (Finland). Only actual invoiced amounts are used — no estimates."

---

**Process benchmark:**
> Sales rep: "Our customer says they're running at 3.1% silica in concentrate and 64.5% iron. They think that's normal — is it?"
>
> Agent: "For silica concentrate, 3.1% is at the **77th percentile** of the historical dataset — meaning 77% of comparable operating periods achieved lower silica (better). For a metric where lower is better, this puts them in the weaker end of the distribution. Iron at 64.5% is at the **28th percentile**, below the historical median of 65.2%. The most correlated control levers with silica are Ore Pulp pH (correlation 0.20) and Column Air Flow (-0.18) — sites with adjusted airflow tend to see lower silica carry-over."

## Intermediate Artifacts

Running `python -m ingestion.run_all` generates two artifacts in `processed/`, both committed to the repo:

- `service_records.db` — SQLite database of extracted FSR records.
- `chroma_db/` — ChromaDB vector store for product brochures and service records.

## Limitations

- **Thin sample sizes**: The dataset contains 50 FSRs; no service type has more than a few invoiced records per currency. Ranges are wide and confidence is low by design — the architecture is correct, the data just needs to grow.
- **Image-only PDF specs**: Technical specifications embedded only in diagrams cannot be extracted by any text parser. The agent will say so rather than fabricate.
- **Multi-currency output**: Costs are grouped by currency and presented as separate ranges — no FX conversion is applied.
- **FSR-0028**: This file is truncated mid-table; cost is null and it is handled gracefully.

## What I'd Improve with More Time
- **Incremental ingestion** — current pipeline wipes and re-embeds the full corpus on every run; track file hashes to only process changed files.
- **Evaluation suite** with labelled Q&A pairs to measure retrieval precision.

