"""
Process benchmarker — flotation process data analysis.

Pure pandas, no LLM, no retrieval. Loads the CSV once at import time
and reuses the same DataFrame for every query.

Public API:
    benchmark_process(readings)  ->  dict   agent tool entry point
    benchmark(readings)          ->  dict   direct use / tests
    load_data(csv_path)          ->  None   called by ingestion / tests
"""

import pandas as pd
import numpy as np

import config

# Module-level cache — loaded once at import, reused for every query.
_df: pd.DataFrame | None = None

# Maps agent-facing key names to CSV column names.
COLUMN_MAP: dict[str, str] = {
    "silica_pct":      "% Silica Concentrate",
    "iron_pct":        "% Iron Concentrate",
    "feed_iron_pct":   "% Iron Feed",
    "feed_silica_pct": "% Silica Feed",
}

# Columns that represent output quality (benchmarked against).
# All other numeric columns are treated as controllable process levers.
QUALITY_COLUMNS: list[str] = [
    "% Iron Concentrate",
    "% Silica Concentrate",
]


def load_data(csv_path: str = config.DATA_FLOTATION_CSV) -> None:
    """Load the flotation CSV into the module-level cache.

    The CSV uses European comma-decimal notation ("1,31"), so decimal=","
    is required for pandas to parse numeric columns correctly.

    Raises ValueError if the required quality columns are absent.
    """
    global _df
    _df = pd.read_csv(csv_path, decimal=",")
    missing = [col for col in QUALITY_COLUMNS if col not in _df.columns]
    if missing:
        raise ValueError(f"Missing required columns in flotation CSV: {missing}")


def benchmark_process(readings: dict[str, float]) -> dict:
    """Agent tool entry point: validate input types then delegate to benchmark().

    Accepts the readings dict from the LLM tool call and ensures every value
    can be coerced to float before passing to the computation layer.
    """
    if not readings:
        return {"found": False, "reason": "No readings provided"}

    validated: dict[str, float] = {}
    for key, value in readings.items():
        try:
            validated[key] = float(value)
        except (TypeError, ValueError):
            return {"found": False, "reason": f"Non-numeric value for '{key}': {value}"}

    return benchmark(validated)


def benchmark(readings: dict[str, float]) -> dict:
    """Compute percentile ranks and top correlated controls for the given readings.

    Returns found=False if the CSV hasn't been loaded or no recognised metrics
    were provided. All arithmetic is done in pandas — the LLM only narrates
    the returned dict.
    """
    global _df
    if _df is None:
        try:
            load_data()
        except Exception as exc:
            return {"found": False, "reason": str(exc)}

    mapped = {
        key: (COLUMN_MAP[key], value)
        for key, value in readings.items()
        if COLUMN_MAP.get(key) in _df.columns
    }
    if not mapped:
        return {"found": False, "reason": "No recognisable process metrics provided"}

    return {
        "found": True,
        "metrics": _compute_metrics(mapped),
        "top_correlated_controls": _top_correlated_controls(),
        "sample_size": len(_df),
    }


def _compute_metrics(mapped: dict[str, tuple[str, float]]) -> dict:
    """Compute percentile rank and quartile bounds for each mapped metric."""
    metrics = {}
    for key, (col, user_value) in mapped.items():
        series = _df[col].dropna()
        pct_rank = float((series < user_value).mean() * 100)
        metrics[key] = {
            "user_value":      user_value,
            "column_mapped":   col,
            "percentile_rank": round(pct_rank, 1),
            "p25":             float(series.quantile(0.25)),
            "median":          float(series.quantile(0.50)),
            "p75":             float(series.quantile(0.75)),
            "assessment":      _assessment_label(pct_rank),
        }
    return metrics


def _top_correlated_controls(n: int = 3) -> list[dict]:
    """Return the n control columns most correlated with any quality column.

    Each control column appears at most once, ranked by its highest absolute
    correlation across all quality columns.
    """
    quality_cols = [c for c in QUALITY_COLUMNS if c in _df.columns]
    control_cols = [
        c for c in _df.select_dtypes(include=[np.number]).columns
        if c not in quality_cols
    ]
    if not quality_cols or not control_cols:
        return []

    # Collect (control_col, abs_corr, raw_corr) for every quality x control pair.
    pairs: list[tuple[str, float, float]] = []
    for qcol in quality_cols:
        for ccol in control_cols:
            try:
                corr = _df[[qcol, ccol]].dropna().corr().iloc[0, 1]
                if not np.isnan(corr):
                    pairs.append((ccol, abs(corr), corr))
            except Exception:
                pass

    # Deduplicate by control column, keeping the highest-|corr| entry, then top n.
    seen: set[str] = set()
    top: list[dict] = []
    for col, _, raw_corr in sorted(pairs, key=lambda x: x[1], reverse=True):
        if col not in seen:
            seen.add(col)
            top.append({"column": col, "correlation": round(raw_corr, 3)})
        if len(top) >= n:
            break
    return top


def _assessment_label(pct_rank: float) -> str:
    """Map a percentile rank to a quartile label."""
    if pct_rank < 25:
        return "bottom_quartile"
    if pct_rank < 50:
        return "lower_half"
    if pct_rank < 75:
        return "upper_half"
    return "top_quartile"


# Pre-load on import — best-effort; silently skips if the CSV isn't present yet.
try:
    load_data()
except Exception:
    pass
