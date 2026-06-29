import pytest
import pandas as pd
import numpy as np
from agent.benchmarker import benchmark, load_data, _assessment_label
import agent.benchmarker as fa


@pytest.fixture(autouse=True)
def inject_test_df():
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "% Iron Concentrate": np.random.normal(65, 3, n),
        "% Silica Concentrate": np.random.normal(2.5, 0.8, n),
        "Flotation Column 01 Air Flow": np.random.normal(200, 30, n),
        "Flotation Column 01 Level": np.random.normal(150, 20, n),
        "Feed Rate": np.random.normal(100, 10, n),
    })
    fa._df = df
    yield
    fa._df = None


def test_percentile_rank_between_0_and_100():
    result = benchmark({"iron_pct": 65.0})
    assert result["found"] is True
    rank = result["metrics"]["iron_pct"]["percentile_rank"]
    assert 0.0 <= rank <= 100.0


def test_benchmark_returns_top_correlated_controls():
    result = benchmark({"iron_pct": 65.0})
    assert result["found"] is True
    corrs = result["top_correlated_controls"]
    assert isinstance(corrs, list)
    assert len(corrs) >= 1
    for c in corrs:
        assert "column" in c
        assert "correlation" in c


def test_unknown_metric_key_returns_error_not_crash():
    result = benchmark({"nonexistent_metric": 99.9})
    assert result["found"] is False
    assert "reason" in result


def test_assessment_label_matches_percentile():
    assert _assessment_label(10.0) == "bottom_quartile"
    assert _assessment_label(40.0) == "lower_half"
    assert _assessment_label(70.0) == "upper_half"
    assert _assessment_label(90.0) == "top_quartile"