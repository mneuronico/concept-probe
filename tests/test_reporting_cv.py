"""Tests for the cross-validated correctness model in reporting."""
import numpy as np
import pytest

from concept_probe.reporting import _fit_model


def _rows(n=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        correct = i % 2 == 0
        signal = 1.0 if correct else -1.0
        rows.append(
            {
                "correct": correct,
                "scores": {
                    "focus": signal + rng.normal(scale=0.8),
                    "planning": rng.normal(scale=1.0),
                },
            }
        )
    return rows


def test_fit_model_requires_two_per_class():
    rows = [{"correct": True, "scores": {"p": 1.0}}, {"correct": False, "scores": {"p": 0.0}}]
    result = _fit_model(rows, ["p"])
    assert result["cv_folds"] == 0
    assert "at least 2 rows of each class" in result["note"]


def test_fit_model_stratified_cv():
    pytest.importorskip("statsmodels")
    result = _fit_model(_rows(), ["focus", "planning"], seed=7)
    assert result["n"] == 40
    assert 2 <= result["cv_folds"] <= 5
    assert result["metrics_source"] == "out_of_fold_cv"
    # informative feature should classify well out-of-fold
    assert result["metrics"]["accuracy"] > 0.6
    names = [c["feature"] for c in result["coefficients"]]
    assert names == ["(intercept)", "focus", "planning"]
    coef = {c["feature"]: c for c in result["coefficients"]}
    assert coef["focus"]["coef"] > 0
    assert coef["focus"]["ci_low"] <= coef["focus"]["coef"] <= coef["focus"]["ci_high"]
