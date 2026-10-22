import numpy as np
import pytest

from pcos_screen.metrics import (bootstrap_ci, choose_thresholds, expected_calibration_error,
                                 paired_bootstrap_auc_diff, sens_spec, summarize)


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 3000)
    y = (rng.uniform(0, 1, 3000) < p).astype(int)  # perfectly calibrated by construction
    return y, p


def test_summarize_on_calibrated_scores(synthetic):
    y, p = synthetic
    s = summarize(y, p)
    assert 0.7 < s["roc_auc"] < 0.9
    assert s["ece"] < 0.05  # ~0.03 is pure sampling noise with 300 rows per bin
    assert 0.85 < s["calibration_slope"] < 1.15 and abs(s["calibration_intercept"]) < 0.15


def test_summarize_rejects_nan():
    with pytest.raises(ValueError):
        summarize([0, 1], [np.nan, 0.5])


def test_thresholds_meet_targets(synthetic):
    y, p = synthetic
    th = choose_thresholds(y, p, 0.9, 0.8)
    assert th["low"] < th["high"]
    assert sens_spec(y, p, th["low"])[0] >= 0.9
    assert sens_spec(y, p, th["high"])[1] >= 0.8


def test_thresholds_stay_ordered_for_useless_model():
    y = np.array([0, 1] * 50)
    p = np.full(100, 0.5)
    th = choose_thresholds(y, p)
    assert th["high"] > th["low"]


def test_bootstrap_ci_contains_estimate(synthetic):
    from sklearn.metrics import roc_auc_score
    y, p = synthetic
    lo, hi = bootstrap_ci(y, p, n=200)
    assert lo < roc_auc_score(y, p) < hi


def test_paired_bootstrap_detects_noise(synthetic):
    y, p = synthetic
    noisy = np.clip(p + np.random.default_rng(1).normal(0, 0.4, len(p)), 0, 1)
    d = paired_bootstrap_auc_diff(y, p, noisy, n=200)
    assert d["diff"] > 0 and d["ci_low"] > 0 and d["p_value"] < 0.05


def test_ece_detects_miscalibration(synthetic):
    y, p = synthetic
    assert expected_calibration_error(y, np.clip(p * 0.5, 0, 1)) > 0.2
