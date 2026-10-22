import numpy as np
import pandas as pd
import pytest

from pcos_screen.config import ConfigError, load_config
from pcos_screen.data import (DataSourceError, coverage_table, load_kerala_clinical, load_survey_mumbai,
                              yes_no)
from pcos_screen.schema import BINARY_FEATURES, BOUNDS, FEATURE_NAMES, SOURCE, SOURCES, TARGET


def test_all_sources_loaded_with_canonical_schema(data):
    assert set(data[SOURCE]) == set(SOURCES)
    assert list(data.columns) == list(FEATURE_NAMES) + [TARGET, SOURCE]
    assert len(data) > 1000
    assert set(data[TARGET].unique()) == {0, 1}


def test_binary_features_are_binary_or_missing(data):
    for f in BINARY_FEATURES:
        vals = set(data[f].dropna().unique())
        assert vals <= {0.0, 1.0}, f


def test_numeric_bounds_enforced(data):
    for col, (lo, hi) in BOUNDS.items():
        s = data[col].dropna()
        assert s.between(lo, hi).all(), col


def test_expected_source_sizes_and_prevalence(data):
    stats = data.groupby(SOURCE)[TARGET].agg(["size", "mean"])
    assert stats.loc["kerala", "size"] == 541
    assert 0.30 < stats.loc["kerala", "mean"] < 0.35  # 177 / 541
    assert (stats["mean"] > 0.1).all() and (stats["mean"] < 0.9).all()


def test_core_symptoms_always_observed_in_kerala(data):
    k = data[data[SOURCE] == "kerala"]
    for f in ("irregular_cycle", "excess_hair", "acne", "skin_darkening", "weight_gain", "bmi"):
        assert k[f].notna().mean() > 0.99, f


def test_coverage_table_shape(data):
    cov = coverage_table(data)
    assert cov.shape == (len(FEATURE_NAMES), len(SOURCES))
    assert ((cov >= 0) & (cov <= 1)).all().all()


def test_yes_no_mapping():
    s = pd.Series(["Yes", "NO", " yes ", "Maybe", None, 1, 0, "Not Applicable", "Yes(Detected Positive)"])
    out = yes_no(s).tolist()
    assert out[:3] == [1.0, 0.0, 1.0]
    assert np.isnan(out[3]) and np.isnan(out[4]) and np.isnan(out[7])
    assert out[5:7] == [1.0, 0.0] and out[8] == 1.0


def test_missing_file_raises(cfg, tmp_path):
    with pytest.raises(DataSourceError):
        load_survey_mumbai(tmp_path / "nope.csv", cfg)


def test_malformed_csv_raises(cfg, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(DataSourceError):
        load_survey_mumbai(bad, cfg)


def test_kerala_clinical_feature_sets(cfg, data):  # `data` skips the test when raw files are absent
    spec = cfg["sources"]["kerala"]
    df, sets = load_kerala_clinical(cfg.path("raw_dir") / spec["file"], spec["sheet"], cfg)
    assert len(df) == 541
    assert len(sets["questionnaire"]) < len(sets["questionnaire+anthropometry"]) < len(
        sets["questionnaire+anthropometry+labs+ultrasound"])
    assert df["follicles_total"].notna().mean() > 0.99


def test_config_overrides_and_errors(tmp_path):
    cfg = load_config(overrides=["training.n_trials=7", "new.section.key=abc"])
    assert cfg.get("training.n_trials") == 7 and cfg.get("new.section.key") == "abc"
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")
    with pytest.raises(ConfigError):
        load_config(overrides=["no_equals_sign"])
