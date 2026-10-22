import numpy as np
import pandas as pd
import pytest

from pcos_screen.models import MODEL_SPECS, Preprocessor, SoftVote, build_ensemble, monotone_vector
from pcos_screen.schema import FEATURE_BY_NAME, SOURCE, TARGET

FAST = {
    "logreg": {"C": 0.5, "l1_ratio": 0.2},
    "random_forest": {"max_depth": 4, "min_samples_leaf": 10, "max_features": 0.5},
    "hist_gb": {"learning_rate": 0.1, "max_iter": 60, "max_leaf_nodes": 7, "min_samples_leaf": 20, "l2_regularization": 1.0},
    "lightgbm": {"n_estimators": 60, "learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 20},
    "xgboost": {"n_estimators": 60, "learning_rate": 0.1, "max_depth": 3},
}


def test_preprocessor_columns_imputation_and_scaling(data, model_features):
    prep = Preprocessor(model_features, impute=True, scale=True).fit(data)
    out = prep.transform(data)
    assert list(out.columns) == list(model_features) + [f"src_{s}" for s in sorted(data[SOURCE].unique())]
    assert not out.isna().any().any()
    assert abs(out["age"].mean()) < 1e-6
    # Each row has exactly one source dummy set.
    assert (out.filter(like="src_").sum(axis=1) == 1).all()


def test_preprocessor_unknown_source_and_missing_columns(data, model_features):
    prep = Preprocessor(model_features).fit(data)
    row = data.iloc[[0]].copy()
    row[SOURCE] = "never_seen"
    assert prep.transform(row).filter(like="src_").sum(axis=1).iloc[0] == 0
    with pytest.raises(ValueError):
        prep.transform(data.drop(columns=["age"]))
    with pytest.raises(TypeError):
        prep.transform(data.to_numpy())


def test_monotone_vector_matches_schema():
    cols = ["age", "irregular_cycle", "src_kerala"]
    assert monotone_vector(cols) == [0, FEATURE_BY_NAME["irregular_cycle"].monotone, 0]


@pytest.mark.parametrize("name", list(MODEL_SPECS))
def test_every_model_fits_and_predicts_probabilities(data, model_features, name):
    est = MODEL_SPECS[name].build(FAST[name], 0, model_features).fit(data, data[TARGET])
    p = est.predict_proba(data)[:, 1]
    assert p.shape == (len(data),) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(data[TARGET], p) > 0.75  # in-sample sanity, not a performance claim


@pytest.mark.parametrize("name", ["lightgbm", "xgboost", "hist_gb"])
def test_monotone_constraints_hold(data, model_features, name):
    """Turning a symptom on must never lower the predicted risk."""
    est = MODEL_SPECS[name].build(FAST[name], 0, model_features).fit(data, data[TARGET])
    rows = data.sample(200, random_state=1).reset_index(drop=True)
    for f in model_features:
        if FEATURE_BY_NAME[f].monotone != 1 or FEATURE_BY_NAME[f].kind != "binary":
            continue
        off, on = rows.copy(), rows.copy()
        off[f], on[f] = 0.0, 1.0
        diff = est.predict_proba(on)[:, 1] - est.predict_proba(off)[:, 1]
        assert (diff >= -1e-9).all(), f"{name}: {f} lowered risk"


def test_soft_vote_averages(data, model_features):
    members = {m: FAST[m] for m in ("logreg", "lightgbm", "random_forest")}
    ens = build_ensemble(members, 0, model_features).fit(data, data[TARGET])
    parts = [MODEL_SPECS[m].build(FAST[m], 0, model_features).fit(data, data[TARGET]).predict_proba(data)[:, 1]
             for m in members]
    np.testing.assert_allclose(ens.predict_proba(data)[:, 1], np.mean(parts, axis=0), rtol=1e-6)
    with pytest.raises(ValueError):
        SoftVote([]).fit(data, data[TARGET])
