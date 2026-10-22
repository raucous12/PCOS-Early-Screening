"""Shared fixtures. The models here are fitted on the fly, so the tests never depend
on the (slow) full training run having happened."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pcos_screen.config import load_config  # noqa: E402
from pcos_screen.data import DataSourceError, load_all  # noqa: E402
from pcos_screen.metrics import choose_thresholds  # noqa: E402
from pcos_screen.models import MODEL_SPECS  # noqa: E402
from pcos_screen.predict import PCOSPredictor  # noqa: E402
from pcos_screen.schema import TARGET  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def data(cfg):
    # The raw datasets are not in the repository; on a fresh clone the tests that need
    # them are skipped (not failed) until `python scripts/download_data.py` has been run.
    try:
        return load_all(cfg)
    except DataSourceError as exc:
        pytest.skip(f"raw data not available ({exc}); run: python scripts/download_data.py")


@pytest.fixture(scope="session")
def model_features(cfg):
    return tuple(cfg.get("features.model_features"))


def _artifact(est, feats, data, name):
    p = est.predict_proba(data)[:, 1]
    return {"version": "test", "created": "test", "model_name": name, "estimator": est,
            "features": list(feats), "reference_source": "kerala", "calibration": "none",
            "thresholds": choose_thresholds(data[TARGET], p), "baseline": {"age": 26.0, "bmi": 22.0},
            "feature_cuts": {"overweight_bmi": 25.0, "fast_food_per_week": 3, "exercise_days_per_week": 3},
            "summary": {"roc_auc": 0.0}}


@pytest.fixture(scope="session")
def artifact(data, model_features):
    est = MODEL_SPECS["logreg"].build({"C": 0.3, "l1_ratio": 0.0}, 0, model_features).fit(data, data[TARGET])
    return _artifact(est, model_features, data, "logreg")


@pytest.fixture(scope="session")
def tree_artifact(data, model_features):
    params = {"n_estimators": 150, "learning_rate": 0.05, "num_leaves": 7, "min_child_samples": 20}
    est = MODEL_SPECS["lightgbm"].build(params, 0, model_features).fit(data, data[TARGET])
    return _artifact(est, model_features, data, "lightgbm")


@pytest.fixture(scope="session")
def predictor(artifact):
    return PCOSPredictor(artifact)


@pytest.fixture(scope="session")
def model_file(artifact, tmp_path_factory):
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(artifact, path)
    return path
