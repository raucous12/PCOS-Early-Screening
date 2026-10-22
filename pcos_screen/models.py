"""Model zoo: preprocessing, candidate classifiers and their Optuna search spaces.

All models consume a DataFrame with the canonical features plus a ``source``
column. ``Preprocessor`` one-hot encodes the source as a fixed effect, so symptom
effects are learned *within* each source instead of being confounded by the
very different PCOS prevalence of each dataset.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .schema import BINARY_FEATURES, FEATURE_BY_NAME, FEATURE_NAMES, NUMERIC_FEATURES, SOURCE

SRC_PREFIX = "src_"


# Mixins come before BaseEstimator so scikit-learn's estimator tags (e.g. "this is a
# classifier") resolve correctly; calibration and scoring rely on them.
class Preprocessor(TransformerMixin, BaseEstimator):
    """Select canonical features, optionally impute and scale them, and one-hot encode the source.

    * ``impute``: numeric gets the median, binary gets the mean (a probability,
      so a missing answer counts as "average"). Tree models that handle NaN
      natively can skip this.
    * ``scale``: standardise the numeric columns (age, BMI) for linear models.
      Binary columns stay 0/1, so coefficients read as log-odds per symptom.
    """

    def __init__(self, features: tuple[str, ...] = FEATURE_NAMES, impute: bool = True, scale: bool = False):
        self.features = features
        self.impute = impute
        self.scale = scale

    def fit(self, X: pd.DataFrame, y=None):
        X = self._check(X)
        feats = X[list(self.features)].astype(float)
        num = [c for c in self.features if c in NUMERIC_FEATURES]
        binary = [c for c in self.features if c in BINARY_FEATURES]
        self.numeric_ = num
        self.fill_values_ = pd.concat([feats[num].median(), feats[binary].mean()])
        # Columns that are entirely missing in the training data fall back to neutral values.
        self.fill_values_ = self.fill_values_.fillna({c: 0.5 for c in binary}).fillna(0.0)
        imputed = feats.fillna(self.fill_values_)
        self.mean_ = imputed[num].mean()
        self.std_ = imputed[num].std(ddof=0).replace(0, 1.0)
        self.sources_ = sorted(X[SOURCE].astype(str).unique())
        self.feature_names_out_ = list(self.features) + [SRC_PREFIX + s for s in self.sources_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "fill_values_")
        X = self._check(X)
        out = X[list(self.features)].astype(float).copy()
        if self.impute:
            out = out.fillna(self.fill_values_)
        if self.scale:
            for c in self.numeric_:
                out[c] = (out[c] - self.mean_[c]) / self.std_[c]
        src = X[SOURCE].astype(str)
        for s in self.sources_:
            out[SRC_PREFIX + s] = (src == s).astype(float).to_numpy()
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)

    def _check(self, X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Preprocessor expects a pandas DataFrame")
        missing = [c for c in (*self.features, SOURCE) if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing columns: {missing}")
        unknown = [c for c in self.features if c not in FEATURE_BY_NAME]
        if unknown:
            raise ValueError(f"Unknown features: {unknown}")
        return X


def monotone_vector(columns) -> list[int]:
    return [FEATURE_BY_NAME[c].monotone if c in FEATURE_BY_NAME else 0 for c in columns]


class MonotoneBoosting(ClassifierMixin, BaseEstimator):
    """Gradient boosting (LightGBM, XGBoost or sklearn HistGB) with monotone constraints.

    The constraint vector is built at ``fit`` time from the column names, because
    the number of source dummies changes between CV folds. A symptom feature
    can then only *increase* predicted risk, which matches medical knowledge and
    stabilises the model on small data.
    """

    def __init__(self, backend: str = "lightgbm", params: dict | None = None, random_state: int = 0):
        self.backend = backend
        self.params = params
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y):
        cons = monotone_vector(X.columns)
        p = dict(self.params or {})
        if self.backend == "lightgbm":
            from lightgbm import LGBMClassifier
            self.model_ = LGBMClassifier(monotone_constraints=cons, monotone_constraints_method="advanced",
                                         random_state=self.random_state, n_jobs=1, verbose=-1, **p)
        elif self.backend == "xgboost":
            from xgboost import XGBClassifier
            self.model_ = XGBClassifier(monotone_constraints=tuple(cons), random_state=self.random_state,
                                        n_jobs=1, eval_metric="logloss", tree_method="hist", **p)
        elif self.backend == "hist_gb":
            self.model_ = HistGradientBoostingClassifier(monotonic_cst=cons, random_state=self.random_state,
                                                         early_stopping=False, **p)
        else:
            raise ValueError(f"Unknown backend {self.backend}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.model_.fit(X, np.asarray(y))
        self.classes_ = self.model_.classes_
        self.columns_ = list(X.columns)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return self.model_.predict_proba(X[self.columns_])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


class SoftVote(ClassifierMixin, BaseEstimator):
    """Average the probabilities of already-configured pipelines."""

    def __init__(self, estimators: list[Any] | None = None):
        self.estimators = estimators

    def fit(self, X, y):
        self.estimators_ = [clone(e).fit(X, y) for e in self.estimators or []]
        if not self.estimators_:
            raise ValueError("SoftVote needs at least one estimator")
        self.classes_ = self.estimators_[0].classes_
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "estimators_")
        return np.mean([e.predict_proba(X) for e in self.estimators_], axis=0)

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


# --------------------------------------------------------------------------- model specs
Features = tuple[str, ...]


@dataclass
class ModelSpec:
    name: str
    build: Callable[[dict, int, Features], Pipeline]
    space: Callable[[Any], dict] | None
    description: str


def _logreg(p: dict, seed: int, feats: Features = FEATURE_NAMES) -> Pipeline:
    clf = LogisticRegression(solver="saga", max_iter=5000, random_state=seed,
                             C=p.get("C", 1.0), l1_ratio=p.get("l1_ratio", 0.0))
    return Pipeline([("prep", Preprocessor(feats, impute=True, scale=True)), ("clf", clf)])


def _logreg_space(t) -> dict:
    return {"C": t.suggest_float("C", 1e-3, 10.0, log=True), "l1_ratio": t.suggest_float("l1_ratio", 0.0, 1.0)}


def _rf(p: dict, seed: int, feats: Features = FEATURE_NAMES) -> Pipeline:
    clf = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=1, **p)
    return Pipeline([("prep", Preprocessor(feats, impute=True, scale=False)), ("clf", clf)])


def _rf_space(t) -> dict:
    return {"max_depth": t.suggest_int("max_depth", 2, 10),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 30),
            "max_features": t.suggest_float("max_features", 0.2, 0.9)}


def _hgb(p: dict, seed: int, feats: Features = FEATURE_NAMES) -> Pipeline:
    return Pipeline([("prep", Preprocessor(feats, impute=False)),
                     ("clf", MonotoneBoosting("hist_gb", p, seed))])


def _hgb_space(t) -> dict:
    return {"learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter": t.suggest_int("max_iter", 50, 400),
            "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 4, 31),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 5, 60),
            "l2_regularization": t.suggest_float("l2_regularization", 1e-3, 10.0, log=True)}


def _lgbm(p: dict, seed: int, feats: Features = FEATURE_NAMES) -> Pipeline:
    return Pipeline([("prep", Preprocessor(feats, impute=False)), ("clf", MonotoneBoosting("lightgbm", p, seed))])


def _lgbm_space(t) -> dict:
    return {"n_estimators": t.suggest_int("n_estimators", 50, 500),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": t.suggest_int("num_leaves", 3, 31),
            "min_child_samples": t.suggest_int("min_child_samples", 5, 60),
            "subsample": t.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 20.0, log=True)}


def _xgb(p: dict, seed: int, feats: Features = FEATURE_NAMES) -> Pipeline:
    return Pipeline([("prep", Preprocessor(feats, impute=False)), ("clf", MonotoneBoosting("xgboost", p, seed))])


def _xgb_space(t) -> dict:
    return {"n_estimators": t.suggest_int("n_estimators", 50, 500),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": t.suggest_int("max_depth", 2, 6),
            "min_child_weight": t.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "subsample": t.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 20.0, log=True)}


MODEL_SPECS: dict[str, ModelSpec] = {
    "logreg": ModelSpec("logreg", _logreg, _logreg_space, "Elastic-net logistic regression"),
    "random_forest": ModelSpec("random_forest", _rf, _rf_space, "Random forest"),
    "hist_gb": ModelSpec("hist_gb", _hgb, _hgb_space, "sklearn HistGradientBoosting (monotone)"),
    "lightgbm": ModelSpec("lightgbm", _lgbm, _lgbm_space, "LightGBM (monotone constraints)"),
    "xgboost": ModelSpec("xgboost", _xgb, _xgb_space, "XGBoost (monotone constraints)"),
}
ENSEMBLE_MEMBERS = ("logreg", "lightgbm", "random_forest")


def build_ensemble(member_params: dict[str, dict], seed: int, feats: Features = FEATURE_NAMES) -> SoftVote:
    return SoftVote([MODEL_SPECS[m].build(member_params[m], seed, feats) for m in ENSEMBLE_MEMBERS])
