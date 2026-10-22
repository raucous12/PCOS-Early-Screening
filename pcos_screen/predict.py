"""Inference: load the trained artifact and score one questionnaire, with explanations."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .questionnaire import Answers, to_features
from .schema import FEATURE_BY_NAME, SOURCE

log = logging.getLogger(__name__)
EPS = 1e-6

# Answers explained together: BMI and the overweight flag describe one thing.
EXPLAIN_GROUPS: dict[str, tuple[str, ...]] = {"body_weight": ("bmi", "overweight")}
GROUP_LABELS = {"body_weight": "Body-mass index (BMI)"}


class ModelNotAvailable(RuntimeError):
    """Raised when the model artifact is missing or unreadable."""


def _logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return float(np.log(p / (1 - p)))


class PCOSPredictor:
    REQUIRED_KEYS = ("estimator", "features", "thresholds", "reference_source", "baseline")

    def __init__(self, artifact: dict[str, Any]):
        missing = [k for k in self.REQUIRED_KEYS if k not in artifact]
        if missing:
            raise ModelNotAvailable(f"Model artifact is missing keys: {missing}")
        self.artifact = artifact
        self.estimator = artifact["estimator"]
        self.features: list[str] = list(artifact["features"])
        self.thresholds: dict[str, float] = artifact["thresholds"]
        self.reference_source: str = artifact["reference_source"]
        self.baseline: dict[str, float] = artifact["baseline"]
        self.cuts: dict[str, float] = artifact.get("feature_cuts") or {}
        self.support: dict[str, list[float]] = artifact.get("support") or {}

    @classmethod
    def load(cls, path: str | Path) -> "PCOSPredictor":
        path = Path(path)
        if not path.is_file():
            raise ModelNotAvailable(f"Model file not found: {path}. Train it with `python -m pcos_screen.train`.")
        try:
            artifact = joblib.load(path)
        except Exception as exc:  # corrupt file, incompatible library versions, ...
            raise ModelNotAvailable(f"Could not load model {path}: {exc}") from exc
        if not isinstance(artifact, dict):
            raise ModelNotAvailable(f"Unexpected artifact type in {path}")
        log.info("Loaded model '%s' v%s (created %s)", artifact.get("model_name"), artifact.get("version"),
                 artifact.get("created"))
        return cls(artifact)

    # ------------------------------------------------------------------ core
    def features_for(self, answers: Answers) -> dict[str, float | None]:
        return to_features(answers,
                           overweight_bmi=self.cuts.get("overweight_bmi") or 25.0,
                           fast_food_cut=self.cuts.get("fast_food_per_week") or 3,
                           exercise_cut=self.cuts.get("exercise_days_per_week") or 3)

    def _frame(self, rows: list[dict[str, float | None]]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        for f in self.features:
            if f not in df:
                df[f] = np.nan
        df = df[self.features].astype(float)
        df[SOURCE] = self.reference_source
        return df

    def probabilities(self, rows: list[dict[str, float | None]]) -> np.ndarray:
        p = self.estimator.predict_proba(self._frame(rows))[:, 1]
        if not np.isfinite(p).all():
            raise FloatingPointError("Model produced non-finite probabilities")
        return p

    def band(self, p: float) -> str:
        if p >= self.thresholds["high"]:
            return "high"
        if p >= self.thresholds["low"]:
            return "moderate"
        return "low"

    def _baseline_value(self, feature: str) -> float:
        if feature in self.baseline:
            return float(self.baseline[feature])
        return 0.0  # binary symptom: "absent"

    def predict(self, answers: Answers) -> dict[str, Any]:
        feats = self.features_for(answers)
        row = {f: feats.get(f) for f in self.features}
        # Clamp numeric inputs to the range seen in training (no linear extrapolation).
        outside = []
        for f, (lo, hi) in self.support.items():
            v = row.get(f)
            if v is not None and not lo <= v <= hi:
                outside.append(FEATURE_BY_NAME[f].label)
                row[f] = float(min(max(v, lo), hi))

        # Occlusion explanation: move each answered item (or group) to its "typical,
        # symptom-free" value and measure the change in log-odds. Works for any model.
        variants, keys = [row], []
        handled: set[str] = set()
        for group, members in EXPLAIN_GROUPS.items():
            present = [m for m in members if m in self.features and row.get(m) is not None]
            if present:
                alt = dict(row)
                for m in present:
                    alt[m] = self._baseline_value(m) if m != "overweight" else float(self.baseline.get("bmi", 22) >= 25)
                variants.append(alt)
                keys.append(group)
            handled |= set(members)
        for f in self.features:
            # Age is an adjustment, not a sign: explanations are "compared with someone of your age".
            if f in handled or f == "age" or row.get(f) is None:
                continue
            if row[f] == self._baseline_value(f):
                continue  # already at the baseline: contributes nothing
            alt = dict(row)
            alt[f] = self._baseline_value(f)
            variants.append(alt)
            keys.append(f)
        # "No symptoms" reference for the same person: same age, healthy BMI, every sign absent.
        reference_row = {f: (self._baseline_value(f) if f != "overweight" else 0.0) for f in self.features}
        if "age" in reference_row:
            reference_row["age"] = row["age"]
        variants.append(reference_row)

        probs = self.probabilities(variants)
        p = float(probs[0])
        lp = _logit(p)
        contributions = []
        for key, q in zip(keys, probs[1:-1]):
            delta = lp - _logit(float(q))
            if abs(delta) < 0.02:
                continue
            label = GROUP_LABELS.get(key) or FEATURE_BY_NAME[key].label
            value = feats.get("bmi") if key == "body_weight" else row.get(key)
            contributions.append({"feature": key, "label": label, "value": value,
                                  "log_odds": round(delta, 3),
                                  "direction": "raises" if delta > 0 else "lowers"})
        contributions.sort(key=lambda c: -abs(c["log_odds"]))
        not_applicable = {"conception_difficulty"} if answers.conception_difficulty == "not_applicable" else set()
        unknown = [FEATURE_BY_NAME[f].label for f in self.features
                   if row.get(f) is None and f not in not_applicable]
        return {
            "probability": round(p, 4),
            "band": self.band(p),
            "thresholds": {"low": self.thresholds["low"], "high": self.thresholds["high"]},
            "reference_probability": round(float(probs[-1]), 4),
            "contributions": contributions,
            "unanswered": unknown,
            "outside_training_range": outside,
            "features": row,
            "model": {"name": self.artifact.get("model_name"), "version": self.artifact.get("version"),
                      "created": self.artifact.get("created")},
        }

    def info(self) -> dict[str, Any]:
        a = self.artifact
        return {"name": a.get("model_name"), "version": a.get("version"), "created": a.get("created"),
                "calibration": a.get("calibration"), "features": self.features,
                "feature_labels": {f: FEATURE_BY_NAME[f].label for f in self.features},
                "thresholds": self.thresholds, "summary": a.get("summary", {}),
                "reference_source": self.reference_source}
