"""Train, validate, select, calibrate and save the PCOS screening model.

Pipeline
--------
1. Load and harmonise every enabled data source (``data.load_all``).
2. **Nested, repeated, stratified CV**: Optuna tunes each candidate model on the
   inner folds only (log-loss objective); the outer folds give unbiased
   out-of-fold (OOF) predictions.
3. **Leave-one-source-out (LOSO)** external validation: tune and fit on 3 sources,
   test on the unseen 4th.
4. Select the model with the best mean LOSO AUC, preferring a simpler model when it
   is within ``training.simplicity_margin``.
5. Pick a calibration method (none / sigmoid / isotonic) by out-of-fold Brier score.
6. Fit the final model on all data, derive risk-band thresholds from calibrated OOF
   predictions, and save a versioned artifact plus a JSON metrics report.
7. Extra analyses: feature-set ablation, and a Kerala "clinical ceiling" benchmark
   showing how much lab tests and ultrasound add.

Run ``python -m pcos_screen.train --help`` for options.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import __version__
from .config import Config, ConfigError, load_config
from .data import DataSourceError, coverage_table, load_all, load_kerala_clinical
from .metrics import (bootstrap_ci, choose_thresholds, paired_bootstrap_auc_diff, reliability_curve,
                      summarize)
from .models import ENSEMBLE_MEMBERS, MODEL_SPECS, build_ensemble
from .schema import FEATURE_BY_NAME, FEATURE_NAMES, SOURCE, TARGET

log = logging.getLogger("pcos_screen.train")
COMPLEXITY_ORDER = ["logreg", "random_forest", "hist_gb", "lightgbm", "xgboost", "ensemble"]


# --------------------------------------------------------------------------- utilities
def strat_labels(df: pd.DataFrame) -> pd.Series:
    """Stratify folds on source x label, so every fold keeps every source's prevalence."""
    return df[SOURCE].astype(str) + "_" + df[TARGET].astype(str)


def reference_for(train_sources: list[str], preferred: str) -> str:
    return preferred if preferred in train_sources else sorted(train_sources)[0]


def with_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df.copy()
    out[SOURCE] = source
    return out


def _check_finite(p: np.ndarray, where: str) -> np.ndarray:
    if not np.isfinite(p).all():
        raise FloatingPointError(f"Non-finite predictions in {where}")
    return p


# --------------------------------------------------------------------------- tuning
def tune(name: str, X: pd.DataFrame, feats: tuple[str, ...], n_trials: int, inner_folds: int,
         seed: int) -> dict:
    """Optuna TPE search on inner stratified folds; returns the best parameter dict."""
    import optuna

    spec = MODEL_SPECS[name]
    if spec.space is None or n_trials <= 0:
        return {}
    y = X[TARGET].to_numpy()
    splits = list(StratifiedKFold(inner_folds, shuffle=True, random_state=seed).split(X, strat_labels(X)))

    def objective(trial) -> float:
        params = spec.space(trial)
        losses = []
        for tr, va in splits:
            model = spec.build(params, seed, feats).fit(X.iloc[tr], y[tr])
            p = _check_finite(model.predict_proba(X.iloc[va])[:, 1], f"{name} tuning")
            losses.append(log_loss(y[va], np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))
        return float(np.mean(losses))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, catch=(ValueError, FloatingPointError))
    if not study.best_trials:
        raise RuntimeError(f"All tuning trials failed for {name}")
    return spec.space(optuna.trial.FixedTrial(study.best_params))


def build(name: str, params: dict, seed: int, feats: tuple[str, ...]):
    if name == "ensemble":
        return build_ensemble(params, seed, feats)
    return MODEL_SPECS[name].build(params, seed, feats)


def tune_all(models: list[str], X: pd.DataFrame, feats, n_trials, inner_folds, seed) -> dict[str, Any]:
    """Tune every base model needed by ``models`` once; the ensemble reuses member params."""
    needed = {m for m in models if m != "ensemble"}
    if "ensemble" in models:
        needed |= set(ENSEMBLE_MEMBERS)
    tuned = {m: tune(m, X, feats, n_trials, inner_folds, seed) for m in sorted(needed)}
    if "ensemble" in models:
        tuned["ensemble"] = {m: tuned[m] for m in ENSEMBLE_MEMBERS}
    return tuned


# --------------------------------------------------------------------------- fold tasks
def run_fold(task: dict, data: pd.DataFrame, models: list[str], feats, cfg_t: dict, seed: int) -> dict:
    """Tune and fit every model on one train split; predict the test split.

    ``task['kind']`` is ``cv`` (test rows keep their own source) or ``loso`` (the
    held-out source is unseen, so its rows are scored with the reference source).
    """
    warnings.filterwarnings("ignore")
    train, test = data.iloc[task["train"]], data.iloc[task["test"]]
    if task["kind"] == "loso":
        test = with_source(test, reference_for(sorted(train[SOURCE].unique()), task["reference"]))
    out = {"task": {k: v for k, v in task.items() if k not in ("train", "test")}, "preds": {}, "params": {},
           "errors": {}}
    tuned = tune_all(models, train, feats, cfg_t["n_trials"], cfg_t["inner_folds"], seed + task["id"])
    for m in models:
        try:
            est = build(m, tuned[m], seed, feats).fit(train, train[TARGET].to_numpy())
            out["preds"][m] = _check_finite(est.predict_proba(test)[:, 1], m)
            out["params"][m] = tuned[m]
        except (ValueError, FloatingPointError, MemoryError) as exc:
            log.warning("Model %s failed on %s: %s", m, out["task"], exc)
            out["errors"][m] = repr(exc)
    return out


def make_tasks(data: pd.DataFrame, cfg: Config) -> list[dict]:
    t = cfg["training"]
    seed = cfg.get("seed", 42)
    tasks, tid = [], 0
    strat = strat_labels(data)
    for r in range(t["outer_repeats"]):
        skf = StratifiedKFold(t["outer_folds"], shuffle=True, random_state=seed + r)
        for k, (tr, te) in enumerate(skf.split(data, strat)):
            tasks.append({"kind": "cv", "repeat": r, "fold": k, "id": tid, "train": tr, "test": te})
            tid += 1
    sources = sorted(data[SOURCE].unique())
    if len(sources) > 1:
        for s in sources:
            te = np.flatnonzero(data[SOURCE] == s)
            tr = np.flatnonzero(data[SOURCE] != s)
            tasks.append({"kind": "loso", "held_out": s, "id": tid, "train": tr, "test": te,
                          "reference": cfg.get("reference_source", "kerala")})
            tid += 1
    return tasks


def execute(tasks, data, models, feats, cfg) -> list[dict]:
    t = cfg["training"]
    n_jobs = t.get("n_jobs", -1)
    args = (data, models, feats, {"n_trials": t["n_trials"], "inner_folds": t["inner_folds"]}, cfg.get("seed", 42))
    try:
        return Parallel(n_jobs=n_jobs, verbose=5)(delayed(run_fold)(task, *args) for task in tasks)
    except MemoryError:
        log.warning("Out of memory with n_jobs=%s; retrying sequentially", n_jobs)
        return [run_fold(task, *args) for task in tasks]


# --------------------------------------------------------------------------- aggregation
def aggregate(results: list[dict], tasks: list[dict], data: pd.DataFrame, models: list[str], cfg: Config) -> dict:
    test_rows = {t["id"]: t["test"] for t in tasks}
    y = data[TARGET].to_numpy()
    n_rep = cfg["training"]["outer_repeats"]
    n_boot = cfg["training"].get("bootstrap", 1000)
    report: dict[str, Any] = {}
    oof_mean: dict[str, np.ndarray] = {}
    for m in models:
        oof = np.full((n_rep, len(data)), np.nan)
        fold_auc, fold_brier = [], []
        loso = {}
        for res in results:
            if m not in res["preds"]:
                continue
            task, p = res["task"], res["preds"][m]
            idx = test_rows[task["id"]]
            if task["kind"] == "cv":
                oof[task["repeat"], idx] = p
                fold_auc.append(roc_auc_score(y[idx], p))
                fold_brier.append(brier_score_loss(y[idx], p))
            else:
                s = summarize(y[idx], p)
                s["roc_auc_ci"] = bootstrap_ci(y[idx], p, n=n_boot)
                loso[task["held_out"]] = s
        if np.isnan(oof).all():
            log.warning("No successful CV predictions for %s; excluded", m)
            continue
        p_mean = np.nanmean(oof, axis=0)
        if np.isnan(p_mean).any():
            log.warning("%s: %d rows without OOF predictions; excluded", m, int(np.isnan(p_mean).sum()))
            continue
        oof_mean[m] = p_mean
        per_source = {s: summarize(y[(data[SOURCE] == s).to_numpy()], p_mean[(data[SOURCE] == s).to_numpy()])
                      for s in sorted(data[SOURCE].unique())}
        summary = summarize(y, p_mean)
        summary["roc_auc_ci"] = bootstrap_ci(y, p_mean, n=n_boot)
        loso_aucs = [v["roc_auc"] for v in loso.values() if "roc_auc" in v]
        report[m] = {
            "cv": {"fold_auc_mean": float(np.mean(fold_auc)), "fold_auc_sd": float(np.std(fold_auc)),
                   "fold_brier_mean": float(np.mean(fold_brier)), "pooled_oof": summary,
                   "per_source": per_source},
            "loso": {"per_source": loso, "mean_auc": float(np.mean(loso_aucs)) if loso_aucs else float("nan")},
        }
    return {"models": report, "oof": oof_mean}


def select_model(report: dict, margin: float) -> tuple[str, str]:
    scored = {m: r["loso"]["mean_auc"] if np.isfinite(r["loso"]["mean_auc"]) else r["cv"]["pooled_oof"]["roc_auc"]
              for m, r in report.items()}
    if not scored:
        raise RuntimeError("Every candidate model failed")
    best = max(scored, key=scored.get)
    for m in COMPLEXITY_ORDER:
        if m in scored and scored[m] >= scored[best] - margin:
            reason = (f"best LOSO AUC {scored[best]:.3f} ({best}); chose '{m}' ({scored[m]:.3f}), "
                      f"the simplest model within {margin:.3f}")
            return m, reason
    return best, f"best LOSO AUC {scored[best]:.3f}"


# --------------------------------------------------------------------------- calibration
def _calibrated(est, method: str, X: pd.DataFrame, seed: int):
    if method == "none":
        return est
    splits = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strat_labels(X)))
    return CalibratedClassifierCV(est, method=method, cv=splits)


def _cal_fold(task, data, name, params, feats, methods, seed):
    warnings.filterwarnings("ignore")
    train, test = data.iloc[task["train"]], data.iloc[task["test"]]
    preds = {}
    for method in methods:
        est = _calibrated(build(name, params, seed, feats), method, train, seed)
        est.fit(train, train[TARGET].to_numpy())
        preds[method] = _check_finite(est.predict_proba(test)[:, 1], f"calibration {method}")
    return task["repeat"], task["test"], preds


def compare_calibration(data, tasks, name, fold_params, feats, cfg) -> dict:
    methods = ["none", "sigmoid", "isotonic"]
    choice = cfg["training"].get("calibration", "auto")
    if choice != "auto":
        methods = [choice]
    cv_tasks = [t for t in tasks if t["kind"] == "cv" and t["id"] in fold_params]
    seed = cfg.get("seed", 42)
    res = Parallel(n_jobs=cfg["training"].get("n_jobs", -1))(
        delayed(_cal_fold)(t, data, name, fold_params[t["id"]], feats, methods, seed) for t in cv_tasks)
    y = data[TARGET].to_numpy()
    n_rep = cfg["training"]["outer_repeats"]
    out = {}
    for method in methods:
        oof = np.full((n_rep, len(data)), np.nan)
        for rep, idx, preds in res:
            oof[rep, idx] = preds[method]
        p = np.nanmean(oof, axis=0)
        out[method] = {"oof": p, "summary": summarize(y, p)}
    best = min(out, key=lambda k: out[k]["summary"]["brier"])
    return {"chosen": best, "results": out}


# --------------------------------------------------------------------------- extra analyses
def ablation(data: pd.DataFrame, cfg: Config) -> list[dict]:
    """Fixed-hyper-parameter LOSO comparison of feature sets (why lifestyle answers were dropped)."""
    model_feats = tuple(cfg.get("features.model_features", FEATURE_NAMES))
    lifestyle, wellbeing = {"fast_food", "regular_exercise"}, {"fatigue", "mood_swings"}
    variants = {
        "all 14 questionnaire features": FEATURE_NAMES,
        "without lifestyle (fast food, exercise)": tuple(f for f in FEATURE_NAMES if f not in lifestyle),
        "without lifestyle and wellbeing": tuple(f for f in FEATURE_NAMES if f not in lifestyle | wellbeing),
        "selected model features": model_feats,
    }
    params = {"logreg": {"C": 0.3, "l1_ratio": 0.0},
              "lightgbm": {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 7, "min_child_samples": 20}}
    rows = []
    for vname, feats in variants.items():
        for m, p in params.items():
            aucs = {}
            for s in sorted(data[SOURCE].unique()):
                tr, te = data[data[SOURCE] != s], data[data[SOURCE] == s]
                te = with_source(te, reference_for(sorted(tr[SOURCE].unique()), cfg.get("reference_source")))
                est = MODEL_SPECS[m].build(p, 0, feats).fit(tr, tr[TARGET].to_numpy())
                aucs[s] = float(roc_auc_score(te[TARGET], est.predict_proba(te)[:, 1]))
            rows.append({"variant": vname, "n_features": len(feats), "model": m, "loso_auc": aucs,
                         "loso_mean_auc": float(np.mean(list(aucs.values())))})
    return rows


def clinical_benchmark(cfg: Config) -> list[dict]:
    """Kerala-only CV AUC for questionnaire vs + anthropometry vs + labs/ultrasound."""
    from lightgbm import LGBMClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    spec = cfg["sources"].get("kerala", {})
    if not spec.get("enabled", True):
        return []
    df, sets = load_kerala_clinical(cfg.path("raw_dir") / spec["file"], spec.get("sheet"), cfg)
    y = df[TARGET].to_numpy()
    rows = []
    for set_name, cols in sets.items():
        X = df[cols].astype(float)
        candidates = {
            "logreg": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                            LogisticRegression(C=0.5, max_iter=5000)),
            "lightgbm": lambda: LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7,
                                               min_child_samples=15, subsample=0.8, subsample_freq=1,
                                               colsample_bytree=0.8, verbose=-1, n_jobs=1, random_state=0),
        }
        for m, make in candidates.items():
            aucs = []
            for r in range(cfg["training"]["outer_repeats"]):
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
                    est = make().fit(X.iloc[tr], y[tr])
                    aucs.append(roc_auc_score(y[te], est.predict_proba(X.iloc[te])[:, 1]))
            rows.append({"feature_set": set_name, "n_features": len(cols), "model": m,
                         "auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs))})
    return rows


def global_importance(est, data: pd.DataFrame, feats, seed: int) -> list[dict]:
    """Permutation importance (drop in AUC) of each feature for the final model."""
    from sklearn.inspection import permutation_importance

    X = data[list(feats) + [SOURCE]]
    pi = permutation_importance(est, X, data[TARGET], scoring="roc_auc", n_repeats=20, random_state=seed,
                                n_jobs=1)
    cols = list(X.columns)
    rows = [{"feature": c, "label": FEATURE_BY_NAME[c].label, "importance": float(pi.importances_mean[i]),
             "sd": float(pi.importances_std[i])} for i, c in enumerate(cols) if c in FEATURE_BY_NAME]
    return sorted(rows, key=lambda r: -r["importance"])


def numeric_support(data: pd.DataFrame, feats, lo: float = 0.01, hi: float = 0.99) -> dict[str, list[float]]:
    """1st-99th percentile of each numeric feature; inference clamps inputs to this range so
    a linear model never extrapolates (e.g. to 12-year-olds, who are rare in the data)."""
    from .schema import NUMERIC_FEATURES

    return {f: [float(data[f].quantile(lo)), float(data[f].quantile(hi))]
            for f in feats if f in NUMERIC_FEATURES and data[f].notna().any()}


def logreg_odds_ratios(est) -> list[dict] | None:
    """Odds ratio per feature if the final estimator (or its calibrated base) is a pipeline LR."""
    base = est
    if isinstance(est, CalibratedClassifierCV):
        base = est.calibrated_classifiers_[0].estimator
    if not hasattr(base, "named_steps") or "clf" not in base.named_steps:
        return None
    clf, prep = base.named_steps["clf"], base.named_steps["prep"]
    if not hasattr(clf, "coef_"):
        return None
    rows = []
    for name, coef in zip(prep.feature_names_out_, clf.coef_[0]):
        if name in FEATURE_BY_NAME:
            unit = "per 1 SD" if name in prep.numeric_ else "present vs absent"
            if name in prep.numeric_:
                unit += f" ({prep.std_[name]:.1f} {'years' if name == 'age' else 'kg/m2'})"
            rows.append({"feature": name, "label": FEATURE_BY_NAME[name].label, "coef": float(coef),
                         "odds_ratio": float(np.exp(coef)), "unit": unit})
    return rows


# --------------------------------------------------------------------------- main
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train the PCOS questionnaire screening model")
    ap.add_argument("--config", default=None, help="YAML config (default: config/default.yaml)")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="Override a config key, e.g. --set training.n_trials=10 (repeatable)")
    ap.add_argument("--models", nargs="+", default=None, help=f"Subset of {COMPLEXITY_ORDER}")
    ap.add_argument("--quick", action="store_true", help="Fast smoke run (2x2 CV, 3 trials, 50 bootstraps)")
    ap.add_argument("--skip-extras", action="store_true", help="Skip the ablation and clinical benchmark")
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    overrides = list(args.overrides)
    if args.quick:
        overrides += ["training.outer_folds=2", "training.outer_repeats=2", "training.inner_folds=2",
                      "training.n_trials=3", "training.bootstrap=50"]
    try:
        cfg = load_config(args.config, overrides)
        data = load_all(cfg)
    except (ConfigError, DataSourceError) as exc:
        log.error("%s", exc)
        return 2

    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)
    feats = tuple(cfg.get("features.model_features", FEATURE_NAMES))
    unknown = [f for f in feats if f not in FEATURE_BY_NAME]
    if unknown:
        log.error("Unknown model features in config: %s", unknown)
        return 2
    models = args.models or list(cfg["training"]["models"])
    bad = [m for m in models if m not in COMPLEXITY_ORDER]
    if bad:
        log.error("Unknown models: %s (choose from %s)", bad, COMPLEXITY_ORDER)
        return 2

    model_dir, report_dir, processed = cfg.path("model_dir"), cfg.path("report_dir"), cfg.path("processed_dir")
    for d in (model_dir, report_dir, processed):
        d.mkdir(parents=True, exist_ok=True)
    data.to_csv(processed / "harmonized.csv", index=False)
    log.info("Harmonised data: %d rows, %d sources, PCOS prevalence %.1f%%", len(data),
             data[SOURCE].nunique(), 100 * data[TARGET].mean())

    t0 = time.time()
    tasks = make_tasks(data, cfg)
    log.info("Running %d fold tasks (nested CV + LOSO) for models %s", len(tasks), models)
    results = execute(tasks, data, models, feats, cfg)
    agg = aggregate(results, tasks, data, models, cfg)
    report, oof = agg["models"], agg["oof"]
    for m, r in report.items():
        log.info("%-14s CV AUC %.3f +/- %.3f | pooled OOF AUC %.3f | LOSO mean AUC %.3f", m,
                 r["cv"]["fold_auc_mean"], r["cv"]["fold_auc_sd"], r["cv"]["pooled_oof"]["roc_auc"],
                 r["loso"]["mean_auc"])

    chosen, reason = select_model(report, cfg["training"].get("simplicity_margin", 0.01))
    log.info("Selected model: %s (%s)", chosen, reason)
    y = data[TARGET].to_numpy()
    comparisons = {m: paired_bootstrap_auc_diff(y, oof[chosen], oof[m], n=cfg["training"].get("bootstrap", 1000))
                   for m in oof if m != chosen}

    fold_params = {r["task"]["id"]: r["params"][chosen] for r in results
                   if r["task"]["kind"] == "cv" and chosen in r["params"]}
    cal = compare_calibration(data, tasks, chosen, fold_params, feats, cfg)
    method = cal["chosen"]
    log.info("Calibration: %s", {k: round(v["summary"]["brier"], 4) for k, v in cal["results"].items()})
    p_cal = cal["results"][method]["oof"]
    th = choose_thresholds(y, p_cal, cfg.get("thresholds.low_band_sensitivity", 0.9),
                           cfg.get("thresholds.high_band_specificity", 0.8))
    log.info("Risk bands: low < %.3f <= moderate < %.3f <= high", th["low"], th["high"])

    log.info("Fitting final model on all %d rows", len(data))
    final_params = tune_all([chosen], data, feats, cfg["training"]["n_trials"], cfg["training"]["inner_folds"], seed)
    final = _calibrated(build(chosen, final_params[chosen], seed, feats), method, data, seed)
    final.fit(data, y)

    ref = cfg.get("reference_source", "kerala")
    importance = global_importance(final, data, feats, seed)
    odds = logreg_odds_ratios(final)
    extras = {}
    if not args.skip_extras:
        log.info("Running feature-set ablation and Kerala clinical benchmark")
        extras["ablation"] = ablation(data, cfg)
        extras["clinical_benchmark"] = clinical_benchmark(cfg)

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metrics = {
        "version": __version__, "created": created, "selected_model": chosen, "selection_reason": reason,
        "features": list(feats), "reference_source": ref, "calibration": method,
        "calibration_comparison": {k: v["summary"] for k, v in cal["results"].items()},
        "final_oof": summarize(y, p_cal) | {"roc_auc_ci": bootstrap_ci(y, p_cal, n=cfg["training"].get("bootstrap", 1000))},
        "reliability": reliability_curve(y, p_cal),
        "thresholds": th, "auc_vs_chosen": comparisons, "models": report,
        "importance": importance, "odds_ratios": odds, "final_params": final_params[chosen],
        "data": {"n": int(len(data)), "prevalence": float(y.mean()),
                 "by_source": data.groupby(SOURCE)[TARGET].agg(["size", "mean"]).rename(
                     columns={"size": "n", "mean": "prevalence"}).to_dict(orient="index"),
                 "coverage": coverage_table(data).to_dict()},
        "runtime_seconds": round(time.time() - t0, 1),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "cpu_count": os.cpu_count()},
        "config": cfg.data, **extras,
    }
    baseline = {"age": float(data["age"].median()), "bmi": 22.0}
    artifact = {
        "version": __version__, "created": created, "model_name": chosen, "estimator": final,
        "features": list(feats), "reference_source": ref, "calibration": method, "thresholds": th,
        "baseline": baseline, "support": numeric_support(data, feats), "feature_cuts": {k: cfg.get(f"features.{k}") for k in
                                               ("overweight_bmi", "fast_food_per_week", "exercise_days_per_week")},
        "summary": {"roc_auc": metrics["final_oof"]["roc_auc"], "roc_auc_ci": metrics["final_oof"]["roc_auc_ci"],
                    "loso_mean_auc": report[chosen]["loso"]["mean_auc"],
                    "loso_per_source": {s: v["roc_auc"] for s, v in report[chosen]["loso"]["per_source"].items()},
                    "n": int(len(data)), "sources": int(data[SOURCE].nunique()),
                    "low_sensitivity": th["low_sensitivity"], "high_specificity": th["high_specificity"]},
    }
    model_path = model_dir / "pcos_screen_model.joblib"
    joblib.dump(artifact, model_path)
    (report_dir / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2), encoding="utf-8")
    pd.DataFrame({"source": data[SOURCE], "y": y, **{f"oof_{m}": p for m, p in oof.items()},
                  "oof_final_calibrated": p_cal}).to_csv(report_dir / "oof_predictions.csv", index=False)
    log.info("Saved model -> %s", model_path)
    log.info("Saved metrics -> %s", report_dir / "metrics.json")

    try:
        from .report import write_reports
        docs_dir = cfg.path("docs_dir") if cfg.get("paths.docs_dir") else cfg.root / "docs"
        write_reports(metrics, pd.read_csv(report_dir / "oof_predictions.csv"), report_dir, docs_dir)
    except Exception as exc:  # reporting must never lose a trained model
        log.warning("Report generation failed: %s", exc)
    log.info("Done in %.1fs", time.time() - t0)
    return 0


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


if __name__ == "__main__":
    sys.exit(main())
