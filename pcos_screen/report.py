"""Static evaluation report: PNG figures + an auto-generated model card."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

log = logging.getLogger(__name__)

# Reference categorical palette (fixed order, validated for colour-vision deficiency).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
MODEL_ORDER = ["logreg", "random_forest", "hist_gb", "lightgbm", "xgboost", "ensemble"]
MODEL_LABEL = {"logreg": "Logistic regression", "random_forest": "Random forest", "hist_gb": "HistGB",
               "lightgbm": "LightGBM", "xgboost": "XGBoost", "ensemble": "Ensemble"}
SOURCE_LABEL = {"kerala": "Kerala clinics", "survey_in_2018": "India survey 2018",
                "survey_lifestyle": "Lifestyle survey", "survey_mumbai": "Mumbai survey"}


def _style(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, path: Path):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(oof: pd.DataFrame, models: list[str], chosen: str, path: Path):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for i, m in enumerate([m for m in MODEL_ORDER if m in models]):
        col = f"oof_{m}"
        if col not in oof:
            continue
        fpr, tpr, _ = roc_curve(oof["y"], oof[col])
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(oof["y"], oof[col])
        ax.plot(fpr, tpr, color=SERIES[i], lw=2.6 if m == chosen else 1.6,
                label=f"{MODEL_LABEL[m]}  AUC {auc:.3f}" + ("  (selected)" if m == chosen else ""))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls="--")
    _style(ax, "ROC curves: out-of-fold, nested 5x5 CV", "False-positive rate (1 - specificity)",
           "True-positive rate (sensitivity)")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")
    _save(fig, path)


def plot_calibration(reliability: list[dict], path: Path, method: str, model: str = ""):
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    x = [r["mean_predicted"] for r in reliability]
    y = [r["observed"] for r in reliability]
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls="--", label="Perfect calibration")
    how = "no recalibration needed" if method == "none" else f"{method} recalibration"
    ax.plot(x, y, color=SERIES[0], lw=2, marker="o", ms=7, markeredgecolor=SURFACE, markeredgewidth=2,
            label=f"{MODEL_LABEL.get(model, 'Selected model')} ({how})")
    _style(ax, "Calibration: out-of-fold, deciles", "Predicted probability", "Observed PCOS rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    _save(fig, path)


def plot_loso(models_report: dict, path: Path):
    models = [m for m in MODEL_ORDER if m in models_report]
    sources = sorted({s for m in models for s in models_report[m]["loso"]["per_source"]})
    if not sources:
        return
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    width = 0.8 / len(models)
    for i, m in enumerate(models):
        vals = [models_report[m]["loso"]["per_source"].get(s, {}).get("roc_auc", np.nan) for s in sources]
        xs = np.arange(len(sources)) + (i - (len(models) - 1) / 2) * width
        ax.bar(xs, vals, width=width * 0.9, color=SERIES[i], label=MODEL_LABEL[m])
    ax.axhline(0.5, color=AXIS, lw=1, ls="--")
    ax.set_xticks(np.arange(len(sources)), [SOURCE_LABEL.get(s, s) for s in sources])
    ax.set_ylim(0.4, 1.0)
    _style(ax, "External validation: AUC on an unseen source (leave-one-source-out)", "", "ROC-AUC")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, ncol=3, loc="upper left")
    _save(fig, path)


def plot_benchmark(rows: list[dict], path: Path):
    if not rows:
        return
    df = pd.DataFrame(rows)
    sets = list(dict.fromkeys(df["feature_set"]))
    models = list(dict.fromkeys(df["model"]))
    nice = {"questionnaire": "Questionnaire only",
            "questionnaire+anthropometry": "+ body measurements & BP",
            "questionnaire+anthropometry+labs+ultrasound": "+ blood tests & ultrasound"}
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    height = 0.8 / len(models)
    for i, m in enumerate(models):
        sub = df[df["model"] == m].set_index("feature_set").reindex(sets)
        ys = np.arange(len(sets)) + (i - (len(models) - 1) / 2) * height
        ax.barh(ys, sub["auc_mean"], height=height * 0.9, xerr=sub["auc_sd"], color=SERIES[i],
                error_kw={"ecolor": MUTED, "lw": 1, "capsize": 0}, label=MODEL_LABEL.get(m, m))
        for yv, v, sd in zip(ys, sub["auc_mean"], sub["auc_sd"]):
            ax.text(v + sd + 0.008, yv, f"{v:.3f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(np.arange(len(sets)), [nice.get(s, s) for s in sets])
    ax.set_xlim(0.5, 1.06)
    ax.invert_yaxis()
    _style(ax, "Kerala clinics: what doctors' tests add (5x5 CV AUC)", "ROC-AUC")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=len(models))
    _save(fig, path)


def plot_importance(metrics: dict, path: Path):
    odds = metrics.get("odds_ratios")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    if odds:
        # Lollipops from OR = 1 on a log axis: ratios are multiplicative, and 1 means "no effect".
        rows = sorted(odds, key=lambda r: r["odds_ratio"])
        labels = [r["label"] + ("  (per SD)" if r["unit"].startswith("per") else "") for r in rows]
        ors = np.array([r["odds_ratio"] for r in rows])
        ys = np.arange(len(rows))
        ax.hlines(ys, 1.0, ors, color=SERIES[0], lw=2)
        ax.plot(ors, ys, "o", color=SERIES[0], ms=8, markeredgecolor=SURFACE, markeredgewidth=2)
        for y, v in zip(ys, ors):
            ax.text(v * (1.06 if v >= 1 else 0.94), y, f"{v:.2f}", va="center", ha="left" if v >= 1 else "right",
                    fontsize=8.5, color=INK2)
        ax.set_yticks(ys, labels)
        ax.set_xscale("log")
        ax.set_xlim(0.6, max(6.0, ors.max() * 1.4))
        ax.set_xticks([0.75, 1, 1.5, 2, 3, 4, 5], ["0.75", "1", "1.5", "2", "3", "4", "5"])
        ax.minorticks_off()
        ax.axvline(1.0, color=AXIS, lw=1, ls="--")
        _style(ax, "Adjusted odds ratios: final logistic model", "Odds ratio, log scale (1 = no effect)")
    else:
        rows = sorted(metrics["importance"], key=lambda r: r["importance"])
        ax.barh([r["label"] for r in rows], [r["importance"] for r in rows], xerr=[r["sd"] for r in rows],
                color=SERIES[0], height=0.6, error_kw={"ecolor": MUTED, "lw": 1})
        _style(ax, "Permutation importance (drop in AUC)", "Mean decrease in ROC-AUC")
    _save(fig, path)


def _fmt(v, nd=3):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def model_card(metrics: dict) -> str:
    m = metrics["selected_model"]
    rep = metrics["models"]
    fo = metrics["final_oof"]
    th = metrics["thresholds"]
    lines = [
        "# Model card: PCOS questionnaire screening model",
        "",
        f"*Auto-generated by `python -m pcos_screen.train` on {metrics['created']} (v{metrics['version']}).*",
        "",
        "## Intended use",
        "Educational **screening** of PCOS/PCOD risk from self-reported answers, for teenage girls and adult "
        "women. It is **not a diagnostic device**. A high score means \"talk to a doctor\", and a low score "
        "does not rule PCOS out. In adolescents, and anyone within 8 years of their first period, the "
        "2023 guideline rule engine leads the result and the ML score is supporting information.",
        "",
        "## Data",
        f"{metrics['data']['n']} labelled people from {len(metrics['data']['by_source'])} independent sources "
        f"(overall PCOS prevalence {metrics['data']['prevalence']:.1%}).",
        "",
        "| Source | n | PCOS prevalence |",
        "|---|---|---|",
    ]
    for s, v in metrics["data"]["by_source"].items():
        lines.append(f"| {SOURCE_LABEL.get(s, s)} | {v['n']} | {v['prevalence']:.1%} |")
    lines += [
        "",
        f"**Model features ({len(metrics['features'])}):** " + ", ".join(f"`{f}`" for f in metrics["features"]),
        "",
        "## Selected model",
        f"**{MODEL_LABEL.get(m, m)}**, calibration: **{metrics['calibration']}**. Selection rule: "
        f"{metrics['selection_reason']}.",
        "",
        "| Metric (out-of-fold, calibrated) | Value |",
        "|---|---|",
        f"| ROC-AUC (95% CI) | {fo['roc_auc']:.3f} ({fo['roc_auc_ci'][0]:.3f}-{fo['roc_auc_ci'][1]:.3f}) |",
        f"| PR-AUC | {fo['pr_auc']:.3f} |",
        f"| Brier score | {fo['brier']:.3f} |",
        f"| Calibration slope / intercept | {fo['calibration_slope']:.2f} / {fo['calibration_intercept']:.2f} |",
        f"| Expected calibration error | {fo['ece']:.3f} |",
        f"| Mean leave-one-source-out AUC | {rep[m]['loso']['mean_auc']:.3f} |",
        "",
        "## Risk bands",
        f"* **Low**: p < {th['low']:.3f}. Sensitivity at this cut-off is {th['low_sensitivity']:.1%}, so "
        f"only about {1 - th['low_sensitivity']:.0%} of PCOS cases score below it.",
        f"* **Moderate**: {th['low']:.3f} <= p < {th['high']:.3f}.",
        f"* **High**: p >= {th['high']:.3f}. Specificity at this cut-off is {th['high_specificity']:.1%} "
        f"(sensitivity {th['high_sensitivity']:.1%}).",
        "",
        "## Model comparison",
        "| Model | CV AUC (mean +/- SD over folds) | Pooled OOF AUC | Brier | LOSO mean AUC | "
        + " | ".join(f"LOSO {SOURCE_LABEL.get(s, s)}" for s in sorted(metrics['data']['by_source'])) + " |",
        "|---|---|---|---|---|" + "---|" * len(metrics["data"]["by_source"]),
    ]
    for name in [x for x in MODEL_ORDER if x in rep]:
        r = rep[name]
        loso = [_fmt(r["loso"]["per_source"].get(s, {}).get("roc_auc")) for s in sorted(metrics["data"]["by_source"])]
        lines.append(f"| {MODEL_LABEL[name]}{' **(selected)**' if name == m else ''} | "
                     f"{r['cv']['fold_auc_mean']:.3f} +/- {r['cv']['fold_auc_sd']:.3f} | "
                     f"{r['cv']['pooled_oof']['roc_auc']:.3f} | {r['cv']['pooled_oof']['brier']:.3f} | "
                     f"{_fmt(r['loso']['mean_auc'])} | " + " | ".join(loso) + " |")
    if metrics.get("auc_vs_chosen"):
        lines += ["", "Paired bootstrap of the AUC difference (selected minus other):", "",
                  "| Other model | AUC diff | 95% CI | p |", "|---|---|---|---|"]
        for name, d in metrics["auc_vs_chosen"].items():
            lines.append(f"| {MODEL_LABEL.get(name, name)} | {d['diff']:+.3f} | {d['ci_low']:+.3f} to "
                         f"{d['ci_high']:+.3f} | {d['p_value']:.3f} |")
    if metrics.get("odds_ratios"):
        lines += ["", "## Adjusted odds ratios (final model)", "", "| Answer | Odds ratio | Unit |", "|---|---|---|"]
        for r in sorted(metrics["odds_ratios"], key=lambda r: -r["odds_ratio"]):
            lines.append(f"| {r['label']} | {r['odds_ratio']:.2f} | {r['unit']} |")
    if metrics.get("ablation"):
        lines += ["", "## Feature-set ablation (fixed hyper-parameters, mean LOSO AUC)", "",
                  "| Feature set | n | Logistic | LightGBM |", "|---|---|---|---|"]
        ab = pd.DataFrame(metrics["ablation"])
        for v, g in ab.groupby("variant", sort=False):
            vals = g.set_index("model")["loso_mean_auc"]
            lines.append(f"| {v} | {g['n_features'].iloc[0]} | {_fmt(vals.get('logreg'))} | {_fmt(vals.get('lightgbm'))} |")
    if metrics.get("clinical_benchmark"):
        lines += ["", "## Clinical ceiling (Kerala cohort only)", "",
                  "| Inputs | n features | Model | CV AUC |", "|---|---|---|---|"]
        for r in metrics["clinical_benchmark"]:
            lines.append(f"| {r['feature_set']} | {r['n_features']} | {MODEL_LABEL.get(r['model'], r['model'])} | "
                         f"{r['auc_mean']:.3f} +/- {r['auc_sd']:.3f} |")
        lines += ["", "Adding blood tests and ultrasound raises accuracy a lot. That is why a high questionnaire "
                      "score should lead to a clinical work-up, not a self-diagnosis."]
    lines += [
        "",
        "## Limitations",
        "* Survey labels are self-reported diagnoses. Women already diagnosed may notice and report symptoms "
        "more often (awareness bias).",
        "* Every source is from India. Performance in other populations is unknown. Leave-one-source-out "
        "validation is the best available proxy.",
        "* Few training records are from girls under 18. For them, the guideline rule engine is primary.",
        "* Hirsutism is self-assessed. Clinicians use the modified Ferriman-Gallwey score, which this "
        "questionnaire only approximates.",
        "* This model has not been prospectively validated and is not a medical device.",
        "",
        "## Figures",
        "![ROC](../reports/roc_curves.png)",
        "![Calibration](../reports/calibration.png)",
        "![LOSO](../reports/loso_auc.png)",
        "![Effects](../reports/feature_effects.png)",
        "![Benchmark](../reports/clinical_benchmark.png)",
    ]
    return "\n".join(lines) + "\n"


def write_reports(metrics: dict, oof: pd.DataFrame, report_dir: Path, docs_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9.5})
    models = list(metrics["models"])
    plot_roc(oof, models, metrics["selected_model"], report_dir / "roc_curves.png")
    plot_calibration(metrics["reliability"], report_dir / "calibration.png", metrics["calibration"],
                     metrics["selected_model"])
    plot_loso(metrics["models"], report_dir / "loso_auc.png")
    plot_importance(metrics, report_dir / "feature_effects.png")
    plot_benchmark(metrics.get("clinical_benchmark", []), report_dir / "clinical_benchmark.png")
    (docs_dir / "MODEL_CARD.md").write_text(model_card(metrics), encoding="utf-8")
    log.info("Wrote figures to %s and model card to %s", report_dir, docs_dir / "MODEL_CARD.md")


def main(argv=None) -> int:
    """Rebuild figures and the model card from an existing training run (no retraining)."""
    import argparse
    import json

    from .config import ConfigError, load_config

    ap = argparse.ArgumentParser(description="Regenerate report figures and MODEL_CARD.md")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        cfg = load_config(args.config, args.overrides)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2
    report_dir = cfg.path("report_dir")
    metrics_path, oof_path = report_dir / "metrics.json", report_dir / "oof_predictions.csv"
    if not metrics_path.is_file() or not oof_path.is_file():
        log.error("No training outputs in %s; run `python -m pcos_screen.train` first", report_dir)
        return 2
    docs_dir = cfg.path("docs_dir") if cfg.get("paths.docs_dir") else cfg.root / "docs"
    write_reports(json.loads(metrics_path.read_text(encoding="utf-8")), pd.read_csv(oof_path), report_dir, docs_dir)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
