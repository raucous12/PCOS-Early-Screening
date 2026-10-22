# PCOS Early Check

A questionnaire-based **early-screening tool for PCOS/PCOD** in teenage girls and adult women. It pairs a rule engine built on the **2023 International Evidence-based PCOS Guideline** with a **machine-learning risk model** trained on 1,135 real people from 4 independent datasets, and serves both through an accessible, mobile-first web app.

> ⚠️ **Educational use only. This is not a diagnosis.** A high result means "see a doctor". A low result does not rule PCOS out.

## Results at a glance

| | Value |
|---|---|
| Training data | 1,135 people · 4 sources (1 clinical cohort from 10 Kerala hospitals + 3 questionnaire surveys) · 35.9% PCOS |
| Selected model | Elastic-net logistic regression (C = 1.30, L1 ratio 0.35) with per-source fixed effects, 10 self-reportable features |
| Discrimination (nested 5×5 CV) | **ROC-AUC 0.879** (95% CI 0.857–0.898), PR-AUC 0.816 |
| Generalisation (leave-one-source-out) | mean AUC **0.782**; **0.824** on the clinically diagnosed Kerala cohort when it is held out entirely |
| Calibration | slope 1.02, intercept 0.01, ECE 0.034, Brier 0.130 |
| Risk bands | **Low** < 0.198 (catches 90% of PCOS cases above it) · **High** ≥ 0.320 (80% specificity) |
| Clinical ceiling (Kerala) | questionnaire 0.883 → + blood tests and ultrasound **0.957**: the tool triages, and doctors confirm |
| Tests | 107 automated tests (unit, model-behaviour, API, end-to-end training) + 10 reference personas + manual browser QA |

Tree ensembles (XGBoost, LightGBM, random forest) fit the training sources 1–1.7 AUC points better, a statistically significant difference. They did **not** generalise better to unseen populations, and on the held-out clinical cohort logistic regression was best. Following the selection rule fixed before training (the simplest model within 0.01 of the best leave-one-source-out AUC), the interpretable model was chosen. See [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

Strongest answers (adjusted odds ratios): irregular periods **4.2×**, rapid weight gain **3.1×**, excess hair **2.9×**, dark velvety skin patches **2.8×**, difficulty conceiving **2.4×**, acne **2.0×**.

## Quick start (Windows PowerShell)

The raw datasets are not stored in this repository (they belong to their original publishers). The download script fetches them from pinned upstream commits and verifies each file's SHA-256 hash. The trained model in `models/` works without them; they are needed only for training and for the data-dependent tests.

```powershell
git clone https://github.com/raucous12/PCOS-Early-Screening.git
cd PCOS-Early-Screening
.\run.ps1 setup      # create the virtual env (outside OneDrive) and install requirements
python scripts/download_data.py   # fetch and verify the 4 raw datasets into data/raw/
.\run.ps1 train      # full nested-CV training, about 22 min on 12 CPU cores
.\run.ps1 test       # 107 automated tests, about 1 min
.\run.ps1 serve      # http://127.0.0.1:8000  (API docs at /docs)
```

On other platforms, use the same commands directly:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python scripts/download_data.py
python -m pcos_screen.train            # --quick for a 2-minute smoke run
python -m pytest -q
python -m app.main --port 8000
```

Useful options: `python -m pcos_screen.train --help`, `--set training.n_trials=50`, `--models logreg lightgbm`, `--config my.yaml`. Use `python -m pcos_screen.report` to rebuild figures and the model card without retraining. With the server running, `python scripts/persona_check.py` scores 10 reference personas.

## How it works

```
 Questionnaire (5 steps, ~20 questions)
          │
          ├──► Guideline rule engine ── irregular cycles by years since first period,
          │    (pcos_screen/guideline.py)  androgen signs, adolescent vs adult pathway,
          │                                red flags (no period by 15, virilisation, pregnancy)
          │
          ├──► ML risk model ──────────── calibrated probability + per-answer explanation
          │    (pcos_screen/predict.py)    (occlusion: each sign vs "absent", same age)
          │
          └──► Combination (pcos_screen/screening.py)
                 both clinical features            → HIGH
                 teen / < 8 yrs after menarche     → guideline leads ("at risk" → MODERATE)
                 adult, one feature                → ≥ MODERATE (HIGH if the model is high)
                 adult, no feature                 → model band, capped at MODERATE
```

### ML pipeline (`pcos_screen/train.py`)
1. **Harmonise** 4 heterogeneous datasets onto 14 canonical questionnaire features ([docs/DATA_CARD.md](docs/DATA_CARD.md)).
2. **Confounding control.** PCOS prevalence ranges from 22% to 83% across sources, so a *source fixed effect* is learned. Symptom effects are then estimated within each source, and inference uses the clinical reference source.
3. **Nested, repeated, stratified CV** (5×5 outer, 3-fold inner Optuna TPE search, log-loss objective) for 6 candidates: logistic regression, random forest, HistGB, LightGBM, XGBoost and a soft-vote ensemble. The boosted trees use **monotone constraints**, so a symptom can only raise risk.
4. **Leave-one-source-out** external validation, plus bootstrap CIs and paired bootstrap tests between models.
5. **Calibration choice** (none, Platt or isotonic) by out-of-fold Brier score; the logistic model needed none.
6. **Thresholds** from out-of-fold probabilities: sensitivity ≥ 90% at the low cut-off, specificity ≥ 80% at the high cut-off.
7. **Ablation**: fast food, exercise, fatigue and mood answers *lowered* cross-population AUC (0.761 → 0.783), and exercise showed reverse causation. They are still asked, but they shape advice only.
8. **Safety at inference**: numeric inputs are clamped to the 1st–99th percentile of the training data, so the model never extrapolates (for example to 12-year-olds), and answering "not sure" is imputed.

### Edge cases handled
- **Teens:** gynaecological-age rules (normal irregularity in year 1, 45-day limit in years 1–3), no ultrasound or AMH within 8 years of menarche, "at risk" wording, and the guideline leads.
- **Hormonal contraception** masking cycles, **thyroid/prolactin** mimics, **perimenopause** (age ≥ 45): explanatory notes.
- **Red flags** override relaxed advice: primary amenorrhoea, rapid virilisation, possible pregnancy with absent periods.
- **Missing model file:** the API reports `model_loaded: false` and serves guideline-only results.
- **Bad input:** Pydantic bounds and cross-field checks (age at first period after current age, implausible BMI, height without weight) return readable 422 messages.
- **Training robustness:** missing or corrupt data files raise clear errors (exit code 2); a failing model in one fold is logged and excluded; non-finite predictions are rejected; out-of-memory falls back to sequential execution; a reporting failure never loses a trained model.

## Project structure

```
PCOS-Early-Screening/
├── config/default.yaml        # all paths, sources, features and training settings (CLI-overridable)
├── data/raw/<source>/         # downloaded by scripts/download_data.py (not committed)
├── pcos_screen/               # ML package
│   ├── data.py                # loaders + harmonisation
│   ├── schema.py              # canonical features, monotonicity, bounds
│   ├── questionnaire.py       # Answers dataclass → model features
│   ├── guideline.py           # 2023 guideline rule engine
│   ├── models.py              # preprocessing, monotone boosting, model zoo + search spaces
│   ├── metrics.py             # AUC/PR/Brier/ECE, calibration slope, bootstrap, thresholds
│   ├── train.py               # nested CV, LOSO, selection, calibration, artifact
│   ├── report.py              # figures + auto-generated model card
│   ├── predict.py             # inference + explanations
│   ├── screening.py           # guideline × model combination, advice
│   └── storage.py             # optional anonymous SQLite log (off by default)
├── app/                       # FastAPI backend + static frontend (no third-party JS)
├── tests/                     # 107 pytest tests
├── scripts/                   # download_data.py (datasets), persona_check.py (manual QA)
├── models/                    # trained artifact (joblib)
├── reports/                   # metrics.json, OOF predictions, figures
└── docs/                      # PROJECT_PLAN (Stages 1-3), DATA_CARD, MODEL_CARD
```

## API

```bash
curl -X POST http://127.0.0.1:8000/api/v1/screen -H "Content-Type: application/json" -d '{"age": 24, "height_cm": 160, "weight_kg": 72, "menarche_age": 12, "cycle_length": "46_90", "gap_over_90_days": "yes", "hair": {"chin": 2}, "acne": 2, "weight_gain": true}'
```

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/screen` | Score a questionnaire. Returns the level, the guideline criteria, the model probability, explanations, tests a doctor may order, tips and red flags |
| `GET /api/v1/model` | Model metadata and validation metrics (shown on the "About" page) |
| `GET /api/v1/health` | Liveness, plus whether the model loaded |
| `GET /docs` | Interactive OpenAPI documentation |

## Privacy and safety
No accounts, cookies, third-party scripts or identifiers. Answers are processed in memory, and the optional SQLite log is off by default; when enabled it stores data only with explicit consent. A strict Content-Security-Policy and other security headers are set on every page.

## Limitations
All data is from India. Survey labels are self-reported. Only 19 training records are under 18. Hirsutism is self-rated rather than scored by a clinician. There has been no prospective validation, and this is not a medical device. See the model card for details.

## Licence and contributors
The code, documentation and trained model are released under the [MIT Licence](LICENSE). The third-party datasets are **not** covered and are downloaded from their original sources ([data/raw/README.md](data/raw/README.md)).

Maintained by [@raucous12](https://github.com/raucous12) and [@likhitha220712](https://github.com/likhitha220712).

## Sources
WHO PCOS fact sheet · Teede et al., 2023 International Evidence-based Guideline for PCOS · Adolescent PCOS recommendations, BMC Medicine 2025 (doi:10.1186/s12916-025-03901-w) · Gibson-Helm et al., JCEM 2017 (diagnostic delay) · Kottarathil, *PCOS dataset* (Kaggle) and the survey datasets listed in [docs/DATA_CARD.md](docs/DATA_CARD.md).
