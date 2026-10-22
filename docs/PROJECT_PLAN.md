# PCOS / PCOD Early-Screening Project Plan

This plan follows the *create-project* framework: ideation (Stage 1), validation and form factor (Stage 2), systems and requirements (Stage 3), then execution (Stage 4: the code in this repository).

> **Scope note:** this tool is for **education and screening only**. It estimates how closely a person's answers match those of people diagnosed with PCOS, and it flags the guideline criteria a doctor would check. It does **not** diagnose PCOS. A diagnosis needs a clinician, and sometimes blood tests or an ultrasound.

---

## Stage 1: Ideation and brainstorming

### 1.1 Problem framing

| | |
|---|---|
| **Problem** | Polycystic ovary syndrome (PCOS, called "PCOD" in everyday Indian usage) affects an estimated 10–13% of women of reproductive age. Up to 70% of them are undiagnosed ([WHO fact sheet](https://www.who.int/news-room/fact-sheets/detail/polycystic-ovary-syndrome)). Diagnosis is often slow: in an international survey of 1,385 women, 33.6% waited more than 2 years and 47.1% saw 3 or more health professionals before being diagnosed ([Gibson-Helm et al., JCEM 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC6283441)). |
| **Why early matters** | PCOS is a lifelong metabolic condition (insulin resistance, type 2 diabetes, cardiovascular risk, infertility, endometrial risk, anxiety and depression). Lifestyle and medical management work best when started early, especially in adolescence. |
| **Core insight** | The main features of PCOS can be reported by the patient herself: irregular cycles, excess hair growth, acne, scalp hair loss, dark skin patches, weight gain. A well-designed questionnaire can triage risk *before* any lab test. |
| **Personas** | (1) **Teen (13–19):** irregular periods after menarche, acne, is unsure what is "normal"; a parent may help her answer. (2) **Young adult (20–35):** irregular cycles, weight or skin concerns, may be trying to conceive. (3) **Educator or health worker:** runs awareness camps in schools or colleges and needs a quick, explainable screener. |
| **Objectives** | (a) A non-invasive questionnaire with about 20 questions that takes under 4 minutes. (b) A calibrated risk estimate with a plain-language explanation. (c) Guideline-correct handling of adolescents. (d) Clear next steps and red flags. (e) Honest reporting of model accuracy. |

### 1.2 Three variations

| Variation | Description | Pros | Cons |
|---|---|---|---|
| **Lean / Minimal** | A static rule-based checklist that applies the 2023 International Guideline criteria (irregular cycles plus hyperandrogenism) in the browser. | No server, fully transparent, no data leaves the device | No statistical risk estimate, cannot learn from data, binary output |
| **Automated / Smart** *(chosen core)* | An ML model trained on harmonised real-world clinical and survey data, a guideline rule engine, and per-answer explanations behind a REST API and an adaptive web questionnaire. | Calibrated probabilities, explainable, adapts by age, reproducible training pipeline | Needs a model artifact and a backend; limited by small public datasets |
| **Enterprise / Robust** | Smart, plus authentication, consented storage of answers, clinician dashboard, FHIR export, continuous retraining with drift monitoring, and regulatory (SaMD) documentation. | Scales to hospital or public-health programmes | Heavy compliance (ABDM/HIPAA/GDPR) and clinical validation needed before use |

**Decision:** build the *Smart* variation, with *Enterprise* hooks already in place: config-driven paths, an optional SQLite store that is off by default and keeps no identifiers, a versioned model artifact with model and data cards, and a health-check endpoint.

---

## Stage 2: Industry validation and form factor

### 2.1 Data iteration and justification

Public, individual-level PCOS data is scarce. Searches of Kaggle, GitHub, Figshare, Zenodo, Mendeley and IEEE DataPort found the sources below. Each is kept in its own `data/raw/<source>/` folder and documented in [`DATA_CARD.md`](DATA_CARD.md).

| Source | n | Label | Type | Use |
|---|---|---|---|---|
| Kerala multi-hospital cohort (Kottarathil, Kaggle; GitHub mirrors with identical SHA-256) | 541 | Clinical PCOS diagnosis | Clinical + symptoms + labs + ultrasound | Training and evaluation; clinical-ceiling benchmark |
| India 2018 PCOS Google-Form survey | 186 tested | Self-reported diagnosis | Questionnaire (hirsutism by body region) | Training and evaluation |
| Mumbai menstrual-health survey | 267 | Self-reported diagnosis | Questionnaire (cycle length, hirsutism grades) | Training and evaluation |
| Lifestyle & diet PCOS survey (173 respondents) | 172 | Self-reported diagnosis | Questionnaire (BMI, family history) | Training and evaluation |
| Excluded: Figshare "Symptoms and PCOS" (aggregate table only), unlabeled or undocumented survey (`dataset2`), synthetic Kaggle sets | – | – | – | Not used. Synthetic data would inflate metrics without adding real evidence. |

**About 1,170 real people from 4 independent sources.** The headline generalisation metric is **leave-one-source-out (LOSO)** validation: train on 3 sources and test on the unseen 4th. That is the closest available proxy for "will this work on a new population?"

Clinical knowledge sources:
- 2023 International Evidence-based Guideline for PCOS, plus its adolescent recommendations ([BMC Medicine 2025](https://link.springer.com/article/10.1186/s12916-025-03901-w); [Peña et al. 2020](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7092491/)). Irregular cycles are defined by years since the first period: <1 yr is normal transition; 1–3 yr means <21 or >45 days; >3 yr means <21 or >35 days; >1 yr post-menarche, any cycle over 90 days counts; primary amenorrhoea means no period by age 15. In adolescents, diagnosis needs **both** irregular cycles and hyperandrogenism. Ultrasound and AMH are not used within 8 years of menarche, and a girl with only one feature is "at risk".
- A validated self-administered PCOS screening questionnaire ([PMC7785063](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7785063/)) supports questionnaire-based screening as a method.

### 2.2 Platform decision

| Criterion (weight) | Desktop | **Web app** | Android |
|---|---|---|---|
| Reach to teens and women on any device (30%) | 1 | **5** | 4 |
| Zero install and use in school or college camps (20%) | 1 | **5** | 2 |
| Seamless model updates (15%) | 2 | **5** | 3 |
| Privacy, no data at rest (15%) | 5 | **4** | 4 |
| Development and maintenance cost (10%) | 3 | **5** | 2 |
| Offline use (10%) | 5 | 2 | 5 |
| **Weighted score** | 2.35 | **4.55** | 3.35 |

**Decision: a responsive web application.** It works on phones, needs no install, and model updates ship from the server. The UI is mobile-first, so a PWA or Android wrapper can come later without a rewrite.

---

## Stage 3: Existing systems and technical requirements

### 3.1 Existing-system check
- **No existing codebase.** This is a greenfield project in `AIML/PCOS-Early-Screening`.
- **Integration points:** a JSON REST API (`/api/v1/predict`) lets the screener be embedded in other apps such as school health portals or tele-health intake. The answer schema uses plain fields that map onto FHIR `QuestionnaireResponse` for future EHR or ABDM integration.
- **Hardware:** 12-core CPU, 16 GB RAM, no GPU. Tabular models with about 1k rows train in seconds on CPU, so a GPU is not needed. Deep nets would overfit this data size, so gradient-boosted trees and penalised linear models are the right tools.

### 3.2 Requirements specification

**ML stack:** Python 3.13, pandas, NumPy, scikit-learn (pipelines, calibration, LR, RF, HistGB), LightGBM (monotone constraints), XGBoost, Optuna (nested hyper-parameter search), joblib (artifacts), matplotlib (reports), PyYAML plus argparse (config).

**Backend:** FastAPI with Pydantic v2 for strict input validation and OpenAPI docs, served by Uvicorn.

**Frontend:** dependency-free HTML, CSS and vanilla JS. It is a multi-step adaptive questionnaire with age-aware branching, accessible form controls and light/dark themes. Nothing in it phones home.

**Database:** **SQLite** (Python standard library) for an *optional*, anonymous log of answers and scores (`storage.enabled: false` by default). It needs no server, has ACID guarantees, and suits a single-node web app. A NoSQL store adds nothing at this scale.

**Quality:** pytest unit, integration and API tests, model-behaviour tests (monotonicity, invariance, guideline edge cases), plus a scripted manual QA pass in a real browser.

### 3.3 ML design

| Topic | Decision |
|---|---|
| **Target** | PCOS yes/no (binary). |
| **Harmonised features** (all answerable without tests) | age, BMI or overweight, irregular cycles, weight gain, excess hair growth (body-region based), acne, scalp hair loss, dark skin patches, fast food at least 3 times a week, exercise at least 3 days a week, difficulty conceiving, fatigue, mood swings |
| **Confounding control** | Prevalence differs a lot by source (22–76%), and some features are missing in some sources. A source indicator is included in training (fixed effect), so symptom effects are learned *within* each source and the model cannot use "which survey is this" shortcuts. At inference the source is set to the clinical reference cohort. |
| **Imbalance** | Class weights where needed; metrics are threshold-free (ROC-AUC, PR-AUC) plus Brier score and calibration. There is no SMOTE: oversampling small medical data creates fake patients. |
| **Loss** | Log-loss (binary cross-entropy) for all models, so the probabilities are proper. |
| **Models** | L2/elastic-net logistic regression, Random Forest, HistGradientBoosting, LightGBM with monotone constraints (a symptom can only raise risk), XGBoost, and a soft-voting ensemble. |
| **Validation** | (1) Repeated stratified 5×5 CV, stratified by source and label. (2) **LOSO** external validation. (3) Bootstrap 95% CIs. (4) Hyper-parameters tuned in *inner* folds only (nested), to avoid optimistic bias. |
| **Calibration** | Isotonic or Platt (sigmoid) calibration via CV; reliability curve and Brier score are reported. |
| **Thresholds** | Screening favours sensitivity. Risk bands are set from out-of-fold probabilities: *Low* < t₁ where sensitivity ≥ 90%, *High* ≥ t₂ where specificity ≥ 80%, and *Moderate* in between. |
| **Benchmark** | A Kerala "clinical ceiling" model that adds hormones, AMH and follicle counts, to show users what a doctor's tests add. |
| **Explainability** | Per-answer log-odds contributions (exact for LR, TreeSHAP `pred_contrib` for LightGBM), shown in plain words. |
| **Safety layer** | A guideline rule engine runs alongside the ML model and handles adolescents (gynaecological age), hormonal contraception masking cycles, and red flags (amenorrhoea over 90 days and pregnancy check, rapid virilisation, primary amenorrhoea). |

### 3.4 Edge cases and risks

| Risk | Mitigation |
|---|---|
| Small n, overfitting | Strong regularisation, monotone constraints, nested CV, simple models favoured unless a complex one wins significantly |
| Teens under-represented in training data | The guideline engine is primary for those less than 8 years post-menarche; the UI states the caveat |
| Self-reported labels and awareness bias in surveys | LOSO plus a clinical-label source; documented in the model card |
| Missing answers ("don't know") | Imputation inside the pipeline; the API accepts `null`; the UI allows "not sure" |
| Out-of-range inputs | Pydantic bounds (for example age 10–60, BMI computed and clipped); 422 errors with readable messages |
| Missing files or corrupt artifact | Clear CLI errors; API `/health` reports `model_loaded=false`, and the UI degrades to guideline-only mode |
| NaN or Inf in features | Validated at load; the training pipeline asserts finite matrices |
| Privacy (health data of minors) | No accounts, no identifiers, storage off by default, no third-party scripts |
