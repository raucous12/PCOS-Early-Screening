# Testing and validation

## 1. Automated tests (`python -m pytest -q`)
| File | What it checks |
|---|---|
| `test_data.py` | Every source loads into the canonical schema; binary features are 0/1/NaN; physiological bounds hold; expected sizes and prevalence; yes/no parsing; missing or malformed files raise `DataSourceError`; config overrides and errors |
| `test_guideline.py` | Cycle thresholds by years since first period (< 1, 1–3, ≥ 3 years); 90-day gaps; unknown menarche; primary amenorrhoea; adolescent vs adult pathway (including within 8 years of menarche); "both features" rule for teens; hair loss counted only for adults; red flags and notes |
| `test_questionnaire.py` | Answers → features: BMI, overweight, guideline-based irregularity, lifestyle cut-offs, the excess-hair rule |
| `test_models.py` | Preprocessor columns, imputation, scaling and unknown sources; every model fits; **monotone constraints hold** (turning a symptom on never lowers risk); the soft-vote average is exact |
| `test_metrics.py` | Calibration metrics on calibrated synthetic scores; NaN rejection; thresholds meet their sensitivity and specificity targets; bootstrap CI and paired test behave correctly |
| `test_predict.py` | Probability range and bands; symptoms raise risk (logistic and tree models); explanations point the right way; **lifestyle answers do not change the score**; unknown and not-applicable handling; clamping to the training range; missing, corrupt or incomplete artifacts |
| `test_screening.py` | The full guideline × model combination table; teen-specific advice; red-flag action override; the model failing gracefully |
| `test_api.py` | Health; happy path; minimal payload; consented storage; 9 invalid-input cases → readable 422s; security headers; static files; guideline-only mode when the model file is missing |
| `test_train_smoke.py` | The training CLI end to end on a tiny budget produces a working artifact, metrics, figures and model card; bad CLI input exits with code 2 |

## 2. Reference personas (`python scripts/persona_check.py`, server running)
These are 10 clinically motivated personas, each with an expected level from the guideline logic: a classic adult case, a healthy adult, irregular cycles only, a lean adult with hirsutism, a teen in the first year after menarche, a teen with 40-day cycles 2 years after menarche, a teen with irregular cycles and severe acne, a teen "at risk", a 15-year-old with no period yet, and an adult with metabolic signs only (capped). **All 10 matched.**

## 3. Manual browser QA (desktop dark, mobile 375 px light)
| Check | Result |
|---|---|
| Full 5-step flow, symptomatic adult | High, both criteria present, explanations ranked |
| Validation (empty age, height without weight) | Inline messages and highlighted fields |
| Unit toggles (ft/in, lb) and live BMI preview | 5′3″ + 72 kg → BMI 28.1 ✓ |
| Conditional questions (no periods → cycle block hidden; long gaps → pregnancy question shown) | ✓ |
| Teen scenarios (A: 40-day cycles 2 years post-menarche → low; B: at risk → moderate; C: no period at 15 → red flag) | ✓ |
| About page metrics, Learn page, print layout | ✓ |
| No horizontal overflow at 375 px on any view | ✓ |

### Bugs found by manual testing and fixed
1. **Mis-clicks during smooth scrolling.** After "Next", the page smooth-scrolled while the user might already be clicking, so an answer could land on the wrong option. The scroll is now instant.
2. Focus did not move to the first invalid field.
3. Adult activity advice also mentioned teen targets.
4. "Not applicable" answers were listed as "not answered".
5. A red flag (for example, no period by 15) appeared next to relaxed "next check-up" advice. Red flags now override the action text.
6. "Age lowers your score" was confusing next to "compared with someone of your age". Age is no longer listed as a reason.
7. A symptom-free 12-year-old got a 28% model score through linear extrapolation in age. Inputs are now clamped to the training data's 1st–99th percentile, and teens see a reliability note.
8. The hair-growth options wrapped awkwardly on phones; they are now a 4-column grid.
9. Report figures: value labels collided with error bars, and odds ratios were drawn as bars from 0. They are now a log-scale lollipop centred on 1.
10. `HEAD /` returned 405 (it broke uptime checks); it is now supported.
11. For a girl in her first year after menarche, "irregular cycles" was passed to the model as *unknown* and imputed with the population average, which inflated her score to 27%. By the guideline's definition irregularity is normal at that stage, so the feature is now "absent", and the same persona scores 16%.
12. A focus outline was drawn around each step heading after navigation (headings receive focus from script so screen readers announce the new step). Non-interactive headings no longer show the outline.
