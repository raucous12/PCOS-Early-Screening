# Data card

All data is public, individual-level and de-identified. Each source sits in its own folder under `data/raw/`. Nothing synthetic is used for training or evaluation.

| Source key | File | Origin | n used | Label | PCOS % |
|---|---|---|---|---|---|
| `kerala` | `kerala/PCOS_data_without_infertility.xlsx` (sheet `Full_new`) | Kaggle *Polycystic ovary syndrome (PCOS)* by P. Kottarathil: 10 hospitals in Kerala, India. Retrieved from the GitHub mirror [aryashah2k/Prediction-Of-PCOS-Using-Optimized-ML-Classifiers](https://github.com/aryashah2k/Prediction-Of-PCOS-Using-Optimized-ML-Classifiers); its SHA-256 `B663EC9F…F742` matches a second independent mirror. | 541 | Clinical diagnosis | 32.7 |
| `survey_in_2018` | `survey_in_2018/dataset1.csv` | Indian Google-Form PCOS survey (2018), from [manika1430/PCOS-detection-using-classifiation-techniques](https://github.com/manika1430/PCOS-detection-using-classifiation-techniques). Only respondents who had been tested are kept; exact duplicate rows are dropped. | 161 | Self-reported test result | 83.2 |
| `survey_mumbai` | `survey_mumbai/allData.csv` | Mumbai-area menstrual-health survey, from [Sathwikapaturi/PCOD-Prediction](https://github.com/Sathwikapaturi/PCOD-Prediction) | 261 | Self-reported diagnosis | 22.6 |
| `survey_lifestyle` | `survey_lifestyle/pcos_raw.csv` | 173-respondent lifestyle and diet PCOS survey, from [Caspal26/pcos-risk-prediction](https://github.com/Caspal26/pcos-risk-prediction). One ambiguous label (multi-checkbox) is dropped. | 172 | Self-reported diagnosis | 22.1 |

**Total: 1,135 people, 35.9% with PCOS.**

`survey_in_2018/kashish_variant.csv` is a 292-row export of the same 2018 survey, with identical timestamps. It is kept for provenance but **not loaded**, to avoid double-counting.

## Excluded sources (and why)
- **`dataset2` yes/no survey** (same repository as the 2018 survey): the label column uses undocumented codes ("mb n", …), so it cannot be labelled reliably.
- **Figshare "Symptoms and PCOS"**: an aggregate table only, with no individual rows.
- **Synthetic Kaggle sets** (for example "PCOS Prediction Dataset (Top 75 Countries)", 1,000-row generated sets): these would inflate metrics without adding real evidence.
- **Kerala variants with extra columns** (for example `pcos-detection.csv` with "bloated" and "stress" columns): those columns do not exist in the original release.

## Harmonisation (raw → canonical feature)
| Feature | Kerala | 2018 survey | Mumbai survey | Lifestyle survey |
|---|---|---|---|---|
| `age` | Age (yrs) | "age at which you had PCOS" | bucket midpoint | bucket midpoint |
| `bmi` | weight / height² | – | – | weight / height² (the "Height_ft" column holds cm) |
| `overweight` | BMI ≥ 25 | self-report | self-report ("Maybe" → missing) | BMI ≥ 25 |
| `irregular_cycle` | Cycle(R/I): 4 = irregular, 2 = regular (other codes → missing) | irregular/missed periods | self-report OR cycle "keeps varying" / "36+ days" | Menstrual_Irregularity |
| `weight_gain` | Weight gain (Y/N) | rapid weight gain (conditional question) | weight gain / hard to lose | – |
| `excess_hair` | hair growth (Y/N) | any face or chest region ≥ 4/5, or inner thighs = 5 | any face or chest region ≥ moderate, or thighs excessive | Hirsutism |
| `acne` | Pimples | acne or skin tags | acne or skin tags | – |
| `hair_loss` | Hair loss | hair thinning or loss | hair thinning or loss | – |
| `skin_darkening` | Skin darkening | dark patches on neck or underarms | dark patches | – |
| `conception_difficulty` | – | difficulty conceiving | difficulty conceiving ("N/A" → missing) | yes (doctor-diagnosed or not) vs no |
| `fast_food`* | Fast food (Y/N) | eat outside ≥ 3/week | eat outside ≥ 3/week | fried food ≥ 3 days/week |
| `regular_exercise`* | Reg. exercise | ≥ 3 days/week | ≥ 3 days/week | 3-4×/week or daily |
| `fatigue`*, `mood_swings`* | – | yes/no | yes/no | – |

\* Collected and harmonised, but **not used by the risk model**. The leave-one-source-out ablation showed they reduce cross-population accuracy, and exercise showed reverse causation (see the model card).

A "–" means the source never asked that question. Those values stay missing. The model imputes them, and a per-source fixed effect absorbs the difference.

## Known biases
- Every source is Indian; ethnicity-specific hirsutism norms and BMI cut-offs may differ elsewhere.
- Survey respondents self-selected through social media, and the 2018 survey is enriched for women with PCOS.
- Survey labels are self-reported diagnoses, and diagnosed women may notice symptoms more (awareness bias).
- Few respondents are under 18 (the lifestyle survey's "<20" group is the main teen signal).

## Licensing
These datasets were shared publicly by their authors for research and education. Check each original source's licence before redistributing the raw files.
