# Raw data (not included in the repository)

The four datasets used to train the model belong to their original publishers and are
**not redistributed here**. Download them, verified byte for byte against the copies the
model was trained on, with:

```bash
python scripts/download_data.py
```

| Folder | Source | Pinned file |
|---|---|---|
| `kerala/` | Kottarathil, *Polycystic ovary syndrome (PCOS)*, Kaggle; mirrored in [aryashah2k/Prediction-Of-PCOS-Using-Optimized-ML-Classifiers](https://github.com/aryashah2k/Prediction-Of-PCOS-Using-Optimized-ML-Classifiers) | `PCOS_data_without_infertility.xlsx` |
| `survey_in_2018/` | [manika1430/PCOS-detection-using-classifiation-techniques](https://github.com/manika1430/PCOS-detection-using-classifiation-techniques) | `dataset1.csv` |
| `survey_mumbai/` | [Sathwikapaturi/PCOD-Prediction](https://github.com/Sathwikapaturi/PCOD-Prediction) | `allData.csv` |
| `survey_lifestyle/` | [Caspal26/pcos-risk-prediction](https://github.com/Caspal26/pcos-risk-prediction) | `data/raw/pcos_raw.csv` |

Each file is pinned to a specific upstream commit and checked against a SHA-256 hash
(see `scripts/download_data.py`). If an upstream repository disappears, download the
Kerala dataset directly from Kaggle and place it in `kerala/` under the same file name.
See `docs/DATA_CARD.md` for how each source is harmonised.
