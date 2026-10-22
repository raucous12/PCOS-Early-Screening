"""Load each raw PCOS data source and harmonise it onto the canonical schema.

Every loader returns a DataFrame with the columns ``FEATURE_NAMES + [TARGET, SOURCE]``.
A feature that a source never asked about is left as NaN. The models handle that
through imputation inside the pipeline (or natively, for tree models) together
with a per-source fixed effect.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import Config
from .schema import BOUNDS, FEATURE_NAMES, SOURCE, TARGET

log = logging.getLogger(__name__)


class DataSourceError(RuntimeError):
    """Raised when a raw data file is missing or does not look as expected."""


# --------------------------------------------------------------------------- helpers
def _norm(col: str) -> str:
    """Normalise a column header: lower-case alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", str(col).lower())


def _find(df: pd.DataFrame, *candidates: str, required: bool = True) -> str | None:
    """Find a column whose normalised name equals (or starts with) a candidate."""
    normed = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in normed:
            return normed[key]
    for cand in candidates:
        key = _norm(cand)
        for n, original in normed.items():
            if n.startswith(key):
                return original
    if required:
        raise DataSourceError(f"None of the columns {candidates} found; got {list(df.columns)[:12]}...")
    return None


_YES = {"yes", "y", "1", "true", "yes(detected positive)"}
_NO = {"no", "n", "0", "false", "no(detected negative)"}


def yes_no(series: pd.Series) -> pd.Series:
    """Map yes/no-like answers to 1.0/0.0; anything else (incl. 'Maybe', N/A) to NaN."""
    def conv(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v) if v in (0, 1) else np.nan
        s = str(v).strip().lower()
        if s in _YES:
            return 1.0
        if s in _NO:
            return 0.0
        return np.nan
    return series.map(conv).astype(float)


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _at_least(series: pd.Series, threshold: float) -> pd.Series:
    num = _to_num(series)
    return (num >= threshold).astype(float).where(num.notna())


def _bmi(weight_kg: pd.Series, height_cm: pd.Series) -> pd.Series:
    w, h = _to_num(weight_kg), _to_num(height_cm)
    bmi = w / (h / 100.0) ** 2
    return bmi.replace([np.inf, -np.inf], np.nan)


def _finalise(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Ensure canonical columns, enforce bounds, drop unlabeled rows and duplicates."""
    out = df.reindex(columns=list(FEATURE_NAMES) + [TARGET]).copy()
    for col, (lo, hi) in BOUNDS.items():
        bad = (out[col] < lo) | (out[col] > hi)
        if bad.any():
            log.warning("%s: %d %s values outside [%s, %s] set to NaN", source, int(bad.sum()), col, lo, hi)
        out.loc[bad, col] = np.nan
    out = out[out[TARGET].isin([0, 1])]
    out[TARGET] = out[TARGET].astype(int)
    out[SOURCE] = source
    out = out.astype({c: float for c in FEATURE_NAMES})
    if not np.isfinite(out[list(FEATURE_NAMES)].fillna(0).to_numpy()).all():
        raise DataSourceError(f"{source}: non-finite values after harmonisation")
    return out.reset_index(drop=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise DataSourceError(f"Missing data file: {path}")
    try:
        return pd.read_csv(path)
    except Exception as exc:  # malformed CSV, encoding problems, ...
        raise DataSourceError(f"Could not parse {path}: {exc}") from exc


# --------------------------------------------------------------------------- Kerala
KERALA_SYMPTOMS = {
    "weight_gain": ("Weight gain(Y/N)",),
    "excess_hair": ("hair growth(Y/N)",),
    "skin_darkening": ("Skin darkening (Y/N)",),
    "hair_loss": ("Hair loss(Y/N)",),
    "acne": ("Pimples(Y/N)",),
    "fast_food": ("Fast food (Y/N)",),
    "regular_exercise": ("Reg.Exercise(Y/N)",),
}


def read_kerala_raw(path: Path, sheet: str | None = None) -> pd.DataFrame:
    """Read the Kerala workbook, tolerating a renamed sheet."""
    if not path.is_file():
        raise DataSourceError(f"Missing data file: {path}")
    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception as exc:
        raise DataSourceError(f"Could not read workbook {path}: {exc}") from exc
    if sheet and sheet in sheets:
        df = sheets[sheet]
    else:
        df = next((d for d in sheets.values() if _find(d, "PCOS (Y/N)", required=False)), None)
        if df is None:
            raise DataSourceError(f"No sheet with a 'PCOS (Y/N)' column in {path}")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    return df.dropna(how="all")


def load_kerala(path: Path, sheet: str | None, cfg: Config) -> pd.DataFrame:
    raw = read_kerala_raw(path, sheet)
    out = pd.DataFrame(index=raw.index)
    out[TARGET] = _to_num(raw[_find(raw, "PCOS (Y/N)")])
    out["age"] = _to_num(raw[_find(raw, "Age (yrs)")])
    bmi = _bmi(raw[_find(raw, "Weight (Kg)")], raw[_find(raw, "Height(Cm)")])
    out["bmi"] = bmi.fillna(_to_num(raw[_find(raw, "BMI")]))
    out["overweight"] = (out["bmi"] >= cfg.get("features.overweight_bmi", 25.0)).astype(float).where(out["bmi"].notna())
    # Cycle(R/I) is coded 2 = regular, 4 = irregular; any other code is a data-entry error.
    cycle = _to_num(raw[_find(raw, "Cycle(R/I)")])
    out["irregular_cycle"] = cycle.map({2: 0.0, 4: 1.0})
    for feat, names in KERALA_SYMPTOMS.items():
        out[feat] = yes_no(raw[_find(raw, *names)])
    return _finalise(out, "kerala")


# Feature groups for the Kerala-only "clinical ceiling" benchmark.
KERALA_EXTRA_ANTHRO = {
    "waist_hip_ratio": ("Waist:Hip Ratio",),
    "menses_days": ("Cycle length(days)",),
    "marriage_years": ("Marraige Status (Yrs)",),
    "pregnant": ("Pregnant(Y/N)",),
    "abortions": ("No. of aborptions",),
    "pulse": ("Pulse rate(bpm)",),
    "bp_systolic": ("BP _Systolic (mmHg)",),
    "bp_diastolic": ("BP _Diastolic (mmHg)",),
}
KERALA_LABS_SCAN = {
    "fsh": ("FSH(mIU/mL)",), "lh": ("LH(mIU/mL)",), "amh": ("AMH(ng/mL)",),
    "tsh": ("TSH (mIU/L)",), "prl": ("PRL(ng/mL)",), "vit_d3": ("Vit D3 (ng/mL)",),
    "prg": ("PRG(ng/mL)",), "rbs": ("RBS(mg/dl)",), "hb": ("Hb(g/dl)",),
    "follicles_left": ("Follicle No. (L)",), "follicles_right": ("Follicle No. (R)",),
    "follicle_size_left": ("Avg. F size (L) (mm)",), "follicle_size_right": ("Avg. F size (R) (mm)",),
    "endometrium": ("Endometrium (mm)",),
}


def load_kerala_clinical(path: Path, sheet: str | None, cfg: Config) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Kerala data with questionnaire + anthropometric + lab/ultrasound columns.

    Returns the frame and a dict of nested feature sets used by the benchmark.
    """
    base = load_kerala(path, sheet, cfg)
    raw = read_kerala_raw(path, sheet).reset_index(drop=True)
    raw = raw[_to_num(raw[_find(raw, "PCOS (Y/N)")]).isin([0, 1])].reset_index(drop=True)
    if len(raw) != len(base):
        raise DataSourceError("Kerala clinical frame mis-aligned with harmonised frame")
    extra = {}
    for feat, names in {**KERALA_EXTRA_ANTHRO, **KERALA_LABS_SCAN}.items():
        col = _find(raw, *names, required=False)
        extra[feat] = _to_num(raw[col]) if col else np.nan
    df = pd.concat([base, pd.DataFrame(extra)], axis=1)
    df["lh_fsh_ratio"] = (df["lh"] / df["fsh"]).replace([np.inf, -np.inf], np.nan)
    df["follicles_total"] = df["follicles_left"] + df["follicles_right"]
    questionnaire = [f for f in FEATURE_NAMES if df[f].notna().any()]
    anthro = questionnaire + list(KERALA_EXTRA_ANTHRO)
    full = anthro + list(KERALA_LABS_SCAN) + ["lh_fsh_ratio", "follicles_total"]
    return df, {"questionnaire": questionnaire, "questionnaire+anthropometry": anthro,
                "questionnaire+anthropometry+labs+ultrasound": full}


# --------------------------------------------------------------------------- 2018 survey
def _survey_2018_excess_hair(df: pd.DataFrame) -> pd.Series:
    """Body-region hair scores are on a 1 (none) .. 5 (very heavy) scale."""
    face_chest = [_find(df, c) for c in ("Chin", "Cheeks", "Upper Lips", "Between breast")]
    thighs = _find(df, "Inner thighs")
    scores = df[face_chest + [thighs]].apply(_to_num)
    flag = (scores[face_chest] >= 4).any(axis=1) | (scores[thighs] >= 5)
    return flag.astype(float).where(scores.notna().any(axis=1))


def load_survey_in_2018(path: Path, cfg: Config) -> pd.DataFrame:
    raw = _read_csv(path).drop_duplicates()
    tested = yes_no(raw[_find(raw, "Have you been diagnosed(tested) with PCOS")]) == 1
    raw = raw[tested].copy()
    label = raw[_find(raw, "You had/have PCOS")].astype(str).str.strip().str.lower()
    out = pd.DataFrame(index=raw.index)
    out[TARGET] = np.select([label.str.startswith("yes"), label.str.startswith("no")], [1.0, 0.0], np.nan)
    out["age"] = _to_num(raw[_find(raw, "Select your age at which you had PCOS")])
    out["overweight"] = yes_no(raw[_find(raw, "Were you overweight")])
    out["weight_gain"] = yes_no(raw[_find(raw, "Have you experienced significant or rapid weight gain")])
    out["irregular_cycle"] = yes_no(raw[_find(raw, "Have you experienced irregular or missed periods")])
    out["conception_difficulty"] = yes_no(raw[_find(raw, "Difficulty in conceiving")])
    out["excess_hair"] = _survey_2018_excess_hair(raw)
    out["acne"] = yes_no(raw[_find(raw, "Acne or skin tags problem")])
    out["hair_loss"] = yes_no(raw[_find(raw, "Hair thinning or hair loss problems")])
    out["skin_darkening"] = yes_no(raw[_find(raw, "Dark patches on the back of your neck")])
    out["fatigue"] = yes_no(raw[_find(raw, "Feeling of tiredness all the time")])
    out["mood_swings"] = yes_no(raw[_find(raw, "Mood Swings more often than usual")])
    out["fast_food"] = _at_least(raw[_find(raw, "How many times a week did you eat outside")],
                                 cfg.get("features.fast_food_per_week", 3))
    out["regular_exercise"] = _at_least(raw[_find(raw, "How many days in a week did you exercise")],
                                        cfg.get("features.exercise_days_per_week", 3))
    return _finalise(out, "survey_in_2018")


# --------------------------------------------------------------------------- Mumbai survey
_MUMBAI_AGE = {"below 18": 16.0, "less than 18": 16.0, "under 18": 16.0, "18-25": 21.5, "26-30": 28.0,
               "31-35": 33.0, "36-40": 38.0, "40-45": 42.5, "41-45": 43.0, "above 45": 48.0}


def _bucket(series: pd.Series, mapping: dict[str, float], source: str) -> pd.Series:
    keys = series.astype(str).str.strip().str.lower()
    out = keys.map(mapping)
    unknown = sorted(set(keys[out.isna() & series.notna()]))
    if unknown:
        log.warning("%s: unmapped age buckets %s set to NaN", source, unknown)
    return out.astype(float)


def load_survey_mumbai(path: Path, cfg: Config) -> pd.DataFrame:
    raw = _read_csv(path).drop_duplicates()
    out = pd.DataFrame(index=raw.index)
    out[TARGET] = yes_no(raw[_find(raw, "PCOS")])
    out["age"] = _bucket(raw[_find(raw, "Age")], _MUMBAI_AGE, "survey_mumbai")
    out["overweight"] = yes_no(raw[_find(raw, "Overweight")])  # 'Maybe' -> NaN
    out["weight_gain"] = yes_no(raw[_find(raw, "loss weight gain / weight loss")])
    self_irregular = yes_no(raw[_find(raw, "irregular or missed periods")])
    cycle = raw[_find(raw, "Cycle Length")].astype(str).str.strip().str.lower()
    long_or_variable = cycle.isin(["keeps variating", "36+ days"])
    out["irregular_cycle"] = np.where(long_or_variable, 1.0, self_irregular)
    out["conception_difficulty"] = yes_no(raw[_find(raw, "Difficulty in conceiving")])

    grade = {"normal": 0, "moderate": 1, "excessive": 2}
    face_chest = [_find(raw, c) for c in ("Hair growth on Chin", "Hair growth on Cheeks",
                                          "Hair growth Between breasts", "Hair growth on Upper lips")]
    thighs = _find(raw, "Hair growth on Inner thighs")
    g = raw[face_chest + [thighs]].apply(lambda s: s.astype(str).str.strip().str.lower().map(grade))
    flag = (g[face_chest] >= 1).any(axis=1) | (g[thighs] >= 2)
    out["excess_hair"] = flag.astype(float).where(g.notna().any(axis=1))

    out["acne"] = yes_no(raw[_find(raw, "Acne or skin tags")])
    out["hair_loss"] = yes_no(raw[_find(raw, "Hair thinning or hair loss")])
    out["skin_darkening"] = yes_no(raw[_find(raw, "Dark patches")])
    out["fatigue"] = yes_no(raw[_find(raw, "always tired")])
    out["mood_swings"] = yes_no(raw[_find(raw, "more Mood Swings")])
    out["fast_food"] = _at_least(raw[_find(raw, "eat outside per week")], cfg.get("features.fast_food_per_week", 3))
    out["regular_exercise"] = _at_least(raw[_find(raw, "exercise per week")],
                                        cfg.get("features.exercise_days_per_week", 3))
    return _finalise(out, "survey_mumbai")


# --------------------------------------------------------------------------- lifestyle survey
_LIFESTYLE_AGE = {"less than 20": 18.0, "20-25": 22.5, "25-30": 27.5, "30-35": 32.5,
                  "35-44": 39.5, "45 and above": 48.0}


def load_survey_lifestyle(path: Path, cfg: Config) -> pd.DataFrame:
    raw = _read_csv(path).drop_duplicates()
    out = pd.DataFrame(index=raw.index)
    # Multi-checkbox artefacts such as "No, Yes, not diagnosed by a doctor" are ambiguous -> NaN.
    out[TARGET] = yes_no(raw[_find(raw, "PCOS")])
    out["age"] = _bucket(raw[_find(raw, "Age")], _LIFESTYLE_AGE, "survey_lifestyle")
    # Despite its name, Height_ft holds centimetres (e.g. 157.48 = 5 ft 2 in).
    height = _to_num(raw[_find(raw, "Height_ft")])
    height = height.where(height > 100, height * 30.48)
    out["bmi"] = _bmi(raw[_find(raw, "Weight_kg")], height)
    out["overweight"] = (out["bmi"] >= cfg.get("features.overweight_bmi", 25.0)).astype(float).where(out["bmi"].notna())
    out["irregular_cycle"] = yes_no(raw[_find(raw, "Menstrual_Irregularity")])
    out["excess_hair"] = yes_no(raw[_find(raw, "Hirsutism")])
    concept = raw[_find(raw, "Conception_Difficulty")].astype(str).str.strip().str.lower()
    out["conception_difficulty"] = np.select(
        [concept.isin(["yes, not diagnosed by a doctor", "yes, diagnosed by a doctor"]), concept.eq("no")],
        [1.0, 0.0], np.nan)
    out["fast_food"] = _at_least(raw[_find(raw, "Diet_Fried_Food")], cfg.get("features.fast_food_per_week", 3))
    freq = raw[_find(raw, "Exercise_Frequency")].astype(str).str.strip().str.lower()
    out["regular_exercise"] = freq.isin(["3-4 times a week", "daily"]).astype(float).where(raw[_find(raw, "Exercise_Frequency")].notna())
    return _finalise(out, "survey_lifestyle")


# --------------------------------------------------------------------------- orchestration
def load_all(cfg: Config) -> pd.DataFrame:
    """Load every enabled source and stack them into one harmonised frame."""
    raw_dir = cfg.path("raw_dir")
    loaders: dict[str, Callable[[dict], pd.DataFrame]] = {
        "kerala": lambda s: load_kerala(raw_dir / s["file"], s.get("sheet"), cfg),
        "survey_in_2018": lambda s: load_survey_in_2018(raw_dir / s["file"], cfg),
        "survey_mumbai": lambda s: load_survey_mumbai(raw_dir / s["file"], cfg),
        "survey_lifestyle": lambda s: load_survey_lifestyle(raw_dir / s["file"], cfg),
    }
    frames = []
    for name, spec in cfg["sources"].items():
        if not spec.get("enabled", True):
            continue
        if name not in loaders:
            raise DataSourceError(f"No loader for source '{name}'")
        df = loaders[name](spec)
        log.info("Loaded %-17s n=%4d  PCOS=%5.1f%%", name, len(df), 100 * df[TARGET].mean())
        frames.append(df)
    if not frames:
        raise DataSourceError("No data sources enabled")
    data = pd.concat(frames, ignore_index=True)
    if data[TARGET].nunique() < 2:
        raise DataSourceError("Need both PCOS and non-PCOS examples to train")
    return data


def coverage_table(data: pd.DataFrame) -> pd.DataFrame:
    """Share of non-missing answers per feature and source (for the data card)."""
    cov = data.groupby(SOURCE)[list(FEATURE_NAMES)].apply(lambda g: g.notna().mean()).T
    return cov.round(2)
