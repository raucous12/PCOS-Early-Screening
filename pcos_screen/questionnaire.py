"""The patient-facing questionnaire and its mapping onto model features.

``Answers`` is the single in-memory representation of a completed questionnaire.
The API validates requests into it, the guideline engine reads it, and
``to_features`` turns it into the canonical feature vector the ML model uses.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

CycleLength = Literal["lt21", "21_35", "36_45", "46_90", "gt90", "unpredictable", "unknown"]
YesNoUnsure = Literal["yes", "no", "unsure"]
YesNoNA = Literal["yes", "no", "not_applicable"]

HAIR_REGIONS = ("upper_lip", "chin", "cheeks", "chest", "abdomen", "thighs_back")
# Regions where moderate growth already suggests excess androgen (as used in the
# modified Ferriman-Gallwey score). Thighs/back count only when growth is heavy.
ANDROGEN_SENSITIVE = ("upper_lip", "chin", "cheeks", "chest", "abdomen")


@dataclass
class Answers:
    # About you
    age: float
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    # Periods
    periods_started: bool = True
    menarche_age: Optional[float] = None
    cycle_length: CycleLength = "unknown"
    gap_over_90_days: YesNoUnsure = "unsure"
    hormonal_contraception: bool = False
    # Skin & hair. Hair scores: 0 none, 1 a little, 2 moderate, 3 heavy.
    hair: dict[str, int] = field(default_factory=dict)
    acne: int = 0                       # 0 none/rare, 1 mild but frequent, 2 moderate-severe or persistent
    hair_loss: Optional[bool] = None
    skin_darkening: Optional[bool] = None
    weight_gain: Optional[bool] = None
    # Lifestyle and wellbeing
    fast_food_per_week: Optional[float] = None
    exercise_days_per_week: Optional[float] = None
    fatigue: Optional[bool] = None
    mood_swings: Optional[bool] = None
    conception_difficulty: YesNoNA = "not_applicable"
    # Health history (used for safety notes, not by the ML model)
    family_history: YesNoUnsure = "unsure"
    rapid_virilization: bool = False
    possibly_pregnant: YesNoNA = "not_applicable"
    thyroid_or_prolactin: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    # ------------------------------------------------------------------ derived values
    @property
    def bmi(self) -> Optional[float]:
        if not self.height_cm or not self.weight_kg:
            return None
        h = self.height_cm / 100.0
        return round(self.weight_kg / (h * h), 1)

    @property
    def gynecological_age(self) -> Optional[float]:
        """Years since the first period, or None when it cannot be computed."""
        if not self.periods_started or self.menarche_age is None:
            return None
        return max(0.0, float(self.age) - float(self.menarche_age))

    def hair_score(self, region: str) -> int:
        return int(self.hair.get(region, 0) or 0)

    @property
    def excess_hair(self) -> bool:
        return (any(self.hair_score(r) >= 2 for r in ANDROGEN_SENSITIVE)
                or self.hair_score("thighs_back") >= 3)

    @property
    def hair_total(self) -> int:
        return sum(self.hair_score(r) for r in HAIR_REGIONS)


def _flag(value: Optional[bool]) -> float | None:
    return None if value is None else float(bool(value))


def to_features(answers: Answers, overweight_bmi: float = 25.0, fast_food_cut: float = 3,
                exercise_cut: float = 3) -> dict[str, float | None]:
    """Map questionnaire answers onto the canonical model features.

    ``irregular_cycle`` uses the guideline definition, which depends on the years
    since the first period. ``None`` means "unknown" and is imputed by the model.
    """
    from .guideline import assess_cycles  # local import avoids a cycle at import time

    bmi = answers.bmi
    cycles = assess_cycles(answers)
    irregular = {"met": 1.0, "not_met": 0.0}.get(cycles.status)
    # Developmental stage (first year after menarche, or periods not yet due before 15):
    # irregularity is normal by the guideline's definition, so the feature is "absent",
    # not "unknown". Only a truly unknown cycle pattern is left for the model to impute.
    gyn = answers.gynecological_age
    if irregular is None and ((gyn is not None and gyn < 1) or (not answers.periods_started and answers.age < 15)):
        irregular = 0.0
    concept = {"yes": 1.0, "no": 0.0}.get(answers.conception_difficulty)
    return {
        "age": float(answers.age),
        "bmi": bmi,
        "overweight": None if bmi is None else float(bmi >= overweight_bmi),
        "irregular_cycle": irregular,
        "weight_gain": _flag(answers.weight_gain),
        "excess_hair": float(answers.excess_hair),
        "acne": float(answers.acne >= 1),
        "hair_loss": _flag(answers.hair_loss),
        "skin_darkening": _flag(answers.skin_darkening),
        "fast_food": None if answers.fast_food_per_week is None
        else float(answers.fast_food_per_week >= fast_food_cut),
        "regular_exercise": None if answers.exercise_days_per_week is None
        else float(answers.exercise_days_per_week >= exercise_cut),
        "conception_difficulty": concept,
        "fatigue": _flag(answers.fatigue),
        "mood_swings": _flag(answers.mood_swings),
    }
