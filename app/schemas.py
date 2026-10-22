"""Request validation for the screening API (Pydantic v2)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pcos_screen.questionnaire import Answers

def _score():
    return Field(0, ge=0, le=3, description="0 none, 1 a little, 2 moderate, 3 heavy")


class HairScores(BaseModel):
    model_config = ConfigDict(extra="forbid")
    upper_lip: int = _score()
    chin: int = _score()
    cheeks: int = _score()
    chest: int = _score()
    abdomen: int = _score()
    thighs_back: int = _score()


class ScreeningRequest(BaseModel):
    """One completed questionnaire. ``null`` means "not sure / prefer not to say"."""

    model_config = ConfigDict(extra="forbid")

    age: float = Field(..., ge=10, le=60, description="Age in years")
    height_cm: Optional[float] = Field(None, ge=100, le=220)
    weight_kg: Optional[float] = Field(None, ge=25, le=250)

    periods_started: bool = True
    menarche_age: Optional[float] = Field(None, ge=7, le=20, description="Age at first period")
    cycle_length: Literal["lt21", "21_35", "36_45", "46_90", "gt90", "unpredictable", "unknown"] = "unknown"
    gap_over_90_days: Literal["yes", "no", "unsure"] = "unsure"
    hormonal_contraception: bool = False

    hair: HairScores = Field(default_factory=HairScores)
    acne: int = Field(0, ge=0, le=2, description="0 none/rare, 1 mild but frequent, 2 moderate-severe/persistent")
    hair_loss: Optional[bool] = None
    skin_darkening: Optional[bool] = None
    weight_gain: Optional[bool] = None

    fast_food_per_week: Optional[float] = Field(None, ge=0, le=21)
    exercise_days_per_week: Optional[float] = Field(None, ge=0, le=7)
    fatigue: Optional[bool] = None
    mood_swings: Optional[bool] = None
    conception_difficulty: Literal["yes", "no", "not_applicable"] = "not_applicable"

    family_history: Literal["yes", "no", "unsure"] = "unsure"
    rapid_virilization: bool = False
    possibly_pregnant: Literal["yes", "no", "not_applicable"] = "not_applicable"
    thyroid_or_prolactin: bool = False

    store: bool = Field(False, description="Consent to keep these answers anonymously (only if enabled)")

    @model_validator(mode="after")
    def _consistency(self) -> "ScreeningRequest":
        if self.periods_started:
            if self.menarche_age is not None and self.menarche_age > self.age:
                raise ValueError("Age at first period cannot be greater than current age")
        else:
            if self.menarche_age is not None:
                raise ValueError("menarche_age must be empty when periods have not started")
            if self.cycle_length not in ("unknown",) or self.gap_over_90_days == "yes":
                raise ValueError("Cycle answers must be empty when periods have not started")
        if (self.height_cm is None) != (self.weight_kg is None):
            raise ValueError("Provide both height and weight, or neither")
        if self.height_cm and self.weight_kg:
            bmi = self.weight_kg / (self.height_cm / 100) ** 2
            if not 12 <= bmi <= 70:
                raise ValueError(f"Height and weight give an implausible BMI ({bmi:.1f}); please check them")
        return self

    def to_answers(self) -> Answers:
        data = self.model_dump(exclude={"store"})
        return Answers(**data)
