"""Canonical (harmonised) questionnaire feature schema.

Every data source is mapped onto these columns. Each one is something a person
can answer without a blood test or an ultrasound.
"""
from __future__ import annotations

from dataclasses import dataclass

TARGET = "pcos"
SOURCE = "source"
SOURCES = ("kerala", "survey_in_2018", "survey_mumbai", "survey_lifestyle")


@dataclass(frozen=True)
class Feature:
    name: str
    kind: str          # "binary" | "numeric"
    label: str         # plain-language name shown to users
    monotone: int = 0  # +1: can only raise risk, -1: can only lower it, 0: free


FEATURES: tuple[Feature, ...] = (
    Feature("age", "numeric", "Age"),
    Feature("bmi", "numeric", "Body-mass index (BMI)", +1),
    Feature("overweight", "binary", "BMI in the overweight range (25 or more)", +1),
    Feature("irregular_cycle", "binary", "Irregular or missed periods", +1),
    Feature("weight_gain", "binary", "Rapid weight gain / hard to lose weight", +1),
    Feature("excess_hair", "binary", "Excess hair on face, chest, belly or thighs", +1),
    Feature("acne", "binary", "Frequent acne / pimples", +1),
    Feature("hair_loss", "binary", "Scalp hair thinning or hair loss", +1),
    Feature("skin_darkening", "binary", "Dark, velvety skin patches (neck, armpits)", +1),
    Feature("fast_food", "binary", "Fast food or eating out 3+ times a week"),
    Feature("regular_exercise", "binary", "Exercise on 3+ days a week"),
    Feature("conception_difficulty", "binary", "Difficulty getting pregnant", +1),
    Feature("fatigue", "binary", "Feeling tired most of the time"),
    Feature("mood_swings", "binary", "Frequent mood swings"),
)

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)
FEATURE_BY_NAME = {f.name: f for f in FEATURES}
BINARY_FEATURES = tuple(f.name for f in FEATURES if f.kind == "binary")
NUMERIC_FEATURES = tuple(f.name for f in FEATURES if f.kind == "numeric")

# Physiological bounds used to null-out impossible values during cleaning.
BOUNDS = {"age": (10.0, 70.0), "bmi": (12.0, 60.0)}
