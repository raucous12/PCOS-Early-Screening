"""Combine the guideline rule engine and the ML model into one screening result.

Decision logic (documented in docs/MODEL_CARD.md and shown to users):

* Both clinical features (irregular cycles + androgen signs): **high**, whatever the model says.
* Adolescent pathway (under 18, or within 8 years of the first period): the
  guideline leads. One feature ("at risk") gives **moderate**. With no
  features, a high ML score gives **moderate**, otherwise **low**.
* Adults with one feature: at least **moderate**, raised to **high** if the ML band is high.
* Adults with no clinical feature: the ML band, capped at **moderate**, because
  PCOS then needs lab or ultrasound findings that a questionnaire cannot see.
* If no model is loaded, the guideline result alone is used.
"""
from __future__ import annotations

from typing import Any

from .guideline import GuidelineResult, assess
from .predict import PCOSPredictor
from .questionnaire import Answers

LEVEL_ORDER = {"low": 0, "moderate": 1, "high": 2}

HEADLINES = {
    "high": "Your answers show several signs linked to PCOS",
    "moderate": "Your answers show some signs worth checking",
    "low": "Your answers show few signs of PCOS",
}
ACTIONS = {
    "high": "Book an appointment with a gynaecologist or endocrinologist in the next few weeks. "
            "Take this summary with you.",
    "moderate": "Mention these symptoms at your next check-up, or see a doctor if they bother you. "
                "Track your periods for 3 months, since that helps the doctor a lot.",
    "low": "No action is needed now. If your periods become irregular or new symptoms appear, "
           "repeat this check or talk to a doctor.",
}


def _max(a: str, b: str) -> str:
    return a if LEVEL_ORDER[a] >= LEVEL_ORDER[b] else b


def _min(a: str, b: str) -> str:
    return a if LEVEL_ORDER[a] <= LEVEL_ORDER[b] else b


def combine(g: GuidelineResult, ml_band: str | None) -> tuple[str, str]:
    """Return (level, rationale)."""
    if g.pattern == "consistent":
        return "high", "Both clinical criteria from the 2023 international guideline are present."
    if ml_band is None:
        level = "moderate" if g.pattern in ("at_risk", "possible") else "low"
        return level, "Based on the guideline criteria only (the statistical model is unavailable)."
    if g.pathway == "adolescent":
        if g.pattern == "at_risk":
            return "moderate", "One of the two features needed at your stage is present ('at risk')."
        level = "moderate" if ml_band == "high" else "low"
        return level, ("At your stage, the guideline criteria lead. The statistical score "
                       + ("is high, so a check-up is worthwhile." if ml_band == "high" else "is supporting information."))
    if g.pattern == "possible":
        level = _max("moderate", ml_band)
        return level, "One clinical PCOS feature is present; the statistical score " + (
            "is also high." if ml_band == "high" else "adds supporting information.")
    level = _min(ml_band, "moderate")
    return level, ("Neither guideline feature is present, so the statistical score alone sets the level "
                   "(capped at moderate).")


def doctor_tests(g: GuidelineResult) -> list[str]:
    tests = ["Blood tests for androgens (total and free testosterone, SHBG)",
             "Thyroid (TSH) and prolactin, to rule out conditions that look like PCOS",
             "17-hydroxyprogesterone, to rule out non-classic congenital adrenal hyperplasia",
             "Metabolic screening: blood sugar (HbA1c or a glucose tolerance test) and cholesterol"]
    if g.pathway == "adult":
        tests.insert(1, "Pelvic ultrasound or an AMH blood test, if needed to complete the diagnosis")
    return tests


def lifestyle_tips(a: Answers, g: GuidelineResult) -> list[str]:
    tips = []
    teen = g.pathway == "adolescent" and a.age < 18
    if a.exercise_days_per_week is not None and a.exercise_days_per_week < 3:
        tips.append("Move more: aim for 150-300 minutes of moderate (or 75-150 minutes of vigorous) "
                    "activity a week, plus muscle-strengthening on 2 days." if not teen else
                    "Move more: aim for 60 minutes of moderate-to-vigorous activity every day, including "
                    "activities that strengthen muscles and bones 3 times a week.")
    if a.fast_food_per_week is not None and a.fast_food_per_week >= 3:
        tips.append("Cut down on fast food and sugary drinks. Base meals on vegetables, whole grains, "
                    "pulses and protein. This helps insulin resistance, which is common in PCOS.")
    bmi = a.bmi
    if bmi is not None and bmi >= 25 and not teen:
        tips.append("If you have extra weight, losing even 5-10% of it can bring back regular cycles "
                    "and improve other symptoms.")
    if a.fatigue or a.mood_swings:
        tips.append("Constant tiredness or mood swings are worth raising with a doctor. Anxiety and "
                    "depression are more common with PCOS, and support helps.")
    tips.append("Track your periods (start date and length) in a diary or an app. It is the most useful "
                "information you can bring to a doctor.")
    tips.append("Aim for regular sleep (7-9 hours for adults, 8-10 for teens), since poor sleep worsens "
                "insulin resistance.")
    return tips


def screen(answers: Answers, predictor: PCOSPredictor | None) -> dict[str, Any]:
    g = assess(answers)
    ml: dict[str, Any] | None = None
    ml_error = None
    if predictor is not None:
        try:
            ml = predictor.predict(answers)
        except Exception as exc:  # the guideline result is still useful on its own
            ml_error = f"{type(exc).__name__}: {exc}"
    level, rationale = combine(g, ml["band"] if ml else None)
    action = ACTIONS[level]
    if g.red_flags and level != "high":
        # The level is about PCOS likelihood; a red flag needs prompt care whatever the cause.
        action = "Please see a doctor soon about the warning sign below, which needs checking whatever its cause."
    return {
        "level": level,
        "headline": HEADLINES[level],
        "action": action,
        "rationale": rationale,
        "guideline": g.to_dict(),
        "model": ml,
        "model_error": ml_error,
        "bmi": answers.bmi,
        "doctor_tests": doctor_tests(g) if level != "low" else [],
        "lifestyle_tips": lifestyle_tips(answers, g),
        "disclaimer": ("This is an educational screening tool, not a diagnosis. Only a qualified doctor can "
                       "diagnose PCOS, usually with an examination and blood tests, and sometimes an ultrasound."),
    }
