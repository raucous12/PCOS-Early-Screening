import pytest

from pcos_screen.guideline import Criterion, GuidelineResult
from pcos_screen.questionnaire import Answers
from pcos_screen.screening import combine, screen


def G(pattern, pathway="adult"):
    c = Criterion("not_met")
    return GuidelineResult(pathway=pathway, gynecological_age=10, ovulatory_dysfunction=c, hyperandrogenism=c,
                           pattern=pattern, summary="", red_flags=[], notes=[])


@pytest.mark.parametrize("pattern,pathway,ml,expected", [
    ("consistent", "adult", "low", "high"),
    ("consistent", "adolescent", None, "high"),
    ("possible", "adult", "low", "moderate"),
    ("possible", "adult", "high", "high"),
    ("not_suggestive", "adult", "high", "moderate"),   # capped
    ("not_suggestive", "adult", "low", "low"),
    ("indeterminate", "adult", "moderate", "moderate"),
    ("at_risk", "adolescent", "low", "moderate"),
    ("not_suggestive", "adolescent", "high", "moderate"),
    ("not_suggestive", "adolescent", "moderate", "low"),
    ("possible", "adult", None, "moderate"),
    ("not_suggestive", "adult", None, "low"),
])
def test_combination_table(pattern, pathway, ml, expected):
    assert combine(G(pattern, pathway), ml)[0] == expected


def test_screen_end_to_end(predictor):
    r = screen(Answers(age=27, menarche_age=12, cycle_length="46_90", gap_over_90_days="yes",
                       hair={"chin": 3}, acne=2, height_cm=158, weight_kg=72, exercise_days_per_week=1,
                       fast_food_per_week=5, mood_swings=True), predictor)
    assert r["level"] == "high"
    assert r["model"]["probability"] > 0.5
    assert r["doctor_tests"] and any("ultrasound" in t.lower() for t in r["doctor_tests"])
    tips = " ".join(r["lifestyle_tips"])
    assert "150-300 minutes" in tips and "fast food" in tips and "mood swings" in tips


def test_teen_gets_no_ultrasound_and_teen_activity_advice(predictor):
    r = screen(Answers(age=15, menarche_age=12, cycle_length="46_90", gap_over_90_days="no",
                       exercise_days_per_week=1), predictor)
    assert r["guideline"]["pathway"] == "adolescent"
    assert r["level"] == "moderate"  # at risk
    assert not any("ultrasound" in t.lower() for t in r["doctor_tests"])
    assert any("60 minutes" in t for t in r["lifestyle_tips"])


def test_red_flag_overrides_relaxed_action(predictor):
    r = screen(Answers(age=15, periods_started=False), predictor)
    assert r["level"] != "high" and r["guideline"]["red_flags"]
    assert r["action"].startswith("Please see a doctor soon")


def test_adult_activity_tip_does_not_mention_teens(predictor):
    r = screen(Answers(age=30, exercise_days_per_week=0), predictor)
    tip = next(t for t in r["lifestyle_tips"] if t.startswith("Move more"))
    assert "150-300" in tip and "teen" not in tip.lower()


def test_screen_survives_model_failure():
    class Broken:
        def predict(self, _):
            raise RuntimeError("boom")
    r = screen(Answers(age=30, menarche_age=12, cycle_length="21_35", gap_over_90_days="no"), Broken())
    assert r["model"] is None and "boom" in r["model_error"] and r["level"] == "low"
