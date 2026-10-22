import joblib
import pytest

from pcos_screen.predict import ModelNotAvailable, PCOSPredictor
from pcos_screen.questionnaire import Answers

HEALTHY = dict(age=24, height_cm=160, weight_kg=54, menarche_age=12, cycle_length="21_35", gap_over_90_days="no",
               hair_loss=False, skin_darkening=False, weight_gain=False, conception_difficulty="not_applicable")
SYMPTOMATIC = dict(HEALTHY, weight_kg=76, cycle_length="46_90", gap_over_90_days="yes", hair={"chin": 3, "upper_lip": 2},
                   acne=2, hair_loss=True, skin_darkening=True, weight_gain=True)


@pytest.fixture(params=["artifact", "tree_artifact"])
def any_predictor(request):
    return PCOSPredictor(request.getfixturevalue(request.param))


def test_probability_range_and_band(any_predictor):
    out = any_predictor.predict(Answers(**HEALTHY))
    assert 0 <= out["probability"] <= 1
    assert out["band"] in {"low", "moderate", "high"}
    assert out["thresholds"]["low"] < out["thresholds"]["high"]


def test_symptoms_raise_risk(any_predictor):
    lo = any_predictor.predict(Answers(**HEALTHY))
    hi = any_predictor.predict(Answers(**SYMPTOMATIC))
    assert hi["probability"] > lo["probability"] + 0.2
    assert hi["band"] == "high"
    assert lo["band"] in {"low", "moderate"}


def test_explanations_point_the_right_way(predictor):
    out = predictor.predict(Answers(**SYMPTOMATIC))
    names = {c["feature"]: c for c in out["contributions"]}
    for f in ("irregular_cycle", "excess_hair", "skin_darkening", "weight_gain"):
        assert names[f]["direction"] == "raises", f
    assert "body_weight" in names
    # Sorted by absolute effect.
    effects = [abs(c["log_odds"]) for c in out["contributions"]]
    assert effects == sorted(effects, reverse=True)


def test_lifestyle_answers_do_not_change_the_score(predictor):
    """Fast food, exercise, fatigue and mood shape advice only (not model features)."""
    a = predictor.predict(Answers(**HEALTHY))["probability"]
    b = predictor.predict(Answers(**HEALTHY, fast_food_per_week=10, exercise_days_per_week=0, fatigue=True,
                                  mood_swings=True, family_history="yes"))["probability"]
    assert a == b


def test_unknown_answers_are_reported_and_imputed(predictor):
    out = predictor.predict(Answers(age=30))
    assert 0 < out["probability"] < 1
    assert any("BMI" in u for u in out["unanswered"])
    # "Not applicable" is a real answer, not a missing one.
    assert not any("pregnant" in u for u in out["unanswered"])
    out = predictor.predict(Answers(age=30, conception_difficulty="not_applicable"))
    assert not any("pregnant" in u for u in out["unanswered"])


def test_age_is_not_listed_as_a_reason_and_reference_keeps_age(predictor):
    out = predictor.predict(Answers(age=15, **{k: v for k, v in SYMPTOMATIC.items() if k != "age"}))
    assert "age" not in {c["feature"] for c in out["contributions"]}
    assert out["reference_probability"] < out["probability"]


def test_one_symptom_at_a_time_is_monotone(predictor):
    base = predictor.predict(Answers(**HEALTHY))["probability"]
    for change in (dict(skin_darkening=True), dict(weight_gain=True), dict(hair_loss=True), dict(acne=1),
                   dict(hair={"chin": 2}), dict(cycle_length="46_90")):
        assert predictor.predict(Answers(**{**HEALTHY, **change}))["probability"] >= base, change


def test_missing_and_corrupt_artifacts(tmp_path, artifact):
    with pytest.raises(ModelNotAvailable):
        PCOSPredictor.load(tmp_path / "missing.joblib")
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a pickle")
    with pytest.raises(ModelNotAvailable):
        PCOSPredictor.load(bad)
    incomplete = tmp_path / "incomplete.joblib"
    joblib.dump({"estimator": artifact["estimator"]}, incomplete)
    with pytest.raises(ModelNotAvailable):
        PCOSPredictor.load(incomplete)


def test_numeric_inputs_are_clamped_to_training_support(artifact):
    p = PCOSPredictor({**artifact, "support": {"age": [16.0, 48.0], "bmi": [14.6, 35.2]}})
    young = p.predict(Answers(age=12, height_cm=150, weight_kg=95))
    at_floor = p.predict(Answers(age=16, height_cm=150, weight_kg=35.2 * 1.5 ** 2))
    assert young["features"]["age"] == 16.0 and young["features"]["bmi"] == 35.2
    assert set(young["outside_training_range"]) == {"Age", "Body-mass index (BMI)"}
    assert young["probability"] == at_floor["probability"]
    assert PCOSPredictor({**artifact, "support": {"age": [16.0, 48.0]}}).predict(Answers(age=30))["outside_training_range"] == []


def test_info(predictor):
    info = predictor.info()
    assert info["features"] and set(info["feature_labels"]) == set(info["features"])
