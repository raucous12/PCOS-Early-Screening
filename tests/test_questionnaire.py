from pcos_screen.questionnaire import Answers, to_features
from pcos_screen.schema import FEATURE_NAMES


def test_feature_vector_has_every_canonical_feature():
    f = to_features(Answers(age=22))
    assert set(f) == set(FEATURE_NAMES)


def test_bmi_and_overweight():
    a = Answers(age=30, height_cm=160, weight_kg=70)
    assert a.bmi == 27.3
    f = to_features(a)
    assert f["bmi"] == 27.3 and f["overweight"] == 1.0
    assert to_features(Answers(age=30))["bmi"] is None


def test_irregular_cycle_follows_guideline():
    assert to_features(Answers(age=30, menarche_age=12, cycle_length="46_90", gap_over_90_days="no"))["irregular_cycle"] == 1.0
    assert to_features(Answers(age=30, menarche_age=12, cycle_length="21_35", gap_over_90_days="no"))["irregular_cycle"] == 0.0
    # First year after menarche / periods not yet due: irregularity is normal -> "absent".
    assert to_features(Answers(age=12, menarche_age=11.5, cycle_length="unpredictable"))["irregular_cycle"] == 0.0
    assert to_features(Answers(age=13, periods_started=False))["irregular_cycle"] == 0.0
    # A genuinely unknown pattern stays unknown, so the model imputes it.
    assert to_features(Answers(age=25, menarche_age=12, cycle_length="unknown"))["irregular_cycle"] is None


def test_lifestyle_cut_offs_and_unknowns():
    f = to_features(Answers(age=20, fast_food_per_week=3, exercise_days_per_week=2))
    assert f["fast_food"] == 1.0 and f["regular_exercise"] == 0.0
    f = to_features(Answers(age=20))
    assert f["fast_food"] is None and f["hair_loss"] is None and f["conception_difficulty"] is None


def test_excess_hair_rule():
    assert Answers(age=20, hair={"chin": 2}).excess_hair
    assert not Answers(age=20, hair={"chin": 1, "upper_lip": 1}).excess_hair
    assert Answers(age=20, hair={"thighs_back": 3}).excess_hair
    assert Answers(age=20, hair={"chin": 2, "chest": 3}).hair_total == 5
