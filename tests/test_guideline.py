"""Edge cases of the 2023 International Evidence-based Guideline rules."""
import pytest

from pcos_screen.guideline import assess, assess_cycles, is_adolescent_pathway
from pcos_screen.questionnaire import Answers


def A(**kw):
    base = dict(age=25, menarche_age=12, cycle_length="21_35", gap_over_90_days="no")
    base.update(kw)
    return Answers(**base)


@pytest.mark.parametrize("age,menarche,cycle,expected", [
    (13, 12.5, "unpredictable", "indeterminate"),   # < 1 year after menarche: normal transition
    (14, 12, "36_45", "not_met"),                   # 1-3 years: up to 45 days is fine
    (14, 12, "46_90", "met"),
    (14, 12, "lt21", "met"),
    (17, 12, "36_45", "met"),                       # >= 3 years: > 35 days is irregular
    (30, 12, "21_35", "not_met"),
    (30, 12, "gt90", "met"),
    (30, 12, "unpredictable", "met"),
])
def test_cycle_thresholds_by_gynecological_age(age, menarche, cycle, expected):
    assert assess_cycles(A(age=age, menarche_age=menarche, cycle_length=cycle)).status == expected


def test_gap_over_90_days_counts_after_first_year():
    assert assess_cycles(A(age=15, menarche_age=13, cycle_length="21_35", gap_over_90_days="yes")).status == "met"


def test_unknown_menarche_uses_lenient_rule_for_teens_and_strict_for_adults():
    assert assess_cycles(A(age=16, menarche_age=None, cycle_length="36_45")).status == "not_met"
    assert assess_cycles(A(age=22, menarche_age=None, cycle_length="36_45")).status == "met"


def test_unknown_cycle_is_indeterminate():
    assert assess_cycles(A(cycle_length="unknown", gap_over_90_days="unsure")).status == "indeterminate"
    assert assess_cycles(A(cycle_length="unknown", gap_over_90_days="no")).status == "not_met"


def test_primary_amenorrhoea():
    r = assess(Answers(age=15, periods_started=False))
    assert r.ovulatory_dysfunction.status == "met"
    assert any("age 15" in f for f in r.red_flags)
    young = assess(Answers(age=13, periods_started=False))
    assert young.ovulatory_dysfunction.status == "indeterminate" and not young.red_flags


@pytest.mark.parametrize("age,menarche,expected", [
    (16, 12, True), (20, 13, True), (21, 13, False), (25, 12, False), (17, None, True), (30, None, False),
])
def test_pathway(age, menarche, expected):
    assert is_adolescent_pathway(A(age=age, menarche_age=menarche)) is expected


def test_adolescent_needs_both_features():
    both = assess(A(age=16, menarche_age=12, cycle_length="46_90", hair={"chin": 2}))
    assert both.pattern == "consistent"
    one = assess(A(age=16, menarche_age=12, cycle_length="46_90"))
    assert one.pattern == "at_risk"
    none = assess(A(age=16, menarche_age=12))
    assert none.pattern == "not_suggestive"


def test_adult_one_feature_is_possible():
    assert assess(A(age=28, cycle_length="46_90")).pattern == "possible"
    assert assess(A(age=28, acne=2)).pattern == "possible"


def test_hair_loss_counts_for_adults_only():
    assert assess(A(age=30, hair_loss=True)).hyperandrogenism.status == "met"
    assert assess(A(age=15, menarche_age=12, hair_loss=True)).hyperandrogenism.status == "not_met"


def test_mild_acne_and_light_hair_do_not_count():
    r = assess(A(acne=1, hair={"upper_lip": 1, "arms": 3, "thighs_back": 2}))
    assert r.hyperandrogenism.status == "not_met"
    assert assess(A(hair={"thighs_back": 3})).hyperandrogenism.status == "met"


def test_red_flags_and_notes():
    r = assess(A(cycle_length="gt90", possibly_pregnant="yes", rapid_virilization=True,
                 hormonal_contraception=True, thyroid_or_prolactin=True, family_history="yes"))
    text = " ".join(r.red_flags)
    assert "pregnancy test" in text and "tumours" in text
    notes = " ".join(r.notes)
    assert "Hormonal contraception" in notes and "Thyroid" in notes and "mother or sister" in notes


def test_result_serialises():
    d = assess(A()).to_dict()
    assert {"pathway", "ovulatory_dysfunction", "hyperandrogenism", "pattern", "summary"} <= set(d)
