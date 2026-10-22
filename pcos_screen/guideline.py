"""Rule engine for the 2023 International Evidence-based PCOS Guideline.

The guideline defines irregular cycles by the time since the first period
(gynaecological age):

* < 1 year after menarche: irregular cycles are a normal part of puberty.
* 1 to < 3 years: cycles < 21 or > 45 days are irregular.
* >= 3 years: cycles < 21 or > 35 days are irregular.
* > 1 year: any single cycle > 90 days is irregular.
* Primary amenorrhoea: no period by age 15.

In adolescents, and in anyone within 8 years of menarche (when ultrasound is not
used), a diagnosis needs **both** irregular cycles and hyperandrogenism. Someone
with only one of them is "at risk" and should be followed up. Adults are assessed
on 2 of the 3 Rotterdam criteria. This questionnaire can only assess the two
clinical criteria; ovarian morphology and blood androgens need a clinician.

References: Teede et al., 2023 guideline; adolescent recommendations,
BMC Medicine 2025 (doi:10.1186/s12916-025-03901-w).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from .questionnaire import ANDROGEN_SENSITIVE, Answers

Status = Literal["met", "not_met", "indeterminate"]
ADOLESCENT_AGE = 18
ULTRASOUND_GYN_AGE = 8


@dataclass
class Criterion:
    status: Status
    reasons: list[str] = field(default_factory=list)


@dataclass
class GuidelineResult:
    pathway: Literal["adolescent", "adult"]
    gynecological_age: float | None
    ovulatory_dysfunction: Criterion
    hyperandrogenism: Criterion
    pattern: Literal["consistent", "at_risk", "possible", "not_suggestive", "indeterminate"]
    summary: str
    red_flags: list[str]
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def is_adolescent_pathway(a: Answers) -> bool:
    gyn = a.gynecological_age
    return a.age < ADOLESCENT_AGE or (gyn is not None and gyn < ULTRASOUND_GYN_AGE)


def assess_cycles(a: Answers) -> Criterion:
    """Irregular cycles (ovulatory dysfunction) per the guideline definition."""
    if not a.periods_started:
        if a.age >= 15:
            return Criterion("met", ["No period by age 15 (primary amenorrhoea)"])
        return Criterion("indeterminate", ["Periods have not started yet, which is common before age 15"])

    gyn = a.gynecological_age
    if gyn is not None and gyn < 1:
        return Criterion("indeterminate",
                         ["Less than 1 year since the first period, when irregular cycles are normal"])

    # With an unknown menarche age, adults get the >= 3-year rule and younger
    # girls get the more lenient 1-3-year rule, so a teen is not over-labelled.
    if gyn is None:
        long_limit = 35 if a.age >= ADOLESCENT_AGE else 45
    else:
        long_limit = 35 if gyn >= 3 else 45

    reasons: list[str] = []
    cl = a.cycle_length
    if cl == "lt21":
        reasons.append("Cycles are usually shorter than 21 days")
    elif cl == "36_45" and long_limit == 35:
        reasons.append("Cycles are usually longer than 35 days")
    elif cl in ("46_90", "gt90"):
        reasons.append(f"Cycles are usually longer than {long_limit} days")
    elif cl == "unpredictable":
        reasons.append("Cycle length is too unpredictable to say")
    if a.gap_over_90_days == "yes":
        reasons.append("Went more than 90 days without a period")

    if reasons:
        return Criterion("met", reasons)
    if cl == "unknown" and a.gap_over_90_days != "no":
        return Criterion("indeterminate", ["Not enough information about cycle length"])
    window = "21-35" if long_limit == 35 else "21-45"
    return Criterion("not_met", [f"Cycles fall in the expected {window}-day range for your stage"])


def assess_hyperandrogenism(a: Answers, adolescent: bool) -> Criterion:
    """Clinical signs of excess androgen (self-reported)."""
    reasons: list[str] = []
    if a.excess_hair:
        regions = [r.replace("_", " ") for r in ANDROGEN_SENSITIVE if a.hair_score(r) >= 2]
        if a.hair_score("thighs_back") >= 3:
            regions.append("thighs/back")
        reasons.append("Possible hirsutism: moderate-to-heavy hair on " + ", ".join(regions))
    if a.acne >= 2:
        reasons.append("Moderate-to-severe or persistent acne")
    if a.hair_loss and not adolescent:
        reasons.append("Scalp hair thinning (female-pattern hair loss)")
    return Criterion("met", reasons) if reasons else Criterion("not_met", ["No clear clinical signs reported"])


_SUMMARIES = {
    "consistent": ("Your answers match both clinical features doctors use to diagnose PCOS: irregular "
                   "cycles and signs of excess androgen. Please see a doctor (gynaecologist or "
                   "endocrinologist) to confirm and rule out conditions that look similar."),
    "at_risk": ("You have one of the two features needed for a PCOS diagnosis at your stage. The "
                "guideline calls this 'at risk': manage the symptoms and get a follow-up check, "
                "usually within a year or so."),
    "possible": ("You have one of the three PCOS features. A doctor can check the others: blood tests "
                 "for androgens, and an ultrasound or AMH test for adults. PCOS is possible but not "
                 "established."),
    "not_suggestive": "Your answers do not show the main clinical features of PCOS.",
    "indeterminate": ("It is too early, or there is not enough information, to judge your cycles. "
                      "Track your periods and repeat this check later."),
}


def assess(a: Answers) -> GuidelineResult:
    adolescent = is_adolescent_pathway(a)
    cycles = assess_cycles(a)
    ha = assess_hyperandrogenism(a, adolescent)

    if cycles.status == "met" and ha.status == "met":
        pattern = "consistent"
    elif cycles.status == "met" or ha.status == "met":
        pattern = "at_risk" if adolescent else "possible"
    elif cycles.status == "indeterminate":
        pattern = "indeterminate"
    else:
        pattern = "not_suggestive"

    red_flags: list[str] = []
    notes: list[str] = []
    if not a.periods_started and a.age >= 15:
        red_flags.append("No period by age 15: please see a doctor. There are several possible causes, "
                         "not only PCOS.")
    if a.rapid_virilization:
        red_flags.append("Sudden or rapidly worsening hair growth, a deepening voice or other "
                         "'masculinising' changes over a few months need prompt medical review to rule out "
                         "rare hormone-producing tumours.")
    stopped = a.cycle_length == "gt90" or a.gap_over_90_days == "yes"
    if stopped and a.possibly_pregnant == "yes":
        red_flags.append("Your periods have stopped and you might be pregnant: please take a pregnancy test.")
    if stopped:
        notes.append("Going 3 or more months without a period should be checked by a doctor. Long gaps "
                     "can affect the lining of the womb.")
    if a.hormonal_contraception:
        notes.append("Hormonal contraception (the pill, hormonal IUD, implant or injection) can hide "
                     "irregular cycles and acne. Your doctor may assess you after a break from it.")
    if a.thyroid_or_prolactin:
        notes.append("Thyroid problems and high prolactin can look like PCOS. Mention them to your doctor.")
    if a.family_history == "yes":
        notes.append("A mother or sister with PCOS or type 2 diabetes raises your likelihood of PCOS.")
    if adolescent:
        notes.append("At your stage, ultrasound and AMH are not used to diagnose PCOS. Diagnosis relies on "
                     "irregular cycles plus signs of excess androgen.")
    if a.age >= 45:
        notes.append("Around menopause, cycle changes are common for other reasons, so interpret this "
                     "result with care.")
    notes.append("Before diagnosing PCOS, doctors rule out conditions that look similar: thyroid disease, "
                 "high prolactin and non-classic congenital adrenal hyperplasia.")

    return GuidelineResult(
        pathway="adolescent" if adolescent else "adult",
        gynecological_age=a.gynecological_age,
        ovulatory_dysfunction=cycles,
        hyperandrogenism=ha,
        pattern=pattern,
        summary=_SUMMARIES[pattern],
        red_flags=red_flags,
        notes=notes,
    )
