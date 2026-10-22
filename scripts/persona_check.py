"""Manual-QA helper: score reference personas against a running server.

Usage: python scripts/persona_check.py [--url http://127.0.0.1:8000]

Each persona has an expected level, taken from the guideline logic. The script
prints the actual result and exits non-zero on any mismatch, so it also works as
a post-deployment smoke check.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

PERSONAS = [
    ("Adult, classic PCOS picture", "high", dict(
        age=26, height_cm=160, weight_kg=74, menarche_age=12, cycle_length="46_90", gap_over_90_days="yes",
        hair={"chin": 2, "upper_lip": 2}, acne=2, hair_loss=True, skin_darkening=True, weight_gain=True)),
    ("Adult, regular cycles, no signs", "low", dict(
        age=29, height_cm=162, weight_kg=56, menarche_age=13, cycle_length="21_35", gap_over_90_days="no",
        hair_loss=False, skin_darkening=False, weight_gain=False, conception_difficulty="no")),
    ("Adult, only irregular cycles", "moderate", dict(
        age=31, height_cm=165, weight_kg=60, menarche_age=12, cycle_length="36_45", gap_over_90_days="no",
        hair_loss=False, skin_darkening=False, weight_gain=False)),
    ("Lean adult, hirsutism + irregular", "high", dict(
        age=23, height_cm=168, weight_kg=52, menarche_age=12, cycle_length="unpredictable", gap_over_90_days="no",
        hair={"chin": 3}, hair_loss=False, skin_darkening=False, weight_gain=False)),
    ("Teen, first year after menarche", "low", dict(
        age=12, menarche_age=11.5, cycle_length="unpredictable", gap_over_90_days="unsure")),
    ("Teen, 2 yrs post-menarche, 40-day cycles", "low", dict(
        age=15, menarche_age=13, cycle_length="36_45", gap_over_90_days="no", acne=1)),
    ("Teen, irregular + severe acne", "high", dict(
        age=16, menarche_age=12, cycle_length="46_90", gap_over_90_days="yes", acne=2)),
    ("Teen, irregular only (at risk)", "moderate", dict(
        age=17, menarche_age=12, cycle_length="46_90", gap_over_90_days="no")),
    ("15-year-old, no period yet", "moderate", dict(age=15, periods_started=False)),
    ("Adult, metabolic signs only (capped)", "moderate", dict(
        age=27, height_cm=158, weight_kg=82, menarche_age=12, cycle_length="21_35", gap_over_90_days="no",
        skin_darkening=True, weight_gain=True, conception_difficulty="yes")),
]


def post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    args = ap.parse_args(argv)
    endpoint = args.url.rstrip("/") + "/api/v1/screen"
    failures = 0
    print(f"{'Persona':42} {'expected':9} {'actual':9} {'model p':>8}  guideline pattern / pathway")
    for name, expected, payload in PERSONAS:
        try:
            r = post(endpoint, payload)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"Cannot reach {endpoint}: {exc}")
            return 2
        p = (r.get("model") or {}).get("probability")
        ok = r["level"] == expected
        failures += not ok
        print(f"{name:42} {expected:9} {r['level']:9} {('-' if p is None else f'{p:.3f}'):>8}  "
              f"{r['guideline']['pattern']} / {r['guideline']['pathway']}{'' if ok else '   <-- MISMATCH'}")
    print(f"\n{len(PERSONAS) - failures}/{len(PERSONAS)} personas as expected")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
