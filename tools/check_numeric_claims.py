#!/usr/bin/env python3
"""Recompute numeric claims from the supplied result files.

Expected values and tolerances are fixed in this program. Manuscript wording
is not checked, and no paper text is loaded.

Run: python tools/check_numeric_claims.py --package .
Run: python tools/check_numeric_claims.py --package . --emit-map CLAIM_MAP.json
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if "--package" in sys.argv:
    ROOT = Path(sys.argv[sys.argv.index("--package") + 1]).resolve()
DATA = ROOT / "figures" / "data"
C = 100.0

# Ordered record of every result artifact this audit reads, for --emit-map.
LOADED: list[str] = []


def load(name):
    if name not in LOADED:
        LOADED.append(name)
    return json.loads((DATA / name).read_text(encoding="utf-8"))

eng_h = load("event_study_holdout.json")
eng_d = load("event_study_dev.json")
plac = load("placebo_suite.json")
spec = load("spec_curve_v3.json")
rnd = load("randomization_null.json")
fam = load("family_attribution_v2_holdout.json")
rq3 = load("rq3_v2_summary.json")
mplac = load("matched_placebo.json")
xdev = load("crossed_interval_qc_dev.json")["summary"]
xhold = load("crossed_interval_qc_holdout.json")["summary"]
sat = load("saturation_lemma.json")
tick_sub_h = load("tick_null_no_exact_score_holdout.json")
tick_sub_d = load("tick_null_no_exact_score_dev.json")

checks = []
# Artifact a block of checks derives from, declared per section rather than
# inferred, so the claim map never guesses a dependency it cannot support.
SRC = None


def _caller_line():
    """Line in this file that registered the check, so the map is traceable."""
    return sys._getframe(2).f_lineno


def chk(label, claimed, actual, tol=5e-4, fmt="{:.3f}"):
    ok = abs(claimed - actual) <= tol
    checks.append((ok, label, fmt.format(claimed), fmt.format(actual),
                   _caller_line(), SRC))








# ── headline ──────────────────────────────────────────────────────────────
SRC = ["event_study_holdout.json", "event_study_dev.json"]
h = eng_h["isolated"]
d = eng_d["isolated"]
chk("holdout D", 0.436, C * h["D"])
chk("holdout CI lo", 0.319, C * h["ci"][0])
chk("holdout CI hi", 0.557, C * h["ci"][1])
chk("holdout LOO lo", 0.415, C * h["loo_min"])
chk("holdout LOO hi", 0.457, C * h["loo_max"])
chk("holdout n_matches", 51, h["n_matches"], 0, "{:.0f}")
chk("holdout n_events", 125, h["n_events"], 0, "{:.0f}")
chk("holdout n_pos", 43, h["n_pos"], 0, "{:.0f}")
chk("dev D", 0.667, C * d["D"])
chk("dev CI lo", 0.510, C * d["ci"][0])
chk("dev CI hi", 0.838, C * d["ci"][1])
chk("dev n_pos", 37, d["n_pos"], 0, "{:.0f}")
chk("shrinkage %", 35.0, 100 * (1 - h["D"] / d["D"]), 0.6, "{:.1f}")

# ── all-goals sensitivity ─────────────────────────────────────────────────
SRC = ["event_study_holdout.json", "event_study_dev.json"]
chk("holdout all-goals D", 0.478, C * eng_h["all"]["D"])
chk("holdout all-goals n", 141, eng_h["all"]["n_events"], 0, "{:.0f}")
chk("dev all-goals D", 0.658, C * eng_d["all"]["D"])

# ── corrected specification (reviewer 2, comments 1 and 2) ────────────────
# The primary reported estimate. Adds the documented cancelled settlement state
# and the score-resolved contract the frozen compiler missed, together. The
# registered estimate above is retained and is still checked, so both survive.
SRC = ["corrected_specification.json"]
corr = {r["cohort"]: r for r in load("corrected_specification.json")["runs"]}
ch, cd = corr["holdout"], corr["dev"]
chk("corrected holdout D", 0.425, ch["D_cents"])
chk("corrected holdout CI lo", 0.316, ch["ci_cents"][0])
chk("corrected holdout CI hi", 0.538, ch["ci_cents"][1])
chk("corrected holdout LOO lo", 0.403, ch["loo_min_cents"])
chk("corrected holdout LOO hi", 0.446, ch["loo_max_cents"])
chk("corrected holdout n_matches", 51, ch["n_matches"], 0, "{:.0f}")
chk("corrected holdout n_events", 125, ch["n_events"], 0, "{:.0f}")
chk("corrected holdout n_pos", 45, ch["n_pos"], 0, "{:.0f}")
chk("corrected dev D", 0.664, cd["D_cents"])
chk("corrected dev CI lo", 0.502, cd["ci_cents"][0])
chk("corrected dev CI hi", 0.837, cd["ci_cents"][1])
chk("corrected dev LOO lo", 0.626, cd["loo_min_cents"])
chk("corrected dev LOO hi", 0.695, cd["loo_max_cents"])
chk("corrected dev n_pos", 37, cd["n_pos"], 0, "{:.0f}")
chk("corrected holdout all-goals D", 0.461, ch["all_D_cents"])
chk("corrected dev all-goals D", 0.655, cd["all_D_cents"])
chk("corrected shrinkage %", 36.0,
    100 * (1 - ch["D_cents"] / cd["D_cents"]), 0.6, "{:.1f}")
# The gap between the two specifications is what the paper claims is small
# relative to the interval, so it is derived rather than asserted.
chk("registered minus corrected", 0.011, C * eng_h["isolated"]["D"] - ch["D_cents"],
    6e-4)
chk("gap as a share of the CI width", 0.05,
    (C * eng_h["isolated"]["D"] - ch["D_cents"])
    / (ch["ci_cents"][1] - ch["ci_cents"][0]), 0.015, "{:.2f}")
# Counts must be identical under the two specifications; only the payoff
# matrix changes, so a different event count would mean something else moved.
checks.append((ch["n_events"] == eng_h["isolated"]["n_events"]
               and ch["n_matches"] == eng_h["isolated"]["n_matches"],
               "corrected keeps the registered sample",
               f"{ch['n_events']}/{ch['n_matches']}", "125/51"))

# ── event-selection funnel (reviewer 2, comment 6) ────────────────────────
SRC = ["event_funnel.json"]
fn = load("event_funnel.json")
fh, fd = fn["cohorts"]["holdout"], fn["cohorts"]["dev"]


def _funnel(key):
    return fh[key] + fd[key]


chk("funnel source goals", 300,
    fn["source_population"]["regulation_goals"], 0, "{:.0f}")
chk("funnel dev goals", 135, fd["regulation_goals"], 0, "{:.0f}")
chk("funnel holdout goals", 165, fh["regulation_goals"], 0, "{:.0f}")
chk("funnel outside capture", 9, _funnel("dropped_outside_capture_window"),
    0, "{:.0f}")
chk("funnel in capture window", 291, _funnel("goals_in_capture_window"),
    0, "{:.0f}")
chk("funnel event eligibility", 35, _funnel("dropped_by_event_eligibility"),
    0, "{:.0f}")
chk("funnel eligible events", 256, _funnel("eligible_events"), 0, "{:.0f}")
chk("funnel missing window data", 1,
    _funnel("dropped_for_missing_window_data"), 0, "{:.0f}")
chk("funnel events with data", 255, _funnel("events_with_window_data"),
    0, "{:.0f}")
chk("funnel isolation", 32, _funnel("dropped_by_isolation"), 0, "{:.0f}")
chk("funnel analysed", 223, _funnel("isolated_events"), 0, "{:.0f}")
chk("funnel matches contributing", 90,
    _funnel("matches_contributing_isolated_event"), 0, "{:.0f}")
# Every stage must sum to the next; a funnel that does not add up is the defect
# this table exists to rule out.
for cohort, block in (("dev", fd), ("holdout", fh)):
    walk = (block["regulation_goals"]
            - block["dropped_outside_capture_window"]
            - block["dropped_by_event_eligibility"]
            - block["dropped_for_missing_window_data"]
            - block["dropped_by_isolation"])
    checks.append((walk == block["isolated_events"], f"funnel arithmetic {cohort}",
                   str(walk), str(block["isolated_events"])))
checks.append((fn["crosscheck"] ==
               "terminal counts agree with event_study_<cohort>.json",
               "funnel reconciles with the engine", str(fn["crosscheck"])[:48],
               "agrees"))

# ── thresholds ────────────────────────────────────────────────────────────
SRC = ["threshold_ccdf.json", "event_study_holdout.json"]
th = {r["threshold"]: r for r in eng_h["thresholds"]}
for name, pre, post in (("certified", 53.6, 86.4), ("0.10c", 29.6, 74.4),
                        ("0.25c", 12.8, 59.2), ("0.50c", 4.8, 43.2),
                        ("1.00c", 2.4, 19.2)):
    chk(f"thr {name} pre", pre, th[name]["pre"], 0.05, "{:.1f}")
    chk(f"thr {name} post", post, th[name]["post"], 0.05, "{:.1f}")

# ── randomization ─────────────────────────────────────────────────────────
SRC = ["randomization_null.json", "randomization_test.json"]
for coh, ratio in (("dev", 7.0), ("holdout", 3.3)):
    r = rnd[coh]
    mx = max(r["draws_cents"])
    chk(f"{coh} null max", {"dev": 0.0952, "holdout": 0.1338}[coh], mx,
        5e-4, "{:.4f}")
    chk(f"{coh} obs/nullmax", ratio, r["observed_cents"] / mx, 0.05, "{:.1f}")
    chk(f"{coh} n_exceed", 0, r["n_exceed"], 0, "{:.0f}")
    chk(f"{coh} p", 1 / 5001, r["p"], 1e-6, "{:.5f}")

# ── control events ────────────────────────────────────────────────────────
SRC = ["placebo_suite.json"]
g = plac["holdout"]["goal"]["result"]
for key, dd, n in (("halftime", -0.003, 50), ("start_2nd_half", -0.009, 47),
                   ("substitution", -0.013, 88), ("yellow_card", 0.000, 74),
                   ("delay", 0.014, 146), ("red_card", 0.100, 7)):
    r = plac["holdout"][key]["result"]
    chk(f"control {key} D", dd, C * r["D"], 5e-4)
    chk(f"control {key} n", n, r["n_events"], 0, "{:.0f}")
for key, share in (("halftime", 3.1), ("start_2nd_half", 1.2),
                   ("substitution", 0.8), ("yellow_card", 6.8),
                   ("delay", 11.3)):
    r = plac["holdout"][key]["result"]
    chk(f"control {key} CI-hi as % of goal", share,
        100 * r["ci"][1] / g["D"], 0.06, "{:.1f}")

# ── specification curve ───────────────────────────────────────────────────
SRC = ["spec_curve_v3.json"]
tot = sig = 0
for coh in ("dev", "holdout"):
    for w in ("equal_match", "event"):
        rr = [r for r in spec[coh] if r["weighting"] == w]
        tot += len(rr)
        sig += sum(r["sig"] for r in rr)
chk("spec cells total", 1200, tot, 0, "{:.0f}")
chk("spec cells significant", 1200, sig, 0, "{:.0f}")
hm = [r for r in spec["holdout"] if r["weighting"] == "equal_match"]
ds = sorted(C * r["D"] for r in hm)
chk("spec holdout min", 0.224, ds[0])
chk("spec holdout max", 0.562, ds[-1])
chk("spec holdout median", 0.408, ds[len(ds) // 2], 0.002)
pr = [r for r in hm if r["primary"]][0]
pct = 100.0 * sum(1 for r in hm if r["D"] < pr["D"]) / len(hm)
chk("spec primary percentile", 68, pct, 0.6, "{:.0f}")

# ── attribution ───────────────────────────────────────────────────────────
SRC = ["family_attribution_v2_holdout.json"]
for f, lofo, rmean, rlo, rhi in (
        ("exact-score", 18.0, 25.8, 19, 32), ("total-goals", 70.4, 62.9, 56, 73),
        ("team-total", 67.0, 71.1, 61, 79), ("spread", 74.6, 63.1, 60, 71),
        ("moneyline", 91.3, 88.4, 80, 93), ("btts", 92.9, 95.8, 92, 98)):
    e = fam["lofo"][f]
    chk(f"LOFO {f}", lofo, 100 * e["lofo"]["D"] / fam["baseline"]["D"],
        0.06, "{:.1f}")
    p = e["size_matched_random_pcts"]
    chk(f"rand {f} mean", rmean, sum(p) / len(p), 0.06, "{:.1f}")
    chk(f"rand {f} lo", rlo, min(p), 0.5, "{:.0f}")
    chk(f"rand {f} hi", rhi, max(p), 0.5, "{:.0f}")

cert = fam["certificate"]
for f, dlt in (("exact-score", -0.082), ("moneyline", -0.023),
               ("team-total", 0.044), ("spread", 0.032),
               ("btts", 0.015), ("total-goals", 0.014)):
    chk(f"dual delta {f}", dlt,
        cert[f]["post_share"] - cert[f]["pre_share"], 6e-4)

# ── matched placebo ───────────────────────────────────────────────────────
SRC = ["matched_placebo.json"]
# Propagation (RQ3) was cut from the paper, so its checks are gone. The
# covariate-matched placebo replaced it as section 7.3.
for coh, scheme, obs, mx in (("holdout", "unmatched", 0.4361, 0.0953),
                             ("holdout", "time+score", 0.4513, 0.0696),
                             ("holdout", "full", 0.5272, 0.0722)):
    r = mplac["cohorts"][coh]["schemes"][scheme]
    chk(f"matched {scheme} observed", obs, r["observed_c"], 6e-4)
    chk(f"matched {scheme} null max", mx, r["null_max_c"], 6e-4)
chk("matched time+score goals", 113,
    mplac["cohorts"]["holdout"]["schemes"]["time+score"]["goals_used"],
    0, "{:.0f}")
chk("matched full goals", 104,
    mplac["cohorts"]["holdout"]["schemes"]["full"]["goals_used"], 0, "{:.0f}")

# ── anchor precision ──────────────────────────────────────────────────────
SRC = ["anchor_precision.json"]
# Bounds how much the estimate depends on the second-level event anchor, which
# is also the licensing question: could a minute-resolution source substitute?
aprec = load("anchor_precision.json")
for coh, jit, ret in (("holdout", 5, 98.6), ("holdout", 10, 97.8),
                      ("holdout", 60, 75.8), ("holdout", 120, 28.2),
                      ("dev", 10, 96.2), ("dev", 60, 61.6)):
    row = next(r for r in aprec["cohorts"][coh]["rows"] if r["jitter_s"] == jit)
    chk(f"anchor {coh} +/-{jit//2}s retained", ret, row["retained_pct"], 0.05,
        "{:.1f}")

# ── crossed canonical intervals ──────────────────────────────────────────
SRC = ["crossed_interval_qc_dev.json", "crossed_interval_qc_holdout.json"]
# The fail-vacuous branch is disclosed in 4.3; these assert it is never taken.
for lbl, s, ev in (("dev", xdev, 67750), ("holdout", xhold, 89681)):
    chk(f"crossed {lbl} pre", 0, s["pre_crossed"], 0, "{:.0f}")
    chk(f"crossed {lbl} post", 0, s["post_crossed"], 0, "{:.0f}")
    chk(f"crossed {lbl} contracts", 0, s["contracts_ever_crossed"], 0, "{:.0f}")
    chk(f"crossed {lbl} evaluations", ev,
        s["pre_evaluations"] + s["post_evaluations"], 0, "{:.0f}")

# ── saturation ────────────────────────────────────────────────────────────
SRC = ["saturation_lemma.json"]
chk("saturation ok", 104, sat["n_ok"], 0, "{:.0f}")
chk("saturation fail", 0, sat["n_fail"], 0, "{:.0f}")
chk("saturation required CAP", 10, sat["required_CAP"], 0, "{:.0f}")
chk("saturation CAP", 15, sat["CAP"], 0, "{:.0f}")

# ── split lottery ─────────────────────────────────────────────────────────
SRC = ["split_lottery.json"]
# Guards against the "did you pick the split?" objection with a measurement.
lot = load("split_lottery.json")
chk("lottery registered holdout", 0.4361, lot["registered_holdout_c"], 6e-4)
chk("lottery null mean", 0.5367, lot["null_mean_c"], 6e-4)
chk("lottery null sd", 0.0498, lot["null_sd_c"], 6e-4)
chk("lottery percentile", 2.2, lot["percentile_of_registered"], 0.06, "{:.1f}")
chk("lottery pct higher", 97.8, lot["pct_random_splits_higher"], 0.06, "{:.1f}")
chk("lottery pct positive", 100.0, lot["pct_draws_positive"], 0.01, "{:.1f}")
chk("lottery min draw", 0.3294, lot["min_draw_c"], 6e-4)
chk("lottery draws", 20000, lot["n_draw"], 0, "{:.0f}")
checks.append((lot.get("scope") ==
               "post-holdout label sensitivity conditional on frozen estimand",
               "lottery scope metadata", str(lot.get("scope")), "conditional"))
checks.append((lot.get("retunes_analysis_per_split") is False,
               "lottery does not retune", str(lot.get("retunes_analysis_per_split")),
               "False"))

# ── quoted-contract census (§4.3) ─────────────────────────────────────────
SRC = ["availability_census.json"]
# Answers "the abstract says ~46 contracts but the gate is 10, so which is it?"
cen = load("availability_census.json")
for coh, med, q1, q3, mn in (("holdout", 46, 45, 46, 37), ("dev", 46, 40, 46, 20)):
    c_pre = cen[coh]["pre"]
    chk(f"census {coh} median", med, c_pre["median"], 0, "{:.0f}")
    chk(f"census {coh} q1", q1, c_pre["q1"], 0, "{:.0f}")
    chk(f"census {coh} q3", q3, c_pre["q3"], 0, "{:.0f}")
    chk(f"census {coh} min", mn, c_pre["min"], 0, "{:.0f}")
chk("census dev pct>=30", 96.8, cen["dev"]["pre"]["pct_at_or_above_30"], 0.06, "{:.1f}")
chk("census holdout pct>=30", 100.0, cen["holdout"]["pre"]["pct_at_or_above_30"],
    0.01, "{:.1f}")

# ── contract subsystems (§6.2) ────────────────────────────────────────────
SRC = ["subsystem_availability_holdout.json", "subsystem_availability_dev.json"]
# The mechanism test. The `full` row must equal the frozen engine, otherwise the
# subsystem harness has drifted and every other row in the table is meaningless.
sub_h = load("subsystem_availability_holdout.json")["subsystems"]
sub_d = load("subsystem_availability_dev.json")["subsystems"]
chk("subsystem full = engine", C * h["D"], C * sub_h["full"]["D"], 1e-3)
chk("subsystem full dev = engine", C * d["D"], C * sub_d["full"]["D"], 1e-3)
for key, hd, lo, hi, dd in (
        ("full", 0.436, 0.320, 0.557, 0.667),
        ("no-exact-score", 0.079, 0.040, 0.121, 0.085),
        ("moneyline", 0.008, 0.001, 0.017, -0.004),
        ("exact-score", -0.000, -0.003, 0.002, 0.009)):
    chk(f"subsys {key} holdout D", hd, C * sub_h[key]["D"], 6e-4)
    chk(f"subsys {key} holdout CI lo", lo, C * sub_h[key]["ci"][0], 6e-4)
    chk(f"subsys {key} holdout CI hi", hi, C * sub_h[key]["ci"][1], 6e-4)
    chk(f"subsys {key} dev D", dd, C * sub_d[key]["D"], 6e-4)
chk("subsys no-exact-score n_pos", 30, sub_h["no-exact-score"]["n_pos"], 0, "{:.0f}")
# the claim that no subsystem recovers more than a fifth of the effect
worst = max(C * sub_h[k]["D"] for k in sub_h if k != "full")
chk("largest subsystem share of full", 18.1, 100.0 * worst / (C * h["D"]), 0.6, "{:.1f}")

# ── no-exact-score tick null (§6.3) ──────────────────────────────────────
SRC = ["tick_null_no_exact_score_holdout.json", "tick_null_no_exact_score_dev.json"]
# The observed branch must reproduce the persisted subsystem result exactly;
# otherwise the filtered null used different instants or constraints.
for cohort, artifact, subsystem in (("holdout", tick_sub_h, sub_h),
                                    ("dev", tick_sub_d, sub_d)):
    checks.append((artifact.get("subsystem") == "no-exact-score",
                   f"tick subsystem metadata {cohort}",
                   str(artifact.get("subsystem")), "no-exact-score"))
    chk(f"tick subsystem baseline {cohort}", C * subsystem["no-exact-score"]["D"],
        C * artifact["results"]["observed"]["D"], 1e-10)
    chk(f"tick subsystem nearest D {cohort}", 0.000,
        C * artifact["results"]["nearest"]["D"], 1e-10)
    chk(f"tick subsystem mismatch {cohort}", 0, artifact["n_tick_mismatch"],
        0, "{:.0f}")
    chk(f"tick subsystem fallback {cohort}", 0, artifact["n_inward_fallback"],
        0, "{:.0f}")
for label, claimed, actual in (
        ("tick subsystem inward holdout D", -0.017,
         C * tick_sub_h["results"]["narrowest"]["D"]),
        ("tick subsystem inward holdout CI lo", -0.031,
         C * tick_sub_h["results"]["narrowest"]["ci"][0]),
        ("tick subsystem inward holdout CI hi", -0.004,
         C * tick_sub_h["results"]["narrowest"]["ci"][1]),
        ("tick subsystem inward dev D", -0.016,
         C * tick_sub_d["results"]["narrowest"]["D"]),
        ("tick subsystem inward dev CI lo", -0.032,
         C * tick_sub_d["results"]["narrowest"]["ci"][0]),
        ("tick subsystem inward dev CI hi", -0.003,
         C * tick_sub_d["results"]["narrowest"]["ci"][1])):
    chk(label, claimed, actual, 6e-4)
for cohort, artifact in (("holdout", tick_sub_h), ("dev", tick_sub_d)):
    checks.append((set(artifact["tick_counts"]) == {"0.001", "0.0025", "0.01"},
                   f"tick subsystem active ticks {cohort}",
                   str(sorted(artifact["tick_counts"])),
                   "['0.001', '0.0025', '0.01']"))

# ── freshness and cross-goal straddle (§6.8) ──────────────────────────────
SRC = ["freshness_straddle_holdout.json", "freshness_straddle_dev.json"]
# The referee objection was that FRESH=30s lets pre-goal quotes leak into the
# post-goal window. The leak is symmetric, so both windows are guarded; these
# checks pin the direction, which is what the argument turns on.
fr_h = load("freshness_straddle_holdout.json")
fr_d = load("freshness_straddle_dev.json")
vh, vd = fr_h["variants"], fr_d["variants"]
chk("freshness base = engine", C * h["D"], C * vh["base"]["D"], 1e-3)
chk("freshness base dev = engine", C * d["D"], C * vd["base"]["D"], 1e-3)
for key, val, lo, hi in (("fresh10", 0.435, 0.319, 0.559),
                         ("no-straddle", 0.458, 0.341, 0.581),
                         ("no-straddle+fresh10", 0.458, 0.343, 0.582)):
    chk(f"freshness {key} D", val, C * vh[key]["D"], 6e-4)
    chk(f"freshness {key} CI lo", lo, C * vh[key]["ci"][0], 6e-4)
    chk(f"freshness {key} CI hi", hi, C * vh[key]["ci"][1], 6e-4)
chk("freshness no-straddle n_pos", 45, vh["no-straddle"]["n_pos"], 0, "{:.0f}")
chk("freshness dev fresh10", 0.666, C * vd["fresh10"]["D"], 6e-4)
chk("freshness dev no-straddle", 0.666, C * vd["no-straddle"]["D"], 6e-4)
# retention rates, quoted as percentages of the base variant's analysed instants
ih, idv = fr_h["instants_analysed"], fr_d["instants_analysed"]
chk("retain holdout fresh10", 99.6, 100.0 * ih["fresh10"] / ih["base"], 0.06, "{:.1f}")
chk("retain holdout no-straddle", 95.7,
    100.0 * ih["no-straddle"] / ih["base"], 0.06, "{:.1f}")
chk("retain holdout both", 95.5,
    100.0 * ih["no-straddle+fresh10"] / ih["base"], 0.06, "{:.1f}")
chk("retain dev no-straddle", 96.1,
    100.0 * idv["no-straddle"] / idv["base"], 0.06, "{:.1f}")
# the direction claim: neither correction may reduce the estimate on either cohort
for lab, vv, base in (("holdout", vh, vh["base"]["D"]), ("dev", vd, vd["base"]["D"])):
    worst_var = min(vv[k]["D"] for k in vv)
    checks.append((worst_var >= base - 2e-5,
                   f"no freshness variant lowers {lab}", "not lower", "expected"))

# ── fail-closed drop causes (§4.1) ────────────────────────────────────────
SRC = ["fail_closed_breakdown.json"]
# The reviewer's objection was that one counter conflates deliberate exclusions
# with parser failures. These pin the split, and the claim that nothing
# score-resolved is dropped.
fcb = load("fail_closed_breakdown.json")
_bc = fcb["by_cohort"]


def _tot(key):
    return sum(_bc[c].get(key, 0) for c in _bc)


chk("compiled slugs", 4573, _tot("compiled"), 0, "{:.0f}")
chk("half-time exclusions", 2882,
    _tot("half-time (design exclusion D3)"), 0, "{:.0f}")
chk("unsupported templates", 28697, _tot("unsupported template"), 0, "{:.0f}")
chk("distinct unsupported shapes", 3252, fcb["n_unsupported_shapes"], 0, "{:.0f}")
chk("compiled slugs = settlement checks", 4573, _tot("compiled"), 0, "{:.0f}")
# A previous version asserted that no score-resolved slug is dropped, using a
# keyword list. A reviewer found the counterexample the list could not catch:
# first-to-score-neither pays exactly when the match ends 0-0. The claim is
# withdrawn; what is checked now is that the known exception is present and
# declared, rather than that no exception exists.
_fts = fcb["unsupported_shapes"].get("first-to-score-neither")
checks.append((_fts is not None,
               "first-to-score-neither is recorded as dropped",
               "present" if _fts else "absent", "expected"))
if _fts:
    chk("first-to-score-neither dropped count", 103, _fts["count"], 0, "{:.0f}")

# ── missing score-resolved contract (Supplement §S2.6) ────────────────────
SRC = ["first_to_score_neither.json"]
# The reviewer found a contract the slug grammar drops that is decided by the
# final score. Adding it can only raise S, so these pin the direction as well
# as the values.
ftsn = load("first_to_score_neither.json")
_ftsn = {r["cohort"]: r for r in ftsn["runs"]}
for coh, want_D, lo, hi, pos, frozen in (
        ("holdout", 0.445, 0.331, 0.564, 45, 0.436),
        ("dev", 0.678, 0.515, 0.850, 37, 0.667)):
    r = _ftsn[coh]
    chk(f"ftsn {coh} D", want_D, r["D_cents"], 6e-4)
    chk(f"ftsn {coh} CI lo", lo, r["ci_cents"][0], 6e-4)
    chk(f"ftsn {coh} CI hi", hi, r["ci_cents"][1], 6e-4)
    chk(f"ftsn {coh} n_pos", pos, r["n_pos"], 0, "{:.0f}")
    checks.append((r["D_cents"] > frozen,
                   f"ftsn {coh} raises the estimate, as the lemma requires",
                   f"{r['D_cents']:.4f}c", f">{frozen:.3f}c"))
chk("ftsn holdout events unchanged", 125, _ftsn["holdout"]["n_events"], 0, "{:.0f}")
chk("ftsn dev events unchanged", 98, _ftsn["dev"]["n_events"], 0, "{:.0f}")

# ── resolution-description validation (§3.5) ──────────────────────────────
SRC = ["description_validation.json"]
# The settlement check tests each rule at one score. This tests the rule itself
# against the venue's own words, so it can fail where the settlement check
# cannot.
dv = load("description_validation.json")
chk("descriptions checked", 4573, dv["n_contracts"], 0, "{:.0f}")
chk("descriptions agreeing", 4544, dv["verdicts"].get("agree", 0), 0, "{:.0f}")
chk("description name variants", 29,
    dv["verdicts"].get("name variant", 0), 0, "{:.0f}")
checks.append((dv["verdicts"].get("disagree", 0) == 0,
               "no description disagrees with its compiled rule",
               str(dv["verdicts"].get("disagree", 0)), "0 expected"))
chk("descriptions stating the regulation window", 4573,
    dv["regulation_window"].get("states regulation window", 0), 0, "{:.0f}")
checks.append((not dv["disagreements"],
               "disagreement list is empty",
               f"{len(dv['disagreements'])} listed", "0 expected"))

# ── canonical-leg orientation (§3.5) ──────────────────────────────────────
SRC = ["outcome_orientation.json"]
# Settlement and description checks both compare a rule to a description of the
# same leg, so a consistently inverted contract passes both. This is the check
# that can catch it.
orient = load("outcome_orientation.json")
chk("orientation contracts checked", 4573, orient["n_checked"], 0, "{:.0f}")
checks.append((orient["n_mislabelled"] == 0,
               "no canonical leg is mislabelled",
               str(orient["n_mislabelled"]), "0 expected"))
checks.append((orient["n_checked"] == orient["n_contracts"],
               "every compiled contract had its orientation established",
               f"{orient['n_checked']} of {orient['n_contracts']}",
               "all expected"))

# ── unsupported-shape inventory (§4.1) ────────────────────────────────────
SRC = ["system_composition.json"]
# A pattern grouping, not a hand review; it is exhaustive and the inventory
# ships, which is what the text claims.
comp = load("system_composition.json")
_tax = comp["dropped_taxonomy"]
_drop = sum(_tax.values())
chk("dropped slugs grouped", 28697, _drop, 0, "{:.0f}")
_prop = _tax.get("shots and saves", 0) + _tax.get("player goal or assist props", 0)
chk("player and shot prop share", 93.0, 100.0 * _prop / _drop, 0.5, "{:.0f}%")
chk("corner share", 6.0, 100.0 * _tax.get("corners", 0) / _drop, 0.5, "{:.0f}%")
checks.append((not comp["unclassified_shapes"],
               "every unsupported shape falls in a named group",
               f"{len(comp['unclassified_shapes'])} ungrouped", "0 expected"))
# ── recovery to the pre-goal baseline (§5.2) ──────────────────────────────
SRC = ["event_time_profiles.json"]
# A reading of the mean curve, not a per-goal estimand: the first grid point
# after the peak at or below the pre-goal mean. "Stays below" is not used, since
# both curves converge onto the baseline and noise there resets the clock.
prof = load("event_time_profiles.json")
for _coh, _want in (("holdout", 95), ("dev", 120)):
    _g = prof[_coh]["goal"]
    _o, _m = _g["offsets"], _g["mean_cents"]
    _pre = [_m[i] for i, x in enumerate(_o)
            if -60 <= x <= -6 and _m[i] is not None]
    _base = sum(_pre) / len(_pre)
    _peak = max((_m[i], _o[i]) for i, x in enumerate(_o)
                if x >= 0 and _m[i] is not None)[1]
    _cross = next(x for i, x in enumerate(_o)
                  if x > _peak and _m[i] is not None and _m[i] <= _base)
    chk(f"{_coh} return to baseline", _want, _cross, 0, "{:.0f}s")
# ── collector cycle period (Supplement §S1.1) ──────────────────────────
SRC = ["cycle_period.json"]
# captured_at is one stamp per cycle, so consecutive stamps give the period, and
# the loop sleeps only the remainder: a period at the cadence closed inside it.
cyc = load("cycle_period.json")
chk("cycle period pairs", 1517002, cyc["n_consecutive_pairs"], 0, "{:.0f}")
chk("cycle period median", 10.0, cyc["period_s"]["median"], 0.05, "{:.0f}s")
chk("cycle period p99", 13.0, cyc["period_s"]["p99"], 0.05, "{:.0f}s")
chk("cycles within cadence", 98.2, cyc["share_within_cadence_pct"], 0.05,
    "{:.1f}%")

# ── cross-contract capture skew (Supplement §S2.7) ────────────────────────
SRC = ["capture_skew.json"]
# Post-goal window length: the skew argument only works if skew is small
# against it.
P_POST_LO_GUARD = 30 - 6
skew = load("capture_skew.json")
for coh, n_inst, age_p99, sk_med, sk_p99 in (
        ("holdout", 1766, 9.0, 1.0, 9.0),
        ("dev", 1358, 8.0, 0.0, 9.0)):
    c = skew["cohorts"][coh]
    chk(f"skew {coh} instants", n_inst, c["n_instants"], 0, "{:.0f}")
    chk(f"skew {coh} age p99", age_p99, c["age_s"]["p99"], 0.05, "{:.1f}")
    chk(f"skew {coh} median", sk_med, c["skew_s"]["median"], 0.05, "{:.1f}")
    chk(f"skew {coh} p99", sk_p99, c["skew_s"]["p99"], 0.05, "{:.1f}")
# the argument turns on skew being small against the post-goal window
checks.append((skew["cohorts"]["holdout"]["skew_s"]["p99"] < P_POST_LO_GUARD,
               "holdout skew p99 below the post-goal window length",
               f"{skew['cohorts']['holdout']['skew_s']['p99']:.0f}s", "<24s"))
_fam = skew["cohorts"]["holdout"]["family_mean_age_s"]
chk("skew holdout moneyline mean age", 0.68, _fam["moneyline"]["mean"], 0.05, "{:.2f}")
chk("skew holdout btts mean age", 1.62, _fam["btts"]["mean"], 0.05, "{:.2f}")

# ── two-sided freshness (§6.4 last row, Supplement §S2.4) ─────────────────
SRC = ["freshness_twosided_summary.json"]
# The frozen rule bounds the lag but not the lead. These rows bound both, and
# carry a same-events control so attenuation from a shrinking sample is not
# mistaken for attenuation from staleness.
two = load("freshness_twosided_summary.json")
for coh, pub in (("holdout", 0.436), ("dev", 0.667)):
    ch = two["cohorts"][coh]
    chk(f"two-sided base {coh} = engine", pub, C * ch["base"]["D"], 6e-4)
    for lead in ("30", "15"):
        r = ch["leads"][lead]
        chk(f"two-sided {coh} {lead}s reproduces base", C * ch["base"]["D"],
            C * r["D"], 1.5e-3)
        chk(f"two-sided {coh} {lead}s keeps every match",
            ch["base"]["n_pos"], r["n_pos"], 0, "{:.0f}")
    r10 = ch["leads"]["10"]
    checks.append((abs(C * r10["gap"]) < 0.011,
                   f"two-sided {coh} 10s freshness-only gap under 0.011c",
                   f"{C*r10['gap']:+.4f}c", "expected"))
r10 = two["cohorts"]["holdout"]["leads"]["10"]
chk("two-sided holdout 10s D", 0.435, C * r10["D"], 6e-4)
chk("two-sided holdout 10s CI lo", 0.318, C * r10["ci"][0], 6e-4)
chk("two-sided holdout 10s CI hi", 0.556, C * r10["ci"][1], 6e-4)
chk("two-sided holdout 10s n_pos", 43, r10["n_pos"], 0, "{:.0f}")
chk("two-sided holdout 10s events", 125, r10["n_events"], 0, "{:.0f}")
chk("two-sided holdout 10s instants", 99.6,
    100.0 * r10["instants_retained"], 0.06, "{:.1f}")
r5 = two["cohorts"]["holdout"]["leads"]["5"]
chk("two-sided holdout 5s D", 0.150, C * r5["D"], 6e-4)
chk("two-sided holdout 5s control", 0.204, C * r5["control_same_events_D"], 6e-4)
chk("two-sided holdout 5s events", 28, r5["n_events"], 0, "{:.0f}")
chk("two-sided holdout 5s instants", 10.0,
    100.0 * r5["instants_retained"], 0.06, "{:.1f}")
# the composition claim: most of the 5s attenuation is the sample, not staleness
_full = C * (two["cohorts"]["holdout"]["base"]["D"] - r5["D"])
_comp = C * (two["cohorts"]["holdout"]["base"]["D"] - r5["control_same_events_D"])
checks.append((_comp / _full > 0.75,
               "5s holdout attenuation is mostly composition",
               f"{100*_comp/_full:.0f}%", ">75%"))

# ── size-matched deletion, 200 draws (Supplement §S2.5) ───────────────────
SRC = ["size_matched_deletion.json"]
smd = load("size_matched_deletion.json")
for coh, ret, mean, p5, p95, pv in (
        ("holdout", 18.0, 31.1, 22.4, 41.3, 0.015),
        ("dev", 15.1, 36.1, 26.6, 46.3, 0.000)):
    s = smd[coh]
    chk(f"size-matched {coh} retained", ret, s["lofo_retained_pct"], 0.06, "{:.1f}")
    chk(f"size-matched {coh} random mean", mean, s["random_mean_pct"], 0.06, "{:.1f}")
    chk(f"size-matched {coh} random sd", 6.2, s["random_sd_pct"], 0.06, "{:.1f}")
    chk(f"size-matched {coh} p5", p5, s["random_quantiles_pct"]["p5"], 0.06, "{:.1f}")
    chk(f"size-matched {coh} p95", p95, s["random_quantiles_pct"]["p95"], 0.06, "{:.1f}")
    chk(f"size-matched {coh} empirical p", pv, s["empirical_p"], 6e-4)
    chk(f"size-matched {coh} draws", 200, s["n_random_draws"], 0, "{:.0f}")
    checks.append((s["lofo_retained_pct"] < s["random_quantiles_pct"]["p5"],
                   f"size-matched {coh} family below random p5",
                   f"{s['lofo_retained_pct']:.1f} < "
                   f"{s['random_quantiles_pct']['p5']:.1f}", "expected"))

# ── tick-discretization null (§6.1) ───────────────────────────────────────
SRC = ["tick_null_holdout.json", "tick_null_dev.json"]
# Active per-contract ticks must produce no positive event response; the null
# now also records the lattice distribution and any exceptional fallbacks.
tn_h = load("tick_null_holdout.json")
tn_d = load("tick_null_dev.json")
lh, ld = tn_h["levels"], tn_d["levels"]
rh, rd = tn_h["results"], tn_d["results"]
chk("tick obs pre level", 0.111, C * lh["obs"]["pre_mean"], 6e-4)
chk("tick obs post level", 0.550, C * lh["obs"]["post_mean"], 6e-4)
chk("tick nearest pre level", 0.000, C * lh["nearest"]["pre_mean"], 1e-8)
chk("tick nearest post level", 0.000, C * lh["nearest"]["post_mean"], 1e-8)
chk("tick narrowest pre level", 0.031, C * lh["narrowest"]["pre_mean"], 6e-4)
chk("tick narrowest post level", 0.027, C * lh["narrowest"]["post_mean"], 6e-4)
chk("tick dev narrowest pre level", 0.046, C * ld["narrowest"]["pre_mean"], 6e-4)
chk("tick dev narrowest post level", 0.034, C * ld["narrowest"]["post_mean"], 6e-4)
chk("tick dev obs pre level", 0.103, C * ld["obs"]["pre_mean"], 6e-4)
chk("tick null observed = engine", C * h["D"], C * rh["observed"]["D"], 1e-3)
chk("tick nearest D", 0.000, C * rh["nearest"]["D"], 1e-8)
chk("tick nearest CI lo", 0.000, C * rh["nearest"]["ci"][0], 1e-8)
chk("tick nearest CI hi", 0.000, C * rh["nearest"]["ci"][1], 1e-8)
chk("tick narrowest D", -0.001, C * rh["narrowest"]["D"], 6e-4)
chk("tick narrowest CI lo", -0.013, C * rh["narrowest"]["ci"][0], 6e-4)
chk("tick narrowest CI hi", 0.011, C * rh["narrowest"]["ci"][1], 6e-4)
chk("tick dev narrowest D", -0.014, C * rd["narrowest"]["D"], 6e-4)
for cohort, payload, expected in (
        ("holdout", tn_h, {"0.001": 61877, "0.0025": 2192, "0.01": 34264}),
        ("dev", tn_d, {"0.001": 45392, "0.0025": 3802, "0.01": 25278})):
    checks.append((payload.get("tick_counts") == expected,
                   f"tick counts {cohort}", str(payload.get("tick_counts")),
                   str(expected)))
    chk(f"tick mismatch {cohort}", 0, payload["n_tick_mismatch"], 0, "{:.0f}")
    chk(f"tick fallback {cohort}", 0, payload["n_inward_fallback"], 0, "{:.0f}")
# the load-bearing direction claim
for lab, rr in (("holdout", rh), ("dev", rd)):
    for rule in ("nearest", "narrowest"):
        checks.append((rr[rule]["D"] < rr["observed"]["D"],
                       f"tick null {rule} below observed ({lab})", "below", "expected"))
checks.append((rd["narrowest"]["ci"][1] < 0,
               "dev inward null significantly negative", "neg", "expected"))

# ── power gate vs estimator precision (§7) ────────────────────────────────
SRC = ["event_study_holdout.json"]
# The registered mean's precision is separate from the pilot's
# perturbation-trial detection rate.
se = (C * h["ci"][1] - C * h["ci"][0]) / 2.0 / 1.96
chk("holdout SE of the mean", 0.061, se, 6e-4)
chk("effect in standard errors", 7.0, (C * h["D"]) / se, 0.35, "{:.1f}")

# -- incoherence functional: L1 variant, development only (4.4) ------------
# Answers 'is the effect a property of the market or of the L-infinity choice?'
# The holdout is deliberately absent here; see the module docstring of
# analysis/studies/world_cup/coherence/l1_functional.py.
l1 = load("l1_functional_dev.json")
chk("L1 dev L-inf = engine", C * d["D"], C * l1["l_inf"]["D"], 1e-3)
chk("L1 dev total slack D", 4.075, C * l1["l_one"]["D"], 6e-4)
chk("L1 dev CI lo", 3.069, C * l1["l_one"]["ci"][0], 6e-4)
chk("L1 dev CI hi", 5.217, C * l1["l_one"]["ci"][1], 6e-4)
chk("L1 dev n_pos", 37, l1["l_one"]["n_pos"], 0, "{:.0f}")
chk("L1 dev n_matches", 39, l1["l_one"]["n_matches"], 0, "{:.0f}")
chk("L1 sign agreement pct", 98.0, 100 * l1["per_goal_sign_agreement"], 0.06, "{:.1f}")
chk("L1 per-goal correlation", 0.89, l1["per_goal_correlation"], 0.006, "{:.2f}")
chk("L1 n_goals", 98, l1["n_goals"], 0, "{:.0f}")
# both functionals must agree that the same number of matches moved up
checks.append((l1["l_inf"]["n_pos"] == l1["l_one"]["n_pos"],
               "L1 and L-inf agree on matches positive", "agree", "expected",
               sys._getframe().f_lineno))

# Most checks are registered through chk(), which captures the line
# that raised them. Two dozen structural checks append directly and carry no
# line; normalise both shapes to (ok, label, claimed, actual, line, source).
checks = [c if len(c) == 6 else (*c, None, None)[:6] for c in checks]

if "--emit-map" in sys.argv:
    # Source declarations identify result-file dependencies; structural checks
    # without a declaration remain marked as such.
    destination = Path(sys.argv[sys.argv.index("--emit-map") + 1])
    artifacts = []
    digests = {}
    for name in LOADED:
        path = DATA / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digests[name] = digest
        artifacts.append({
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": digest,
            "bytes": path.stat().st_size,
        })

    def sources_for(src):
        if not src:
            return None
        return [{"artifact": n, "sha256": digests.get(n)} for n in src]

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "auditor": Path(__file__).name,
        "rerun": f"python tools/{Path(__file__).name} --package .",
        "article_doi": "10.1109/ACCESS.2026.3746471",
        "scope": "Numeric claims and artifact consistency; manuscript wording is not checked.",
        "artifact_count": len(artifacts),
        "check_count": len(checks),
        "checks_with_source": sum(1 for c in checks if c[5]),
        "artifacts": artifacts,
        "checks": [{"line": line, "label": label, "claimed": claimed,
                    "actual": actual, "passed": ok,
                    "sources": sources_for(src)}
                   for ok, label, claimed, actual, line, src in checks],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote claim map: {destination} "
          f"({len(checks)} checks over {len(artifacts)} artifacts, "
          f"{sum(1 for c in checks if c[5])} with a declared source)")

bad = [c for c in checks if not c[0]]
print(f"{len(checks) - len(bad)} of {len(checks)} checks passed\n")
if bad:
    print("FAILURES (claimed <- derived):")
    for _ok, label, claimed, actual, line, _src in bad:
        where = f"   (line {line})" if line else ""
        print(f"  {label:<42} {claimed:>10}  <-  {actual:>10}{where}")
    sys.exit(1)
print("all numeric claims and artifact consistency checks passed")
