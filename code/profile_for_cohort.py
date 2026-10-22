"""Event-time goal profile for an arbitrary cohort directory.

figure_inputs.py is one of the producers recorded by SHA-256 in PROVENANCE.json,
and its cohort list and output root are both hardcoded, so it cannot be pointed
at results/holdout_corrected without editing it. This reproduces only the goal
profile from that file, for a named cohort directory.

Because a reimplementation can drift, --verify runs it against a cohort whose
profile is already in the frozen event_time_profiles.json and compares every
value. Trust the corrected output only if that comparison is exact.

    python profile_for_cohort.py --verify holdout
    python profile_for_cohort.py holdout_corrected -o /tmp/profile_holdout_corrected.json
"""
import argparse
import gzip
import json
import os
import time

import numpy as np

OUT = "/home/azureuser/projects/wc_coherence"
FIGS = f"{OUT}/figs"

# identical to figure_inputs.py lines 51-57; a divergence here is a silent bug,
# which is what --verify exists to catch
WIN, GRID, ISO_GAP = 300, 5, 300
PRE_LO, PRE_HI, POST_LO, POST_HI = -60, -6, 6, 30
N_BOOT, SEED = 2000, 29
C = 100.0
OFFSETS = list(range(-WIN, WIN + 1, GRID))


def win_mean(arr, gt, centre, lo, hi):
    i0 = int(np.searchsorted(gt, centre + lo))
    i1 = int(np.searchsorted(gt, centre + hi))
    seg = arr[i0:i1 + 1]
    seg = seg[~np.isnan(seg)]
    return float(seg.mean()) if len(seg) else None


def boot_band(per_match_rows, rng):
    ms = sorted(per_match_rows)
    if len(ms) < 3:
        return None
    M = np.vstack([per_match_rows[m] for m in ms])
    mean = np.nanmean(M, axis=0)
    idx = rng.integers(0, len(ms), size=(N_BOOT, len(ms)))
    draws = np.nanmean(M[idx], axis=1)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    return mean, lo, hi, len(ms)


def goal_profile(cohort):
    """The 'goal' class only; control classes need the ESPN key-event feed and
    are unaffected by the payoff correction."""
    p = f"{OUT}/results/{cohort}/match_stats.json.gz"
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}")
    with gzip.open(p, "rt") as f:
        MS = json.load(f)

    rng = np.random.default_rng(SEED)
    per_match, n_ev = {}, 0
    for m, st in MS.items():
        Sc = np.array(st["Sc"], dtype=float)
        gt = st["grid_t0"] + st["grid_step"] * np.arange(st["n_grid"])
        t0, t1, gw = st["t_start"], st["t_end"], sorted(st["goal_w"])
        rows = []
        for w in gw:
            if w - WIN < t0 or w + WIN > t1:
                continue
            if any(abs(w - o) <= ISO_GAP for o in gw if abs(o - w) > 1e-9):
                continue
            if (win_mean(Sc, gt, w, PRE_LO, PRE_HI) is None
                    or win_mean(Sc, gt, w, POST_LO, POST_HI) is None):
                continue
            prof = np.full(len(OFFSETS), np.nan)
            for j, off in enumerate(OFFSETS):
                i = int(np.searchsorted(gt, w + off))
                if 0 <= i < len(gt):
                    prof[j] = Sc[i]
            rows.append(prof)
            n_ev += 1
        if rows:
            stack = np.vstack(rows)
            with np.errstate(invalid="ignore"):
                col_ok = np.isfinite(stack).any(axis=0)
                prof_m = np.full(stack.shape[1], np.nan)
                prof_m[col_ok] = np.nanmean(stack[:, col_ok], axis=0)
            per_match[m] = prof_m

    band = boot_band(per_match, rng)
    if band is None:
        raise SystemExit(f"{cohort}: too few matches for a band")
    mean, lo, hi, nm = band
    return {"cohort": cohort,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "producer": "profile_for_cohort.py",
            "source": f"results/{cohort}/match_stats.json.gz",
            "event_class": "goal",
            "params": {"WIN": WIN, "GRID": GRID, "N_BOOT": N_BOOT, "SEED": SEED},
            "band_caveat": (
                "lo_cents and hi_cents are NOT comparable with the bands in the "
                "frozen event_time_profiles.json. figure_inputs.py draws every "
                "bootstrap from one module-level RNG, so a band there depends on "
                "how many boot_band calls preceded it. This producer starts a "
                "fresh RNG at the same seed, which reproduces the first call "
                "(dev/goal) bit-for-bit and no later one. mean_cents, offsets "
                "and the counts carry no RNG dependence and do reproduce "
                "exactly; verify with --verify holdout."),
            "offsets": OFFSETS,
            "mean_cents": [None if not np.isfinite(v) else C * float(v)
                           for v in mean],
            "lo_cents": [None if not np.isfinite(v) else C * float(v)
                         for v in lo],
            "hi_cents": [None if not np.isfinite(v) else C * float(v)
                         for v in hi],
            "n_matches": nm, "n_events": n_ev}


ap = argparse.ArgumentParser()
ap.add_argument("cohort")
ap.add_argument("-o", "--out")
ap.add_argument("--verify", action="store_true",
                help="compare against the frozen profile for this cohort")
a = ap.parse_args()

prof = goal_profile(a.cohort)
print(f"{a.cohort}: {prof['n_matches']} matches, {prof['n_events']} events")

if a.verify:
    frozen = json.load(open(f"{FIGS}/event_time_profiles.json"))
    if a.cohort not in frozen:
        raise SystemExit(f"{a.cohort} is not in the frozen profiles")
    ref = frozen[a.cohort]["goal"]
    worst, counts = {}, {}
    for k in ("offsets", "n_matches", "n_events"):
        counts[k] = 0 if prof[k] == ref[k] else 1
    for k in ("mean_cents", "lo_cents", "hi_cents"):
        n = 0
        w = 0.0
        for x, y in zip(prof[k], ref[k]):
            if (x is None) != (y is None):
                n += 1
            elif x is not None and abs(x - y) > 1e-9:
                n += 1
                w = max(w, abs(x - y))
        counts[k], worst[k] = n, w

    print(f"\n{'field':<14}{'differs':>9}{'max abs diff':>15}")
    for k, n in counts.items():
        print(f"{k:<14}{n:>9}{worst.get(k, 0.0):>15.2e}")

    # The bootstrap band is drawn from a module-level RNG in figure_inputs, so
    # its state depends on every boot_band call before it. Reproducing the band
    # would mean replaying that whole sequence. The mean does not depend on the
    # RNG, and the mean is the only series scene 5 draws.
    if counts["mean_cents"] or counts["offsets"] or counts["n_events"]:
        raise SystemExit("mean or counts differ - reimplementation is NOT faithful")
    print("\nmean_cents, offsets and counts match exactly.")
    if counts["lo_cents"] or counts["hi_cents"]:
        print("Bands differ only through RNG call order; scene 5 does not draw them.")

if a.out:
    json.dump(prof, open(a.out, "w"))
    print(f"wrote {a.out}")
