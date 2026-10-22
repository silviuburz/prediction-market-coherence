#!/usr/bin/env python3
"""SPIKE STEP 2 (Gate G1) — payoff compiler + settlement validation.

Compiles primary score-determined contracts into typed rules over an explicit
state space, builds payoff-equivalence classes, proves saturation, and validates
every rule against the REALIZED regulation-time score.

Primary families (D2): moneyline(1X2), exact-score incl. `any-other`,
total goals, team totals, BTTS, spread.
EXCLUDED (D3): every first-half / second-half variant (not functions of final score).
EXCLUDED (D4): player props, corners (separate ladder graphs).

Semantics established by spike step 1:
  exact-score-i-j        outcomes ['Yes','No']      canonical = Yes
  exact-score-any-other  outcomes ['Yes','No']      canonical = Yes
  <team>|draw            outcomes ['Yes','No']      canonical = Yes
  total-Xpt5             outcomes ['Over','Under']  canonical = Over
  team-total-side-Xpt5   outcomes ['Over','Under']  canonical = Over
  btts                   outcomes ['Yes','No']      canonical = Yes
  spread-side-Xpt5       outcomes [<fav>,<dog>]     canonical = favoured side covers

CRITICAL: contracts settle on REGULATION time (90'+stoppage), excluding extra
time and penalties. Realized score is therefore taken from the last ESPN
scoreboard row with period <= 2.

Read-only. Run: /home/azureuser/.venv/bin/python -u /tmp/spike2.py
"""
import glob
import gzip
import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher

ROOT = "/home/azureuser/polymarket/analysis/data/world_cup"
LIVE = f"{ROOT}/live"
DEV = ["eng-arg-2026-07-15", "arg-cvi-2026-07-03", "eng-hrv-2026-06-17"]
CAP = 15                      # saturation bound; > max primary threshold + 2


def as_list(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return []
    return x or []


# ── typed rules: each returns payoff of the CANONICAL outcome in state (h,a) ──
def rule_exact(i, j):
    return lambda h, a: 1.0 if (h == i and a == j) else 0.0


def rule_any_other():
    return lambda h, a: 0.0 if (h <= 3 and a <= 3) else 1.0


def rule_ml_home():
    return lambda h, a: 1.0 if h > a else 0.0


def rule_ml_away():
    return lambda h, a: 1.0 if a > h else 0.0


def rule_draw():
    return lambda h, a: 1.0 if h == a else 0.0


def rule_total_over(k):
    return lambda h, a: 1.0 if (h + a) > k else 0.0


def rule_tt_home(k):
    return lambda h, a: 1.0 if h > k else 0.0


def rule_tt_away(k):
    return lambda h, a: 1.0 if a > k else 0.0


def rule_btts():
    return lambda h, a: 1.0 if (h > 0 and a > 0) else 0.0


def rule_spread_home(k):
    return lambda h, a: 1.0 if (h - a) > k else 0.0


def rule_spread_away(k):
    return lambda h, a: 1.0 if (a - h) > k else 0.0


HALF = re.compile(r"(first-half|second-half|halftime)")


def compile_rule(slug, match, t1, t2):
    """slug -> (family, rule) or None if not a primary contract. FAILS CLOSED."""
    s = slug.lower()
    body = s[len(f"fifwc-{match}-"):] if s.startswith(f"fifwc-{match}-") else None
    if body is None:
        return None
    if HALF.search(body):
        return None                                   # D3
    m = re.fullmatch(r"exact-score-(\d+)-(\d+)", body)
    if m:
        return ("exact-score", rule_exact(int(m.group(1)), int(m.group(2))))
    if body == "exact-score-any-other":
        return ("exact-score", rule_any_other())
    if body == t1:
        return ("moneyline", rule_ml_home())
    if body == t2:
        return ("moneyline", rule_ml_away())
    if body == "draw":
        return ("moneyline", rule_draw())
    if body == "btts":
        return ("btts", rule_btts())
    m = re.fullmatch(r"total-(\d+)pt5", body)
    if m:
        return ("total-goals", rule_total_over(int(m.group(1)) + 0.5))
    m = re.fullmatch(r"team-total-(home|away)-(\d+)pt5", body)
    if m:
        k = int(m.group(2)) + 0.5
        return ("team-total", rule_tt_home(k) if m.group(1) == "home" else rule_tt_away(k))
    m = re.fullmatch(r"spread-(home|away)-(\d+)pt5", body)
    if m:
        k = int(m.group(2)) + 0.5
        return ("spread", rule_spread_home(k) if m.group(1) == "home" else rule_spread_away(k))
    return None                                        # unknown -> fail closed


# ── realized regulation-time score from ESPN (period <= 2) ───────────────────
def _norm(s):
    return re.sub(r"[^a-z]", "", (s or "").lower())


def _sim(a, b):
    return SequenceMatcher(None, a, b).ratio() if (a and b) else 0.0


def regulation_score(code_name, t1, t2, mdate):
    """Resolve the ESPN stream for this fixture and return the regulation-time
    score plus which slug code is ESPN's HOME side.

    Polymarket uses ISO-ish codes (hrv, cvi) while ESPN uses its own abbrevs
    (CRO, CPV), so match on TEAM NAME first, then abbrev, then fuzzy contains.
    Orientation is taken from ESPN -- never assumed from slug order.

    NOTE: an `_espn.jsonl.gz` file can contain rows from more than one game
    (scoreboard rollover -- the same artefact that produced spurious score
    reversals). We therefore group rows by `game_id` and never trust the last
    row of the file.
    """
    n1, n2 = _norm(code_name.get(t1)), _norm(code_name.get(t2))
    for p in glob.glob(f"{LIVE}/*/*/_espn.jsonl.gz"):
        rows = []
        try:
            with gzip.open(p, "rt", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            continue
        by_game = defaultdict(list)
        for d in rows:
            by_game[str(d.get("game_id"))].append(d)

        for gid, grows in by_game.items():
            meta = grows[-1]
            hn, an = _norm(meta.get("home_team")), _norm(meta.get("away_team"))
            ha, aa = _norm(meta.get("home_abbrev")), _norm(meta.get("away_abbrev"))

            # date gate: fixture identity is (date, team pair). Allow +/-1 day
            # for midnight rollover.
            kdates = {str(d.get("kickoff_iso") or d.get("captured_at") or "")[:10]
                      for d in grows}
            kdates.discard("")
            if kdates and mdate:
                try:
                    md = int(mdate.replace("-", ""))
                    if not any(abs(int(k.replace("-", "")) - md) <= 1 for k in kdates):
                        continue
                except Exception:
                    pass

            def side_of(code, nm):
                if nm and nm == hn:
                    return "home"
                if nm and nm == an:
                    return "away"
                if code == ha:
                    return "home"
                if code == aa:
                    return "away"
                if nm and hn and (nm in hn or hn in nm):
                    return "home"
                if nm and an and (nm in an or an in nm):
                    return "away"
                # fuzzy fallback (e.g. 'caboverde' vs 'capeverde'), must be
                # both high-similarity AND clearly closer to one side
                sh, sa = _sim(nm, hn), _sim(nm, an)
                if max(sh, sa) >= 0.75 and abs(sh - sa) > 0.10:
                    return "home" if sh > sa else "away"
                return None

            s1, s2 = side_of(t1, n1), side_of(t2, n2)
            if not (s1 and s2 and s1 != s2):
                continue
            home_code = t1 if s1 == "home" else t2
            best = None
            for d in grows:                              # last row with period <= 2
                try:
                    per = int(d.get("period") or 0)
                except Exception:
                    continue
                hs, as_ = d.get("home_score"), d.get("away_score")
                if per and per <= 2 and hs is not None and as_ is not None:
                    best = (int(hs), int(as_))
            if best:
                return (best[0], best[1], home_code,
                        meta.get("home_team"), meta.get("away_team"))
    return None


mp = sorted(glob.glob(f"{ROOT}/daily/*/manifest.json.gz"))[-1]
M = json.load(gzip.open(mp, "rt"))

print("=" * 78)
print("SPIKE 2 — PAYOFF COMPILER + SETTLEMENT VALIDATION (Gate G1)")
print("=" * 78)

overall_ok = True
for match in DEV:
    t1, t2 = match.split("-")[0], match.split("-")[1]

    mkts = []
    for lst in M.get("markets_by_event", {}).values():
        for mk in lst:
            s = (mk.get("slug") or "").lower()
            if s.startswith(f"fifwc-{match}-") or s == f"fifwc-{match}":
                mkts.append(mk)

    # code -> team name, from the moneyline markets' group_item_title
    code_name = {}
    for mk in mkts:
        s = (mk.get("slug") or "").lower()
        for c in (t1, t2):
            if s == f"fifwc-{match}-{c}":
                code_name[c] = mk.get("group_item_title")

    reg = regulation_score(code_name, t1, t2, match[-10:])
    print(f"\n--- {match} ---")
    print(f"  slug codes: {t1}={code_name.get(t1)!r}  {t2}={code_name.get(t2)!r}")
    if not reg:
        print("  ESPN regulation score NOT resolved -> FAIL CLOSED")
        overall_ok = False
        continue
    H, A, home_code, hteam, ateam = reg
    away_code = t2 if home_code == t1 else t1
    print(f"  ESPN home={hteam!r} away={ateam!r}  -> home_code={home_code}")
    print(f"  regulation-time score (period<=2): {hteam} {H} - {A} {ateam}")

    compiled, unknown, fam_ct = [], 0, defaultdict(int)
    for mk in mkts:
        r = compile_rule(mk.get("slug") or "", match, home_code, away_code)
        if r is None:
            unknown += 1
            continue
        fam, fn = r
        fam_ct[fam] += 1
        compiled.append((mk, fam, fn))
    print(f"  contracts: {len(mkts)}  compiled: {len(compiled)}  "
          f"non-primary/failed-closed: {unknown}")
    print(f"  by family: {dict(fam_ct)}")

    # settlement validation against realized regulation score
    fails = []
    checked = 0
    for mk, fam, fn in compiled:
        opx = as_list(mk.get("outcome_prices") or mk.get("outcomePrices"))
        if not opx or str(opx[0]) not in ("0", "1"):
            continue                                    # unresolved -> skip
        realized = 1.0 if str(opx[0]) == "1" else 0.0
        predicted = fn(H, A)
        checked += 1
        if abs(predicted - realized) > 1e-9:
            fails.append((mk.get("slug"), fam, predicted, realized))
    print(f"  settlement-validated: {checked}   MISMATCHES: {len(fails)}")
    for slug, fam, pred, real in fails[:12]:
        print(f"     MISMATCH [{fam}] {slug}: predicted={pred} realized={real}")
    if fails:
        overall_ok = False

# ── payoff-equivalence classes + saturation check ────────────────────────────
print("\n" + "=" * 78)
print("PAYOFF-EQUIVALENCE CLASSES + SATURATION (using eng-hrv rule set)")
print("=" * 78)
match = "eng-hrv-2026-06-17"
t1, t2 = "eng", "hrv"
_mkts = []
for lst in M.get("markets_by_event", {}).values():
    for mk in lst:
        s = (mk.get("slug") or "").lower()
        if s.startswith(f"fifwc-{match}-"):
            _mkts.append(mk)
_code_name = {}
for mk in _mkts:
    s = (mk.get("slug") or "").lower()
    for c in (t1, t2):
        if s == f"fifwc-{match}-{c}":
            _code_name[c] = mk.get("group_item_title")
_reg = regulation_score(_code_name, t1, t2, match[-10:])
if not _reg:
    print("  cannot resolve orientation for the class enumeration -> FAIL")
    overall_ok = False
    rules = []
else:
    _home = _reg[2]
    _away = t2 if _home == t1 else t1
    rules = []
    for mk in _mkts:
        s = (mk.get("slug") or "").lower()
        r = compile_rule(s, match, _home, _away)
        if r:
            rules.append((s, r[1]))
rules.sort()
print(f"  primary contracts in rule set: {len(rules)}")

classes = {}
for h in range(CAP + 1):
    for a in range(CAP + 1):
        vec = tuple(fn(h, a) for _s, fn in rules)
        classes.setdefault(vec, (h, a))
print(f"  distinct payoff vectors over 0..{CAP}: {len(classes)}")

# saturation: states beyond CAP must reuse an existing payoff vector
viol = []
for h, a in [(CAP + 1, 0), (0, CAP + 1), (CAP + 1, CAP + 1), (CAP + 3, 2),
             (2, CAP + 3), (CAP + 5, CAP + 4), (30, 0), (0, 30), (25, 25)]:
    vec = tuple(fn(h, a) for _s, fn in rules)
    if vec not in classes:
        viol.append((h, a))
print(f"  saturation probes beyond CAP: {'PASS' if not viol else f'FAIL {viol}'}")
if viol:
    overall_ok = False

print("\n" + "=" * 78)
print(f"GATE G1: {'PASS' if overall_ok else 'FAIL'}")
print("=" * 78)
sys.exit(0 if overall_ok else 1)
