#!/usr/bin/env python3
"""Shared payoff compiler for the WC coherence study (validated at Gate G1).

Compiles primary score-determined contracts into typed payoff rules over an
explicit state space of payoff-equivalence classes, and resolves fixtures to
ESPN regulation-time scores.

Primary families (D2): moneyline(1X2), exact-score incl. `any-other`,
total goals, team totals, BTTS, spread.
EXCLUDED (D3): first-half / second-half variants (not functions of final score).
EXCLUDED (D4): player props, corners (separate ladder graphs).

Canonical orientation (D7) is always outcome index 0, i.e. clob_token_ids[0].
"""
import glob
import gzip
import json
import re
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher

ROOT = "/home/azureuser/polymarket/analysis/data/world_cup"
LIVE = f"{ROOT}/live"
CAP = 15                       # saturation bound; > max primary threshold + 2


def as_list(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return []
    return x or []


def _norm(s):
    return re.sub(r"[^a-z]", "", (s or "").lower())


def _tokens(s):
    """Sorted word tokens -- makes matching invariant to word order
    ('DR Congo' vs 'Congo DR')."""
    return tuple(sorted(w for w in re.split(r"[^a-z]+", (s or "").lower()) if w))


# Explicit, auditable alias table for genuinely different national-team names
# between Polymarket metadata and ESPN. Preferred over loose fuzzy matching so
# every mapping is reviewable. Keys and values are _norm()-ed forms.
TEAM_ALIASES = {
    "drcongo": "congodr",
    "korearepublic": "southkorea",
    "southkorea": "southkorea",
    "caboverde": "capeverde",
    "ivorycoast": "cotedivoire",
    "cotedivoire": "cotedivoire",
    "unitedstates": "unitedstates",
    "usa": "unitedstates",
    "bosnia": "bosniaherzegovina",
}


def _canon(nm):
    return TEAM_ALIASES.get(nm, nm)


def _sim(a, b):
    return SequenceMatcher(None, a, b).ratio() if (a and b) else 0.0


# ── typed rules: payoff of the CANONICAL outcome in state (h,a) ──────────────
def _exact(i, j):
    return lambda h, a: 1.0 if (h == i and a == j) else 0.0


def _any_other():
    return lambda h, a: 0.0 if (h <= 3 and a <= 3) else 1.0


def _ml_home():
    return lambda h, a: 1.0 if h > a else 0.0


def _ml_away():
    return lambda h, a: 1.0 if a > h else 0.0


def _draw():
    return lambda h, a: 1.0 if h == a else 0.0


def _total_over(k):
    return lambda h, a: 1.0 if (h + a) > k else 0.0


def _tt_home(k):
    return lambda h, a: 1.0 if h > k else 0.0


def _tt_away(k):
    return lambda h, a: 1.0 if a > k else 0.0


def _btts():
    return lambda h, a: 1.0 if (h > 0 and a > 0) else 0.0


def _sp_home(k):
    return lambda h, a: 1.0 if (h - a) > k else 0.0


def _sp_away(k):
    return lambda h, a: 1.0 if (a - h) > k else 0.0


HALF = re.compile(r"(first-half|second-half|halftime)")


def compile_rule(slug, match, home_code, away_code):
    """slug -> (family, rule) or None. FAILS CLOSED on anything unrecognised."""
    s = (slug or "").lower()
    pref = f"fifwc-{match}-"
    if not s.startswith(pref):
        return None
    body = s[len(pref):]
    if HALF.search(body):
        return None                                       # D3
    m = re.fullmatch(r"exact-score-(\d+)-(\d+)", body)
    if m:
        return ("exact-score", _exact(int(m.group(1)), int(m.group(2))))
    if body == "exact-score-any-other":
        return ("exact-score", _any_other())
    if body == home_code:
        return ("moneyline", _ml_home())
    if body == away_code:
        return ("moneyline", _ml_away())
    if body == "draw":
        return ("moneyline", _draw())
    if body == "btts":
        return ("btts", _btts())
    m = re.fullmatch(r"total-(\d+)pt5", body)
    if m:
        return ("total-goals", _total_over(int(m.group(1)) + 0.5))
    m = re.fullmatch(r"team-total-(home|away)-(\d+)pt5", body)
    if m:
        k = int(m.group(2)) + 0.5
        return ("team-total", _tt_home(k) if m.group(1) == "home" else _tt_away(k))
    m = re.fullmatch(r"spread-(home|away)-(\d+)pt5", body)
    if m:
        k = int(m.group(2)) + 0.5
        return ("spread", _sp_home(k) if m.group(1) == "home" else _sp_away(k))
    return None


# ── ESPN fixture resolution ──────────────────────────────────────────────────
_ESPN_INDEX = None


def espn_index():
    """Parse every scoreboard row ONCE and group by game_id.
    A single `_espn.jsonl.gz` can contain rows from several games (scoreboard
    rollover), so grouping by game_id is mandatory."""
    global _ESPN_INDEX
    if _ESPN_INDEX is not None:
        return _ESPN_INDEX
    games = defaultdict(list)
    for p in glob.glob(f"{LIVE}/*/*/_espn.jsonl.gz"):
        try:
            with gzip.open(p, "rt", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    gid = str(d.get("game_id") or "")
                    if gid:
                        games[gid].append(d)
        except Exception:
            continue
    for gid in games:
        games[gid].sort(key=lambda d: str(d.get("captured_at") or ""))
    _ESPN_INDEX = dict(games)
    return _ESPN_INDEX


def regulation_score(code_name, t1, t2, mdate):
    """-> (H, A, home_code, home_team, away_team, game_id, rows) or None.

    Date-gated, name-first matching with an explicit alias table, word-order
    invariant token matching, and a similarity fallback (Polymarket 'Cabo Verde'
    vs ESPN 'Cape Verde'; 'DR Congo' vs 'Congo DR'; 'Korea Republic' vs
    'South Korea'). Orientation is taken from ESPN, never from slug order.
    """
    n1, n2 = _norm(code_name.get(t1)), _norm(code_name.get(t2))
    k1, k2 = _canon(n1), _canon(n2)
    tk1, tk2 = _tokens(code_name.get(t1)), _tokens(code_name.get(t2))
    for gid, grows in espn_index().items():
        meta = grows[-1]
        hn, an = _norm(meta.get("home_team")), _norm(meta.get("away_team"))
        kh, ka = _canon(hn), _canon(an)
        th, ta = _tokens(meta.get("home_team")), _tokens(meta.get("away_team"))
        ha, aa = _norm(meta.get("home_abbrev")), _norm(meta.get("away_abbrev"))
        kdates = {str(d.get("kickoff_iso") or d.get("captured_at") or "")[:10]
                  for d in grows}
        kdates.discard("")
        if kdates and mdate:
            try:
                md = datetime.strptime(mdate, "%Y-%m-%d")
                if not any(abs((datetime.strptime(k, "%Y-%m-%d") - md).days) <= 1
                           for k in kdates):
                    continue
            except Exception:
                pass

        def side_of(code, nm, kn, tk):
            if nm and nm == hn:
                return "home"
            if nm and nm == an:
                return "away"
            if kn and kn == kh:
                return "home"                     # alias table
            if kn and kn == ka:
                return "away"
            if tk and tk == th:
                return "home"                     # word-order invariant
            if tk and tk == ta:
                return "away"
            if code == ha:
                return "home"
            if code == aa:
                return "away"
            if nm and hn and (nm in hn or hn in nm):
                return "home"
            if nm and an and (nm in an or an in nm):
                return "away"
            sh, sa = _sim(nm, hn), _sim(nm, an)
            if max(sh, sa) >= 0.75 and abs(sh - sa) > 0.10:
                return "home" if sh > sa else "away"
            return None

        s1, s2 = side_of(t1, n1, k1, tk1), side_of(t2, n2, k2, tk2)
        if not (s1 and s2 and s1 != s2):
            continue
        home_code = t1 if s1 == "home" else t2
        best = None
        for d in grows:
            try:
                per = int(d.get("period") or 0)
            except Exception:
                continue
            hs, as_ = d.get("home_score"), d.get("away_score")
            if per and per <= 2 and hs is not None and as_ is not None:
                best = (int(hs), int(as_))
        if best:
            return (best[0], best[1], home_code, meta.get("home_team"),
                    meta.get("away_team"), gid, grows)
    return None


def load_manifest():
    mp = sorted(glob.glob(f"{ROOT}/daily/*/manifest.json.gz"))[-1]
    return json.load(gzip.open(mp, "rt")), mp


def match_markets(M, match):
    out = []
    for lst in M.get("markets_by_event", {}).values():
        for mk in lst:
            s = (mk.get("slug") or "").lower()
            if s.startswith(f"fifwc-{match}-") or s == f"fifwc-{match}":
                out.append(mk)
    return out


def build_contracts(M, match):
    """-> (contracts, info) where contracts = list of dicts with payoff rule and
    canonical/complement token ids. FAILS CLOSED on unknown templates."""
    t1, t2 = match.split("-")[0], match.split("-")[1]
    mkts = match_markets(M, match)
    code_name = {}
    for mk in mkts:
        s = (mk.get("slug") or "").lower()
        for c in (t1, t2):
            if s == f"fifwc-{match}-{c}":
                code_name[c] = mk.get("group_item_title")
    reg = regulation_score(code_name, t1, t2, match[-10:])
    if not reg:
        return None, {"error": "espn_unresolved", "n_markets": len(mkts)}
    H, A, home_code, hteam, ateam, gid, espn_rows = reg
    away_code = t2 if home_code == t1 else t1

    contracts, unknown = [], 0
    for mk in mkts:
        r = compile_rule(mk.get("slug") or "", match, home_code, away_code)
        if r is None:
            unknown += 1
            continue
        fam, fn = r
        toks = [str(t) for t in as_list(mk.get("clob_token_ids") or mk.get("clobTokenIds"))]
        contracts.append({
            "slug": (mk.get("slug") or "").lower(),
            "family": fam,
            "rule": fn,
            "condition_id": mk.get("condition_id") or mk.get("conditionId"),
            "canon_token": toks[0] if len(toks) > 0 else None,
            "comp_token": toks[1] if len(toks) > 1 else None,
            "outcomes": as_list(mk.get("outcomes")),
            "neg_risk": mk.get("neg_risk"),
            "outcome_prices": as_list(mk.get("outcome_prices") or mk.get("outcomePrices")),
        })
    info = {"H": H, "A": A, "home_code": home_code, "home_team": hteam,
            "away_team": ateam, "game_id": gid, "n_markets": len(mkts),
            "n_compiled": len(contracts), "n_failed_closed": unknown,
            "espn_rows": espn_rows}
    return contracts, info


def state_classes(contracts, cap=CAP):
    """Payoff-equivalence classes over the score lattice (saturation-proven)."""
    classes = {}
    for h in range(cap + 1):
        for a in range(cap + 1):
            vec = tuple(c["rule"](h, a) for c in contracts)
            classes.setdefault(vec, (h, a))
    reps = sorted(classes.values())
    return reps


def saturation_ok(contracts, cap=CAP):
    known = {tuple(c["rule"](h, a) for c in contracts)
             for h in range(cap + 1) for a in range(cap + 1)}
    probes = [(cap + 1, 0), (0, cap + 1), (cap + 1, cap + 1), (cap + 3, 2),
              (2, cap + 3), (cap + 5, cap + 4), (30, 0), (0, 30), (25, 25), (40, 37)]
    return all(tuple(c["rule"](h, a) for c in contracts) in known for h, a in probes)


def payoff_matrix(contracts, reps):
    """X[j][s] = payoff of contract j in state s."""
    return [[c["rule"](h, a) for (h, a) in reps] for c in contracts]


def cancel_payoff(c):
    """Payoff of the CANONICAL outcome in the match-cancelled state.

    Verified verbatim from frozen `_baseline/markets/<condition_id>.json.gz`
    descriptions (not assumed):
      exact-score : "if it is canceled with no make-up game, the market
                     resolves to '0-0'"        -> 0-0 cell pays 1, others 0
      moneyline   : home/away "resolve 'No'"    -> 0
                    draw      "resolve to 'Yes'"-> 1
      total-goals / team-total / btts / spread :
                    "resolve 50-50"             -> 0.5

    This vector is NOT reproducible by any score state (a genuine 0-0 pays 0 on
    totals, not 0.5), so including it strictly ENLARGES the coherent set and can
    only REDUCE measured slack -- i.e. omitting it biases results in favour of
    finding incoherence.
    """
    fam, slug = c["family"], c["slug"]
    if fam == "exact-score":
        return 1.0 if slug.endswith("-exact-score-0-0") else 0.0
    if fam == "moneyline":
        return 1.0 if slug.endswith("-draw") else 0.0
    if fam in ("total-goals", "team-total", "btts", "spread"):
        return 0.5
    return 0.0


def payoff_matrix_with_cancel(contracts, reps):
    """Payoff matrix with one extra CANCELLED state appended as the last column."""
    X = payoff_matrix(contracts, reps)
    for row, c in zip(X, contracts):
        row.append(cancel_payoff(c))
    return X
