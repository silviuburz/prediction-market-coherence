#!/usr/bin/env python3
"""SPIKE STEP 1 — contract template inventory for the 3 development matches.

Before writing a payoff compiler we must READ the real semantics, not infer them
from slugs. For each primary-family template this prints: slug pattern, count,
the `outcomes` array, `group_item_title`, a sample `question`, and the resolved
`outcome_prices` (needed for settlement validation).

Development matches only: eng-arg-2026-07-15, arg-cvi-2026-07-03, eng-hrv-2026-06-17.
Read-only. Run: /home/azureuser/.venv/bin/python -u /tmp/spike1.py
"""
import glob
import gzip
import json
import re
from collections import defaultdict

ROOT = "/home/azureuser/polymarket/analysis/data/world_cup"
DEV = ["eng-arg-2026-07-15", "arg-cvi-2026-07-03", "eng-hrv-2026-06-17"]


def as_list(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return []
    return x or []


def template_of(slug, match):
    """Reduce a slug to a template by replacing match id and numeric params."""
    s = slug.replace(f"fifwc-{match}-", "")
    s = re.sub(r"\d+pt5", "<X>pt5", s)
    s = re.sub(r"(?<=exact-score-)\d+-\d+", "<i>-<j>", s)
    s = re.sub(r"-\d+$", "-<n>", s)
    return s


mp = sorted(glob.glob(f"{ROOT}/daily/*/manifest.json.gz"))[-1]
M = json.load(gzip.open(mp, "rt"))
print(f"manifest: {mp}\n")

# collect dev-match markets
rows = []
for mkts in M.get("markets_by_event", {}).values():
    for mk in mkts:
        slug = (mk.get("slug") or "").lower()
        for d in DEV:
            if f"fifwc-{d}-" in slug or slug == f"fifwc-{d}":
                rows.append((d, mk))
                break
print(f"dev-match markets found: {len(rows)}")

# ---- primary family classification (score-determined only) ------------------
PRIM = {
    "moneyline": lambda s, m: bool(re.fullmatch(
        rf"fifwc-{re.escape(m)}-(?:[a-z0-9]+|draw)", s)),
    "exact-score": lambda s, m: "exact-score" in s,
    "total-goals": lambda s, m: bool(re.search(r"-total-\d+pt5$", s)),
    "team-total": lambda s, m: bool(re.search(r"-team-total-(home|away)-\d+pt5$", s)),
    "btts": lambda s, m: s.endswith("-btts"),
    "spread": lambda s, m: bool(re.search(r"-spread-(home|away)-\d+pt5$", s)),
}

by_tmpl = defaultdict(list)
prim_count = defaultdict(int)
other = []
for d, mk in rows:
    slug = (mk.get("slug") or "").lower()
    fam = None
    for name, test in PRIM.items():
        try:
            if test(slug, d):
                fam = name
                break
        except Exception:
            pass
    if fam:
        prim_count[fam] += 1
        by_tmpl[(fam, template_of(slug, d))].append(mk)
    else:
        other.append(slug)

print("\n" + "=" * 78)
print("PRIMARY FAMILY COUNTS (across 3 dev matches)")
print("=" * 78)
for k, v in sorted(prim_count.items(), key=lambda kv: -kv[1]):
    print(f"  {k:<14}{v:>6}")
print(f"  {'(non-primary)':<14}{len(other):>6}")

print("\n" + "=" * 78)
print("TEMPLATE DETAIL — semantics needed by the compiler")
print("=" * 78)
for (fam, tmpl), mks in sorted(by_tmpl.items()):
    mk = mks[0]
    outs = as_list(mk.get("outcomes"))
    opx = as_list(mk.get("outcome_prices") or mk.get("outcomePrices"))
    print(f"\n  [{fam}] template: {tmpl}   (n={len(mks)})")
    print(f"     outcomes        : {outs}")
    print(f"     group_item_title: {mk.get('group_item_title')!r}")
    print(f"     question        : {str(mk.get('question'))[:150]!r}")
    print(f"     outcome_prices  : {opx}")
    print(f"     neg_risk        : {mk.get('neg_risk')}   slug: {mk.get('slug')}")

# ---- exact-score grid extent + residual naming ------------------------------
print("\n" + "=" * 78)
print("EXACT-SCORE GRID EXTENT (per dev match)")
print("=" * 78)
for d in DEV:
    cells = sorted({(mk.get("slug") or "").lower() for dd, mk in rows
                    if dd == d and "exact-score" in (mk.get("slug") or "").lower()})
    nums = [re.search(r"exact-score-(\d+)-(\d+)$", c) for c in cells]
    pts = sorted({(int(m.group(1)), int(m.group(2))) for m in nums if m})
    resid = [c for c in cells if not re.search(r"exact-score-\d+-\d+$", c)]
    print(f"  {d}: {len(cells)} cells; numeric={len(pts)}; "
          f"max_h={max((p[0] for p in pts), default=None)} "
          f"max_a={max((p[1] for p in pts), default=None)}")
    print(f"     residual/other cells: {[r.split('-')[-1] for r in resid]}")

# ---- threshold inventory (drives the saturation bound) ----------------------
print("\n" + "=" * 78)
print("THRESHOLD INVENTORY (drives saturation bound)")
print("=" * 78)
th = defaultdict(set)
for d, mk in rows:
    s = (mk.get("slug") or "").lower()
    m = re.search(r"-total-(\d+)pt5$", s)
    if m:
        th["total"].add(int(m.group(1)) + 0.5)
    m = re.search(r"-team-total-(home|away)-(\d+)pt5$", s)
    if m:
        th[f"team-total-{m.group(1)}"].add(int(m.group(2)) + 0.5)
    m = re.search(r"-spread-(home|away)-(\d+)pt5$", s)
    if m:
        th[f"spread-{m.group(1)}"].add(int(m.group(2)) + 0.5)
for k in sorted(th):
    print(f"  {k:<20} {sorted(th[k])}")
mx = max([max(v) for v in th.values()] or [0])
print(f"\n  max threshold across primary families: {mx}")
print(f"  => saturation bound CAP = {int(mx) + 2} is sufficient for these matches")

print("\n" + "=" * 78)
print("SAMPLE NON-PRIMARY SLUGS (excluded from the score graph)")
print("=" * 78)
for s in sorted(other)[:12]:
    print(f"  {s}")
print(f"  ... total non-primary: {len(other)}")
