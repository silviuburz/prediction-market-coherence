#!/usr/bin/env python3
"""Verify the ACTUAL cancellation / abandonment clauses in primary contract text.

The frozen manifest carries no `description` field, so the contingent-settlement
semantics claimed in the draft-readiness audit must be verified against the live
Gamma API before any cancellation state is added to the payoff matrix.

Fetches one representative market per primary family for a sample match and
prints the full description, so the compiled contingent payoff vector is grounded
in the actual rules rather than assumed.

Read-only w.r.t. the corpus. Run: python probe_descriptions.py
"""
import glob
import gzip
import json
import re
import sys
import time
import urllib.request

ROOT = "/home/azureuser/polymarket/analysis/data/world_cup"
MATCHES = ["eng-arg-2026-07-15", "eng-hrv-2026-06-17"]
WANT = [
    ("moneyline", re.compile(r"^fifwc-[a-z0-9]+-[a-z0-9]+-\d{4}-\d{2}-\d{2}-[a-z]{2,4}$")),
    ("draw", re.compile(r"-draw$")),
    ("exact-score", re.compile(r"-exact-score-\d+-\d+$")),
    ("exact-any-other", re.compile(r"-exact-score-any-other$")),
    ("total", re.compile(r"-total-2pt5$")),
    ("btts", re.compile(r"-btts$")),
    ("spread", re.compile(r"-spread-home-1pt5$")),
    ("team-total", re.compile(r"-team-total-home-1pt5$")),
]


def fetch(slug):
    url = f"https://gamma-api.polymarket.com/markets?slug={slug}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "wc-research/1.0"})
        with urllib.request.urlopen(req, timeout=25) as r:
            d = json.loads(r.read().decode())
        if isinstance(d, list) and d:
            return d[0]
    except Exception as e:
        return {"_error": str(e)}
    return None


mp = sorted(glob.glob(f"{ROOT}/daily/*/manifest.json.gz"))[-1]
M = json.load(gzip.open(mp, "rt"))
slugs = {}
for lst in M.get("markets_by_event", {}).values():
    for mk in lst:
        s = (mk.get("slug") or "").lower()
        for match in MATCHES:
            if not s.startswith(f"fifwc-{match}"):
                continue
            for name, rx in WANT:
                if rx.search(s) and (match, name) not in slugs:
                    slugs[(match, name)] = s

print("=" * 96)
print("CONTRACT DESCRIPTION PROBE — verifying contingent settlement clauses")
print("=" * 96)
KEY = re.compile(r"(cancel|abandon|postpon|suspend|void|abandoned|not.{0,15}played|"
                 r"50-50|50/50|refund|resolve[sd]? to|official|extra time|"
                 r"regulation|90 minutes|stoppage)", re.I)

for (match, name), slug in sorted(slugs.items()):
    mk = fetch(slug)
    time.sleep(0.4)
    print("\n" + "-" * 96)
    print(f"[{name}]  {slug}")
    if not mk or mk.get("_error"):
        print(f"   FETCH FAILED: {mk.get('_error') if mk else 'no result'}")
        continue
    print(f"   question : {mk.get('question')}")
    print(f"   outcomes : {mk.get('outcomes')}")
    for fld in ("description", "resolutionSource", "umaResolutionStatus"):
        v = mk.get(fld)
        if not v:
            continue
        txt = re.sub(r"\s+", " ", str(v)).strip()
        print(f"   {fld}:")
        for chunk in [txt[i:i + 150] for i in range(0, min(len(txt), 1500), 150)]:
            print(f"      {chunk}")
        hits = sorted(set(m.group(0).lower() for m in KEY.finditer(txt)))
        if hits:
            print(f"   >> contingency keywords: {hits}")
print("\n" + "=" * 96)
print("Use ONLY the clauses actually present above to build contingent states.")
