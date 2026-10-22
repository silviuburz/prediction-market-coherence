"""Does the canonical leg's outcome *label* agree with the payoff we compiled?

The compiler orients every contract on clob_token_ids[0] and compiles the payoff
of that leg. Index order was trusted but never checked against the label the
index carries. If a market listed its outcomes in the other order, the contract
would be sign-flipped and nothing downstream would notice: the LP would simply
see a different, still-well-formed system.

Three label conventions appear in the corpus, and each pins the orientation:

  Yes / No          the canonical leg is the affirmative one
  Over / Under      the canonical leg is Over
  <team> / <team>   the canonical leg is the team named first, which must be the
                    side the slug names (-home-/-away-, or the trailing code)

Labels are resolved to sides with the compiler's own cascade, so this asserts
the orientation the compiler actually used rather than a second guess at it.

A reviewer asked that this be asserted rather than assumed.
"""
import json
import sys
from collections import Counter

ROOT = "/home/azureuser/projects/wc_coherence"
sys.path.insert(0, f"{ROOT}/code")
import compiler as cp  # noqa: E402

M, _ = cp.load_manifest()
split = json.load(open(f"{ROOT}/registrations/dev_holdout_split.json"))


def side_of(label, home, away):
    """Which side does this outcome label name? Uses the compiler's cascade:
    canonical alias, then word-order-invariant tokens, then similarity."""
    lab = cp._canon(cp._norm(label))
    h, a = cp._canon(cp._norm(home)), cp._canon(cp._norm(away))
    if lab == h:
        return "home"
    if lab == a:
        return "away"
    lt = cp._tokens(label)
    if lt and lt == cp._tokens(home):
        return "home"
    if lt and lt == cp._tokens(away):
        return "away"
    sh, sa = cp._sim(lab, h), cp._sim(lab, a)
    if max(sh, sa) >= 0.75 and abs(sh - sa) >= 0.10:
        return "home" if sh > sa else "away"
    return None


conv, bad, unchecked = Counter(), [], Counter()
n = 0

for cohort in ("dev", "holdout"):
    for match in sorted(split[cohort]):
        contracts, info = cp.build_contracts(M, match)
        if contracts is None:
            continue
        home, away = info["home_team"], info["away_team"]
        parts = match.split("-")
        hc, ac = parts[0], parts[1]

        for c in contracts:
            outs = [str(o).strip() for o in (c.get("outcomes") or [])]
            slug = c["slug"]
            n += 1
            if len(outs) != 2:
                unchecked[f"{len(outs)} outcomes"] += 1
                continue
            first, low = outs[0], outs[0].lower()

            if low in ("yes", "no"):
                conv["Yes / No"] += 1
                expect, got = "Yes", first
            elif low in ("over", "under"):
                conv["Over / Under"] += 1
                expect, got = "Over", first
            else:
                s0 = side_of(outs[0], home, away)
                s1 = side_of(outs[1], home, away)
                if s0 is None or s1 is None or s0 == s1:
                    unchecked[f"labels {outs} vs teams [{home}, {away}]"] += 1
                    continue
                conv["team / team"] += 1
                tail = slug.rsplit("-", 1)[-1]
                if "-home-" in slug or tail == hc:
                    expect = "home"
                elif "-away-" in slug or tail == ac:
                    expect = "away"
                else:
                    unchecked[f"no side marker: {slug}"] += 1
                    continue
                got = s0

            if got != expect:
                bad.append({"cohort": cohort, "match": match, "slug": slug,
                            "outcomes": outs, "expected_first": expect})

print(f"{n:,} compiled contracts checked\n")
print("label convention of the canonical leg:")
for label, k in conv.most_common():
    print(f"  {label:<18}{k:>8,}")
print(f"\n  {'total checked':<18}{sum(conv.values()):>8,}")

if unchecked:
    print(f"\ncontracts whose convention could not be determined: "
          f"{sum(unchecked.values())}")
    for label, k in unchecked.most_common(10):
        print(f"  {label[:70]:<72}{k:>5}")

print(f"\ncanonical leg mislabelled: {len(bad)}")
for b in bad[:10]:
    print(f"  {b['slug']:<52}{b['outcomes']} expected {b['expected_first']}")

out = {"n_contracts": n, "conventions": dict(conv),
       "n_checked": sum(conv.values()), "n_mislabelled": len(bad),
       "mislabelled": bad, "undetermined": dict(unchecked)}
json.dump(out, open(f"{ROOT}/results/outcome_orientation.json", "w"), indent=2)
print(f"\nwrote {ROOT}/results/outcome_orientation.json")
ok = not bad and not unchecked
print("PASS: every canonical leg's label agrees with the compiled payoff" if ok
      else "FAIL: orientation is not established for every contract")
sys.exit(0 if ok else 1)
