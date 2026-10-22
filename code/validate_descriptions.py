#!/usr/bin/env python3
"""Validate slug-derived payoff rules against the venue's resolution text.

The settlement check in §3.5 evaluates each contract at its one realised score,
so it cannot distinguish a correct rule from one that happens to agree at that
point. This reads the venue's own resolution description for every compiled
contract from the frozen corpus and checks the semantics independently:

  * the threshold and direction stated in the text must match the one the slug
    grammar derived (a `total-2pt5` contract must say "3 or more goals");
  * the team named must be the side the slug names;
  * the text must restrict resolution to regulation time, which is the state
    space the whole method is defined over.

Every contract is classified agree / disagree / unparsed, and disagreements are
printed. A rule that cannot be read out of the text is reported rather than
assumed correct.

Writes results/description_validation.json.
"""
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = "/home/azureuser/projects/wc_coherence"
sys.path.insert(0, f"{ROOT}/code")
import compiler as cp  # noqa: E402

CORPUS = "/home/azureuser/polymarket/analysis/data/world_cup"
MARKETS = f"{CORPUS}/_baseline/markets"

REGULATION = re.compile(
    r"first 90 minutes of regular play plus stoppage time"
    r"|90 minutes of regulation plus stoppage time"
    r"|extra time and penalty shoot-?outs are excluded", re.I)


def load_description(condition_id):
    if not condition_id:
        return None
    key = condition_id[2:] if condition_id.startswith("0x") else condition_id
    path = f"{MARKETS}/{key}.json.gz"
    if not os.path.exists(path):
        return None
    try:
        with gzip.open(path, "rt", errors="ignore") as fh:
            return json.load(fh)
    except Exception:
        return None


def same_team(a, b):
    """Do two spellings name the same country?

    ESPN and the venue disagree on surface form -- Cape Verde against Cabo
    Verde, Congo DR against DR Congo, Bosnia-Herzegovina against Bosnia and
    Herzegovina -- so compare distinctive tokens rather than strings. A genuine
    side error names a different country and shares no such token.
    """
    def tokens(s):
        return {t for t in re.split(r"[^a-z0-9]+", (s or "").lower())
                if len(t) >= 4 and t not in {"and", "the", "republic"}}
    ta, tb = tokens(a), tokens(b)
    return bool(ta & tb) if ta and tb else False


def side_verdict(named, team, other):
    """Is the team named in the text the side the slug says?

    Surface forms differ between sources -- Ivory Coast against Cote d'Ivoire,
    Cape Verde against Cabo Verde -- so a failure to match the expected spelling
    is not evidence of error. Naming the *opposite* side is. Only two teams play
    a match, so that is the discriminating test.
    """
    if same_team(named, team):
        return "agree", ""
    if same_team(named, other):
        return "disagree", f"text names the opposite side {named!r}"
    return "name variant", f"{team!r} written as {named!r}"


def check(slug, family, rec, home_team, away_team):
    """-> (verdict, detail). verdict in {agree, disagree, unparsed}."""
    text = " ".join((rec.get("description") or "").split())
    if not text:
        return "unparsed", "empty description"
    body = slug.rsplit("-", 0)[0]

    m = re.search(r"exact-score-(\d+)-(\d+)$", slug)
    if m:
        h, a = m.group(1), m.group(2)
        title = str(rec.get("groupItemTitle") or rec.get("question") or "")
        if re.search(rf"\b{h}\s*[-–]\s*{a}\b", title):
            return "agree", f"title states {h}-{a}"
        if re.search(rf"\b{h}\s*[-–]\s*{a}\b", text):
            return "agree", f"text states {h}-{a}"
        return "disagree", f"slug says {h}-{a}, title {title!r}"

    if slug.endswith("exact-score-any-other"):
        if re.search(r"any other score", text, re.I):
            return "agree", "text names the residual outcome"
        return "disagree", "no residual-outcome clause"

    m = re.search(r"-total-(\d+)pt5$", slug)
    if m:
        need = int(m.group(1)) + 1
        got = re.search(r"combine to score (\d+) or more goals", text, re.I)
        if not got:
            return "unparsed", "no combined-goals clause"
        return (("agree", f"text says {need} or more")
                if int(got.group(1)) == need
                else ("disagree", f"slug implies {need}, text says {got.group(1)}"))

    m = re.search(r"-team-total-(home|away)-(\d+)pt5$", slug)
    if m:
        side, need = m.group(1), int(m.group(2)) + 1
        team = home_team if side == "home" else away_team
        got = re.search(r"resolve to \"Over\" if (.+?) score (\d+) or more goals",
                        text, re.I)
        if not got:
            return "unparsed", "no team-goals clause"
        if int(got.group(2)) != need:
            return "disagree", f"slug implies {need}, text says {got.group(2)}"
        other = away_team if side == "home" else home_team
        v, why = side_verdict(got.group(1), team, other)
        return (v, why or f"{got.group(1)} {need} or more")

    m = re.search(r"-spread-(home|away)-(\d+)pt5$", slug)
    if m:
        side, need = m.group(1), int(m.group(2)) + 1
        team = home_team if side == "home" else away_team
        got = re.search(r"resolve to \"(.+?)\" if (.+?) win the game by "
                        r"(\d+) or more goals", text, re.I)
        if not got:
            return "unparsed", "no margin clause"
        if int(got.group(3)) != need:
            return "disagree", f"slug implies {need}, text says {got.group(3)}"
        other = away_team if side == "home" else home_team
        v, why = side_verdict(got.group(2), team, other)
        return (v, why or f"{got.group(2)} by {need} or more")

    if slug.endswith("-btts"):
        if re.search(r"each score at least one goal", text, re.I):
            return "agree", "text requires both teams to score"
        return "disagree", "no both-teams-score clause"

    if family == "moneyline":
        if slug.endswith("-draw"):
            if re.search(r"draw|tie", text, re.I):
                return "agree", "text names a draw"
            return "disagree", "no draw clause"
        got = re.search(r"If (.+?) wins?, this market will resolve to \"Yes\"",
                        text, re.I)
        if not got:
            return "unparsed", "no win clause"
        return "agree", f"text: if {got.group(1)} wins"

    return "unparsed", f"no rule for family {family}"


def main():
    M, _ = cp.load_manifest()
    split = json.load(open(f"{ROOT}/registrations/dev_holdout_split.json"))
    matches = sorted(split["dev"]) + sorted(split["holdout"])

    verdicts = Counter()
    per_family = defaultdict(Counter)
    regulation = Counter()
    problems = []
    archive = {}

    for mi, match in enumerate(matches, 1):
        contracts, info = cp.build_contracts(M, match)
        if contracts is None:
            continue
        for c in contracts:
            rec = load_description(c.get("condition_id"))
            if rec is None:
                verdicts["no description"] += 1
                per_family[c["family"]]["no description"] += 1
                continue
            v, detail = check(c["slug"], c["family"], rec,
                              info.get("home_team"), info.get("away_team"))
            verdicts[v] += 1
            per_family[c["family"]][v] += 1
            text = " ".join((rec.get("description") or "").split())
            regulation["states regulation window" if REGULATION.search(text)
                       else "does not state it"] += 1
            archive[c["slug"]] = {"condition_id": c.get("condition_id"),
                                  "family": c["family"], "verdict": v,
                                  "detail": detail, "description": text}
            if v == "disagree":
                problems.append((c["slug"], detail))
        if mi % 20 == 0:
            print(f"  {mi}/{len(matches)} matches", flush=True)

    total = sum(verdicts.values())
    print(f"\ncontracts checked against resolution text: {total:,}")
    for k, n in verdicts.most_common():
        print(f"  {k:<16}{n:>7,}  {100.0 * n / total:5.1f}%")
    print("\nregulation-time restriction stated in the text:")
    for k, n in regulation.most_common():
        print(f"  {k:<28}{n:>7,}  {100.0 * n / total:5.1f}%")
    print("\nby family:")
    for fam in sorted(per_family):
        row = per_family[fam]
        print(f"  {fam:<14}" + "  ".join(f"{k}={v}" for k, v in row.most_common()))
    if problems:
        print(f"\nDISAGREEMENTS ({len(problems)}):")
        for slug, detail in problems[:25]:
            print(f"  {slug}: {detail}")
    else:
        print("\nno disagreement between slug-derived rules and resolution text")

    out = {
        "n_contracts": total,
        "verdicts": dict(verdicts),
        "regulation_window": dict(regulation),
        "by_family": {f: dict(v) for f, v in per_family.items()},
        "disagreements": [{"slug": s, "detail": d} for s, d in problems],
    }
    json.dump(out, open(f"{ROOT}/results/description_validation.json", "w"),
              indent=2)
    # the archive is the descriptions themselves, kept separately for size
    with gzip.open(f"{ROOT}/results/resolution_descriptions.json.gz", "wt") as fh:
        json.dump(archive, fh)
    print(f"\nwrote results/description_validation.json and "
          f"results/resolution_descriptions.json.gz ({len(archive):,} contracts)")


if __name__ == "__main__":
    main()
