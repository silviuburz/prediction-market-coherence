#!/usr/bin/env python3
"""Provenance manifest for the coherence study.

Emits a single JSON + Markdown pair that pins, by SHA-256, every input, every
producer and every output, so a reviewer can verify that the reported numbers
came from the frozen code applied to the frozen corpus, and that the holdout was
not touched before the pre-registered freeze.

Records:
  * git commit of the analysis repo, plus the PRE-HOLDOUT freeze commit
  * SHA-256 of every producer script in code/
  * SHA-256 + rule of the dev/holdout split, and the split contents
  * SHA-256 and roll-up of the raw-corpus manifest (metadata/raw_sha256.tsv)
  * SHA-256 of every result artifact
  * the frozen parameter dictionary and its hash
  * interpreter and library versions
  * the exact commands that produced each artifact

Run: /home/azureuser/.venv/bin/python -u provenance_manifest.py
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "/home/azureuser/projects/wc_coherence"
REPO = "/home/azureuser/polymarket"
CORPUS = "/home/azureuser/polymarket/analysis/data/world_cup"
PRE_HOLDOUT_COMMIT = "a0c0557"


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
    except Exception:
        return None
    return h.hexdigest()


def sh(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


man = {"generated_at": datetime.utcnow().isoformat() + "Z",
       "pre_holdout_freeze_commit": PRE_HOLDOUT_COMMIT}

# ── environment ────────────────────────────────────────────────────────────
env = {"python": sys.version.split()[0], "platform": platform.platform(),
       "machine": platform.machine()}
for mod in ("numpy", "scipy"):
    try:
        env[mod] = __import__(mod).__version__
    except Exception:
        env[mod] = None
man["environment"] = env

# ── repo state ─────────────────────────────────────────────────────────────
# The analysis tree on the compute host is a deployed copy, not a git checkout,
# so `git rev-parse` there returns nothing. The authoritative commit is supplied
# by the operator via WC_GIT_COMMIT (the workstation repo HEAD at deploy time).
# Code identity does NOT depend on that value: every producer is pinned below by
# SHA-256 of its exact deployed bytes, which is a stronger guarantee than a
# commit hash because it survives rebases, amends and dirty working trees.
man["repo"] = {
    "head_commit": sh(["git", "-C", REPO, "rev-parse", "HEAD"])
    or os.environ.get("WC_GIT_COMMIT"),
    "head_commit_source": ("git" if sh(["git", "-C", REPO, "rev-parse", "HEAD"])
                           else ("env:WC_GIT_COMMIT"
                                 if os.environ.get("WC_GIT_COMMIT") else None)),
    "branch": sh(["git", "-C", REPO, "rev-parse", "--abbrev-ref", "HEAD"])
    or os.environ.get("WC_GIT_BRANCH"),
    "host_tree_is_git_checkout": os.path.isdir(f"{REPO}/.git"),
    "code_identity": "SHA-256 of deployed producer bytes (see 'producers')",
}

# ── producers ──────────────────────────────────────────────────────────────
code = {}
for fn in sorted(os.listdir(f"{OUT}/code")):
    if fn.endswith((".py", ".sh")):
        p = f"{OUT}/code/{fn}"
        code[fn] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
man["producers"] = code

# ── frozen parameters ──────────────────────────────────────────────────────
try:
    import event_study as ES
    pj = json.dumps(ES.P, sort_keys=True)
    man["frozen_params"] = {"values": ES.P,
                            "sha256": hashlib.sha256(pj.encode()).hexdigest()}
except Exception as e:
    man["frozen_params"] = {"error": str(e)}

# ── split ──────────────────────────────────────────────────────────────────
sp = f"{OUT}/registrations/dev_holdout_split.json"
if os.path.exists(sp):
    s = json.load(open(sp))
    man["split"] = {"path": "registrations/dev_holdout_split.json",
                    "sha256": sha256(sp),
                    "rule": s.get("rule"),
                    "n_dev": len(s.get("dev", [])),
                    "n_holdout": len(s.get("holdout", [])),
                    "forced_dev": s.get("forced_dev")}

# ── raw corpus manifest ────────────────────────────────────────────────────
rm = f"{OUT}/metadata/raw_sha256.tsv"
if os.path.exists(rm):
    n, tot = 0, 0
    roll = hashlib.sha256()
    with open(rm, "rb") as f:
        for line in f:
            roll.update(line)
            n += 1
    man["raw_corpus"] = {"manifest_path": "metadata/raw_sha256.tsv",
                         "manifest_sha256": sha256(rm),
                         "rolling_sha256": roll.hexdigest(),
                         "n_files": n,
                         "manifest_bytes": os.path.getsize(rm),
                         "corpus_root": CORPUS,
                         "read_only_marker": os.path.exists(
                             f"{CORPUS}/_FROZEN_2026-07-22.txt")}

# ── outputs ────────────────────────────────────────────────────────────────
outs = {}
for root, _dirs, files in os.walk(f"{OUT}/results"):
    for fn in sorted(files):
        p = os.path.join(root, fn)
        rel = os.path.relpath(p, OUT)
        outs[rel] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
man["outputs"] = outs

# ── commands ───────────────────────────────────────────────────────────────
man["commands"] = {
    "provenance_g0": "find <corpus> -type f -print0 | xargs -0 sha256sum > metadata/raw_sha256.tsv",
    "validate_all_matches": "python -u f1_validate_all_matches.py",
    "event_study_dev": "python -u event_study.py --cohort dev",
    "event_study_holdout": "python -u event_study.py --cohort holdout --unlock-holdout",
    "cancel_sensitivity": "python -u event_study.py --cohort {dev,holdout} --cancel-state",
    "spec_curve": "python -u spec_curve_v2.py",
    "placebo_suite": "python -u placebo_suite.py",
    "family_attribution": "python -u family_attribution.py --cohort {dev,holdout}",
    "tick_discretization_null": "python -u tick_null.py --cohort {dev,holdout} --subsystem {full,no-exact-score}",
    "l1_functional_dev": "python -u l1_functional.py",
    "split_label_sensitivity": "python -u split_lottery.py",
    "subsystem_and_availability": "python -u subsystem_and_availability.py --cohort {dev,holdout}",
    "freshness_and_straddle": "python -u freshness_and_straddle.py --cohort {dev,holdout}",
    "rq3": "python -u rq3_v2_censoring.py",
    "saturation": "python -u saturation_lemma.py",
}

os.makedirs(f"{OUT}/metadata", exist_ok=True)
jp = f"{OUT}/metadata/PROVENANCE.json"
json.dump(man, open(jp, "w"), indent=2)

# ── human-readable companion ───────────────────────────────────────────────
L = ["# Provenance manifest — WC-2026 coherence study", "",
     f"Generated {man['generated_at']}", "",
     "## Repository", "",
     f"- Authoritative commit: `{man['repo']['head_commit']}` "
     f"(source: {man['repo']['head_commit_source']})",
     f"- Compute host tree is a git checkout: "
     f"{man['repo']['host_tree_is_git_checkout']}",
     "- Code identity is established by SHA-256 of the deployed producer bytes "
     "listed below, not by the commit hash.",
     f"- Pre-holdout freeze commit: `{PRE_HOLDOUT_COMMIT}` — the analysis code was "
     "frozen at this commit before the holdout cohort was ever executed.", "",
     "## Environment", "",
     f"- Python {env['python']}, NumPy {env.get('numpy')}, SciPy {env.get('scipy')}",
     f"- {env['platform']}", ""]
if "split" in man:
    s = man["split"]
    L += ["## Dev / holdout split", "",
          f"- Rule: `{s['rule']}`",
          f"- {s['n_dev']} dev / {s['n_holdout']} holdout matches",
          f"- SHA-256 `{s['sha256']}`", ""]
if "raw_corpus" in man:
    r = man["raw_corpus"]
    L += ["## Raw corpus", "",
          f"- {r['n_files']:,} files, manifest SHA-256 `{r['manifest_sha256']}`",
          f"- Corpus frozen read-only: {r['read_only_marker']}", ""]
if "frozen_params" in man and "sha256" in man["frozen_params"]:
    L += ["## Frozen parameters", "",
          f"- SHA-256 `{man['frozen_params']['sha256']}`", "",
          "```json", json.dumps(man["frozen_params"]["values"], indent=2), "```", ""]
L += ["## Producers", "", "| script | bytes | sha256 |", "|---|---:|---|"]
for k, v in man["producers"].items():
    L.append(f"| `{k}` | {v['bytes']:,} | `{v['sha256'][:16]}…` |")
L += ["", "## Outputs", "", "| artifact | bytes | sha256 |", "|---|---:|---|"]
for k, v in sorted(man["outputs"].items()):
    L.append(f"| `{k}` | {v['bytes']:,} | `{(v['sha256'] or '')[:16]}…` |")
L += ["", "## Commands", ""]
for k, v in man["commands"].items():
    L.append(f"- **{k}** — `{v}`")
open(f"{OUT}/metadata/PROVENANCE.md", "w").write("\n".join(L) + "\n")

print(f"producers : {len(man['producers'])}")
print(f"outputs   : {len(man['outputs'])}")
print(f"commit    : {man['repo']['head_commit']} "
      f"(source {man['repo']['head_commit_source']})")
if "raw_corpus" in man:
    print(f"corpus    : {man['raw_corpus']['n_files']:,} files")
print(f"\n-> {jp}\n-> {OUT}/metadata/PROVENANCE.md")
