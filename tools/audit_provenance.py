#!/usr/bin/env python3
"""Verify the replication package against the frozen provenance record.

The manuscript states that producer identity is established by SHA-256 of the
exact deployed bytes. PROVENANCE.json is that record: 48 producers and 63
outputs, each with a hash and a byte count, written on the analysis host at the
end of the run.

This resolves every recorded producer to bytes we actually hold. A producer may
live in the working tree, or -- because the deployed version was sometimes
superseded by later edits -- in an earlier commit. Both count, as long as the
bytes hash correctly. Anything that resolves nowhere is a hole in the package
and fails the audit.

Outputs are matched by name and, when a cohort subdirectory was flattened on
export, by content hash.

Run: python tools/audit_provenance.py
     python tools/audit_provenance.py --emit docs/.../PROVENANCE_AUDIT.json
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
# --package points the audit at an exported replication package instead of this
# repository, so a reviewer can verify what they actually received. A package has
# no git history and is laid out flat, so producers must resolve from shipped
# files alone.
PACKAGE_MODE = "--package" in sys.argv
if PACKAGE_MODE:
    ROOT = Path(sys.argv[sys.argv.index("--package") + 1]).resolve()
    RECORD = ROOT / "PROVENANCE.json"
    STUDY = ROOT / "code"
    FROZEN = ROOT / "frozen_producers"
    # Mirrors the repository, where data is a sibling of the figure scripts
    # because figures/_style.py resolves it that way.
    DATA = ROOT / "figures" / "data"
else:
    PAPER = ROOT / "docs" / "research" / "paper_wc_coherence"
    RECORD = PAPER / "submission" / "PROVENANCE.json"
    STUDY = ROOT / "analysis" / "studies" / "world_cup" / "coherence"
    FROZEN = PAPER / "submission" / "frozen_producers"
    DATA = ROOT / "figures" / "wc_coherence" / "data"
STUDY_REL = "analysis/studies/world_cup/coherence"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True)


def commits_touching(relative_path: str) -> list[str]:
    result = git("log", "--all", "--format=%H", "--", relative_path)
    return result.stdout.decode("utf-8", "replace").split()


def resolve_producer(name: str, wanted: str) -> tuple[str, str | None]:
    """Locate bytes hashing to `wanted`: working tree first, then history.

    The deployed line endings are CRLF, matching a Windows checkout, so blobs
    recovered from git (stored LF) are also tried in their CRLF form.
    """
    local = STUDY / name
    if local.is_file() and sha256(local.read_bytes()) == wanted:
        return "working tree", None

    frozen = FROZEN / name
    if frozen.is_file() and sha256(frozen.read_bytes()) == wanted:
        return "frozen_producers", None

    if PACKAGE_MODE:
        return "UNRESOLVED", None

    relative = f"{STUDY_REL}/{name}"
    for commit in commits_touching(relative):
        blob = git("show", f"{commit}:{relative}")
        if blob.returncode:
            continue
        for data in (blob.stdout, blob.stdout.replace(b"\n", b"\r\n")):
            if sha256(data) == wanted:
                return "git history", commit[:8]
    return "UNRESOLVED", None


def main() -> int:
    if not RECORD.is_file():
        print(f"missing provenance record: {RECORD}")
        return 1
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    producers = record["producers"]
    outputs = record["outputs"]

    print(f"provenance record : {RECORD.relative_to(ROOT).as_posix()}")
    print(f"recorded at       : {record.get('generated_at')}")
    print(f"code identity     : {record.get('code_identity', '')}")
    environment = record.get("environment", {})
    print(f"environment       : Python {environment.get('python')}, "
          f"NumPy {environment.get('numpy')}, SciPy {environment.get('scipy')}")
    print()

    resolved, from_git, unresolved, details = 0, 0, [], []
    for name in sorted(producers):
        where, commit = resolve_producer(name, producers[name]["sha256"])
        if where == "UNRESOLVED":
            unresolved.append(name)
        else:
            resolved += 1
            if where == "git history":
                from_git += 1
        details.append({"producer": name, "resolved_from": where,
                        "commit": commit,
                        "sha256": producers[name]["sha256"]})
        if where != "working tree":
            marker = "!!" if where == "UNRESOLVED" else "  "
            print(f"  {marker} {name:<34} {where}"
                  f"{f' @ {commit}' if commit else ''}")

    print(f"\nPRODUCERS  {resolved}/{len(producers)} resolved "
          f"({resolved - from_git} in the working tree, {from_git} from history)")
    for name in unresolved:
        print(f"     UNRESOLVED  {name}")

    by_name, by_hash = {}, {}
    for path, meta in outputs.items():
        by_name.setdefault(Path(path).name, []).append(meta["sha256"])
        by_hash[meta["sha256"]] = path
    # The development/holdout registration is recorded under its own top-level
    # key rather than in outputs, because it is an input to the analysis.
    split = record.get("split", {})
    if split.get("sha256"):
        by_hash[split["sha256"]] = split.get("path", "registrations/split.json")

    # The record's outputs section covers results/ only. These three are written
    # to figs/ by figure_inputs.py, itself a recorded producer, so their code
    # provenance holds even though their hashes are outside the record.
    FIGURE_STAGE = {"event_time_profiles.json", "threshold_ccdf.json",
                    "randomization_null.json"}
    # Independently derived from Wikipedia under CC BY-SA; never a host output.
    EXTERNAL = {"wikipedia_wc2026_results.json"}
    # Produced after the record was written, in answer to review. Outside the
    # frozen record by construction: the record cannot describe work that did
    # not exist when it was sealed.
    POST_RECORD = {
        "freshness_twosided_summary.json": "freshness_twosided.py",
        "size_matched_deletion.json": "fam_attr_draws.py",
        "fail_closed_breakdown.json": "fail_closed_breakdown.py",
        "first_to_score_neither.json": "fts_neither_sensitivity.py",
        # Answering the IEEE Access review of 2026-09-20. The corrected
        # specification applies the cancelled settlement state and the missing
        # score-resolved contract together and is the primary estimate of
        # Section V; the funnel is the event-selection table of Section IV-F,
        # derived from the engine's own match_stats rather than recomputed.
        "corrected_specification.json": "corrected_specification.py",
        "event_funnel.json": "event_funnel.py",
        # The supplementary video's event-time scene, regenerated from the
        # corrected run. figure_inputs.py is a recorded producer whose cohort
        # list and output root are hardcoded, so it could not be pointed at the
        # corrected cohort without editing it; this reproduces its goal-profile
        # step for one named cohort and self-checks against the frozen output.
        "event_time_profile_holdout_corrected.json": "profile_for_cohort.py",
        "capture_skew.json": "capture_skew.py",
        "description_validation.json": "validate_descriptions.py",
        "system_composition.json": "system_composition.py",
        "outcome_orientation.json": "validate_orientation.py",
        "cycle_period.json": "cycle_period.py",
    }

    shipped = sorted(DATA.glob("*.json"))
    matched, mismatched, figure_stage, external, unrecorded = [], [], [], [], []
    post_record = []
    artifact_details = []
    for artifact in shipped:
        digest = sha256(artifact.read_bytes())
        if digest in by_hash:
            matched.append(artifact.name)
            recorded_as = by_hash[digest]
        elif artifact.name in by_name:
            mismatched.append(artifact.name)
            recorded_as = "MISMATCH"
        elif artifact.name in FIGURE_STAGE:
            figure_stage.append(artifact.name)
            recorded_as = "figs/ via figure_inputs.py (producer recorded)"
        elif artifact.name in EXTERNAL:
            external.append(artifact.name)
            recorded_as = "externally derived, not a host output"
        elif artifact.name in POST_RECORD:
            post_record.append(artifact.name)
            recorded_as = (f"post-record, produced by "
                           f"{POST_RECORD[artifact.name]}")
        else:
            unrecorded.append(artifact.name)
            recorded_as = None
        artifact_details.append({"artifact": artifact.name,
                                 "recorded_as": recorded_as,
                                 "sha256": digest})

    print(f"\nARTIFACTS  {len(shipped)} shipped")
    print(f"     {len(matched):3}  reconcile bit-for-bit with the frozen record")
    print(f"     {len(figure_stage):3}  figure-stage, from a recorded producer, "
          f"hash outside the record")
    print(f"     {len(external):3}  externally derived by design")
    print(f"     {len(post_record):3}  produced after the record was written")
    print(f"     {len(mismatched):3}  MISMATCH")
    print(f"     {len(unrecorded):3}  unaccounted for")
    for name in figure_stage:
        print(f"        figs/  {name}")
    for name in post_record:
        print(f"        post-record {name}  ({POST_RECORD[name]})")
    for name in mismatched:
        print(f"        MISMATCH    {name}")
    for name in unrecorded:
        print(f"        UNACCOUNTED {name}")

    if "--emit" in sys.argv:
        destination = Path(sys.argv[sys.argv.index("--emit") + 1]).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({
            "record": RECORD.relative_to(ROOT).as_posix(),
            "producer_count": len(producers),
            "producers_resolved": resolved,
            "producers_unresolved": unresolved,
            "artifacts_checked": len(shipped),
            "artifacts_reconciled": len(matched),
            "artifacts_figure_stage": figure_stage,
            "artifacts_external": external,
            "artifacts_post_record": post_record,
            "artifacts_mismatched": mismatched,
            "artifacts_unaccounted": unrecorded,
            "producer_detail": details,
            "artifact_detail": artifact_details,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {destination.relative_to(ROOT).as_posix()}")

    if unresolved or mismatched or unrecorded:
        print("\nFAILED: the package cannot account for every recorded item")
        return 1
    print("\nevery recorded producer resolves and every shipped artifact is "
          "accounted for")
    return 0


if __name__ == "__main__":
    sys.exit(main())
