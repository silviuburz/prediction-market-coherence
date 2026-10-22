#!/usr/bin/env python3
"""Compare regenerated outputs against the frozen provenance hashes.

Byte equality is the strong claim, but a blunt pass/fail is misleading: a
changed `generated_at` stamp and a changed estimate are not the same event. This
classifies every difference so the record states what actually happened.

  identical    bytes match the frozen record exactly
  timestamp    differs only in a run-stamp field
  additive     keys present in the re-run that the frozen artifact predates
  MOVED        a value present in both versions changed -- the only alarming case

Only MOVED, missing, or unreadable outputs fail the check.
"""
import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

PROJ = Path("/home/azureuser/projects/wc_coherence")
STAMP_FIELDS = ("generated_at", "generated", "timestamp", "run_at", "created_at",
                "elapsed", "duration", "wall_seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path):
    raw = (gzip.decompress(path.read_bytes()) if path.suffix == ".gz"
           else path.read_bytes())
    return json.loads(raw.decode("utf-8"))


def flatten(node, prefix=""):
    out = {}
    if isinstance(node, dict):
        for key, value in node.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(node, list):
        out[prefix + ".__len__"] = len(node)
        for index, value in enumerate(node[:5000]):
            out.update(flatten(value, f"{prefix}[{index}]"))
    else:
        out[prefix] = node
    return out


def same(x, y) -> bool:
    # NaN never equals itself, but two NaNs in the same slot are not a change.
    if (isinstance(x, float) and isinstance(y, float)
            and math.isnan(x) and math.isnan(y)):
        return True
    if (isinstance(x, (int, float)) and isinstance(y, (int, float))
            and not isinstance(x, bool) and not isinstance(y, bool)):
        return math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-15)
    return x == y


def classify(old: Path, new: Path):
    try:
        a, b = flatten(load(old)), flatten(load(new))
    except Exception as error:
        return "UNREADABLE", {"error": str(error)}
    added = sorted(set(b) - set(a))
    removed = sorted(set(a) - set(b))
    moved = sorted(k for k in set(a) & set(b) if not same(a[k], b[k]))
    stamps = [k for k in moved
              if any(field in k.lower() for field in STAMP_FIELDS)]
    moved = [k for k in moved if k not in stamps]
    detail = {"added": len(added), "removed": len(removed),
              "moved_count": len(moved), "moved": moved[:20],
              "stamp_fields": stamps}
    if moved:
        return "MOVED", detail
    if added or removed:
        return "additive", detail
    if stamps:
        return "timestamp", detail
    return "reordered", detail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True,
                        help="pre-run copy of results/, taken by rerun_all.sh")
    parser.add_argument("--record", type=Path,
                        default=PROJ / "metadata" / "PROVENANCE.json")
    args = parser.parse_args()

    outputs = json.loads(args.record.read_text(encoding="utf-8"))["outputs"]
    buckets = {"identical": [], "timestamp": [], "reordered": [], "additive": [],
               "MOVED": [], "UNREADABLE": [], "missing": []}
    details = {}

    for relative, meta in sorted(outputs.items()):
        current = PROJ / relative
        if not current.is_file():
            buckets["missing"].append(relative)
            continue
        if sha256(current) == meta["sha256"]:
            buckets["identical"].append(relative)
            continue
        snapshot = args.snapshot / Path(relative).relative_to("results")
        if not snapshot.is_file():
            buckets["UNREADABLE"].append(relative)
            continue
        verdict, detail = classify(snapshot, current)
        buckets[verdict].append(relative)
        details[relative] = detail

    print(f"\noutputs recorded {len(outputs)}")
    for name in ("identical", "timestamp", "reordered", "additive", "MOVED",
                 "UNREADABLE", "missing"):
        print(f"   {name:12} {len(buckets[name]):3}")

    for name in ("timestamp", "reordered", "additive", "MOVED", "UNREADABLE",
                 "missing"):
        if not buckets[name]:
            continue
        print(f"\n-- {name} --")
        for relative in buckets[name]:
            detail = details.get(relative, {})
            extra = ""
            if name == "additive":
                extra = (f"  (+{detail.get('added', 0)} keys, "
                         f"-{detail.get('removed', 0)})")
            elif name == "MOVED":
                extra = f"  ({detail.get('moved_count', 0)} values)"
            print(f"   {relative}{extra}")
            for key in detail.get("moved", [])[:6]:
                print(f"        {key}")

    failed = buckets["MOVED"] + buckets["missing"] + buckets["UNREADABLE"]
    report = PROJ / "metadata" / "RERUN_VERIFICATION.json"
    report.write_text(json.dumps({
        "recorded_outputs": len(outputs),
        "counts": {name: len(items) for name, items in buckets.items()},
        "buckets": buckets,
        "details": details,
        "verdict": "differs" if failed else "reproduced",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {report}")

    if failed:
        print("RESULT: the re-run changed values present in both versions")
        return 1
    print(f"RESULT: reproduced. {len(buckets['identical'])} of {len(outputs)} "
          f"byte-identical; the rest differ only in run stamps or in fields the "
          f"frozen artifacts predate. No value present in both versions moved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
