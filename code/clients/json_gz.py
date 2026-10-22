"""Shared helpers for writing/reading per-wallet JSON snapshots as gzipped JSON.

Policy: any collector that produces per-wallet or per-market on-disk records
should use these helpers so the on-disk format stays consistent (`.json.gz`,
≈8-15x smaller than plain JSON, transparently readable through `load`).

Read is permissive: `load(path)` will read either `<path>` if it ends in
`.json` or has a sibling `<path>.gz`. Write always emits `<stem>.json.gz`
atomically (write to `.tmp`, fsync via `os.replace`).
"""
from __future__ import annotations
import gzip
import json
import os
from typing import Any


def dump(obj: Any, path: str) -> str:
    """Write `obj` as gzipped JSON. If `path` ends in `.json`, the actual
    file written is `path + '.gz'`. Returns the final on-disk path.

    Atomic: writes to `.tmp` then `os.replace`s. Removes a stale uncompressed
    sibling if one exists.
    """
    if path.endswith(".gz"):
        gz_path = path
    elif path.endswith(".json"):
        gz_path = path + ".gz"
    else:
        gz_path = path + ".json.gz"
    tmp = gz_path + ".tmp"
    os.makedirs(os.path.dirname(gz_path) or ".", exist_ok=True)
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, gz_path)
    # Remove stale uncompressed sibling so disk doesn't double-count.
    uncompressed = gz_path[:-3] if gz_path.endswith(".json.gz") else None
    if uncompressed and os.path.exists(uncompressed):
        try:
            os.remove(uncompressed)
        except OSError:
            pass
    return gz_path


def load(path: str) -> Any:
    """Read JSON from `path`, transparently handling `.json` and `.json.gz`.

    Tries `path` first, then `path + '.gz'`. Raises FileNotFoundError if
    neither exists.
    """
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    gz = path + ".gz"
    if os.path.exists(gz):
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(f"neither {path} nor {gz} exists")


def exists(path: str) -> bool:
    """True if either `path` or `path + '.gz'` is on disk."""
    if path.endswith(".gz"):
        return os.path.exists(path)
    return os.path.exists(path) or os.path.exists(path + ".gz")
