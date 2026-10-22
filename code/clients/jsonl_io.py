"""Transparent reader for streaming JSONL data stored as gzip *or* xz.

The world_cup pipeline writes ``.jsonl.gz`` live (gzip appends cheaply), and a
nightly job recompresses *aged* files to ``.jsonl.xz`` (xz/lzma) for maximum,
lossless compression. xz is transformational on "snapshot-of-a-growing-payload"
streams (series comments, ESPN summary) where each record re-includes prior
content, so the redundancy spans *megabytes* — far beyond gzip's 32 KB window
(measured: ~45-80x smaller than gzip-9 on those streams).

These helpers let readers consume either format interchangeably, which makes the
format migration backward-compatible: ``open_jsonl(path)`` reads the ``.gz``
today and the ``.xz`` after recompression with no caller change. Pure standard
library (``gzip`` + ``lzma``); no third-party dependency.

Typical migrations::

    gzip.open(p, "rt")                  -> open_jsonl(p, "rt")
    gzip.open(p)              # binary  -> open_jsonl(p, "rb")
    glob.glob(f"{d}/x*.jsonl.gz")       -> glob_jsonl(f"{d}/x*.jsonl.gz")
    os.path.exists(f"{d}/x.jsonl.gz")   -> exists_jsonl(f"{d}/x.jsonl.gz")

Write paths are intentionally NOT covered: the live supervisor keeps writing
``.jsonl.gz`` via its own append helper, and only the recompressor produces
``.jsonl.xz``.
"""
from __future__ import annotations

import glob as _glob
import gzip
import lzma
import os
from typing import IO

# Compression extensions we understand, in preference order (newest format first).
_EXTS = (".xz", ".gz")


def _strip_known_ext(path: str) -> tuple[str, str | None]:
    """Split a trailing known compression extension off ``path``.

    Returns ``(base_without_ext, ext)`` where ``ext`` is ``None`` if ``path``
    does not end in a recognised compression extension.
    """
    for ext in _EXTS:
        if path.endswith(ext):
            return path[: -len(ext)], ext
    return path, None


def resolve_jsonl(path: str) -> str:
    """Return the on-disk variant of ``path``, transparently handling .gz/.xz.

    When a compressed path is requested, the **preferred** extension wins if
    several variants coexist: ``.xz`` (a recompressed original) is chosen over a
    stray ``.gz`` sibling. This matters because an out-of-band process can drop a
    ``.gz`` next to the canonical ``.xz`` (e.g. a mis-matched re-promotion writing
    a fresh ``.gz`` beside an already-recompressed ``.xz``); we must read the
    ``.xz`` original, not the stray ``.gz``. Falls back to ``path`` unchanged if
    nothing exists, so the caller's ``open`` raises a familiar ``FileNotFoundError``.
    """
    base, ext = _strip_known_ext(path)
    if ext is not None:
        # Compressed path requested: prefer .xz over .gz when both exist.
        for e in _EXTS:  # (".xz", ".gz")
            cand = base + e
            if os.path.exists(cand):
                return cand
        return path
    # Bare ``.jsonl`` (or unrecognised): exact path first, then sibling variants.
    if os.path.exists(path):
        return path
    for e in _EXTS:
        cand = base + e
        if os.path.exists(cand):
            return cand
    return path


def exists_jsonl(path: str) -> bool:
    """True if ``path`` or a sibling ``.gz``/``.xz`` variant is on disk."""
    if os.path.exists(path):
        return True
    base, _ = _strip_known_ext(path)
    return any(os.path.exists(base + ext) for ext in _EXTS)


def open_jsonl(path: str, mode: str = "rt", encoding: str | None = "utf-8",
               errors: str | None = None) -> IO:
    """Open a ``.jsonl[.gz|.xz]`` file transparently, picking the codec by
    extension of whichever variant exists on disk.

    ``path`` may name either the ``.gz`` or ``.xz`` variant (or a bare
    ``.jsonl``). Text mode (``"rt"``) by default; pass ``mode="rb"`` to match a
    caller that used bare ``gzip.open(path)``. ``encoding``/``errors`` are only
    forwarded for text modes.
    """
    resolved = resolve_jsonl(path)
    if resolved.endswith(".gz"):
        opener = gzip.open
    elif resolved.endswith(".xz"):
        opener = lzma.open
    else:
        opener = open
    if "b" in mode:
        return opener(resolved, mode)
    kwargs: dict = {"encoding": encoding}
    if errors is not None:
        kwargs["errors"] = errors
    return opener(resolved, mode, **kwargs)


def glob_jsonl(pattern: str) -> list[str]:
    """Glob a ``.jsonl[.gz|.xz]`` pattern, matching *both* compressions.

    Accepts a pattern ending in ``.gz``, ``.xz`` (e.g. ``*.jsonl.gz``) or a bare
    ``.jsonl``. Returns the union across ``.gz``/``.xz``, de-duplicated by the
    uncompressed stem (preferring ``.xz`` when both exist, which can happen
    transiently mid-recompress), sorted by stem for deterministic ordering.
    """
    base, _ = _strip_known_ext(pattern)
    by_stem: dict[str, str] = {}
    # Iterate least-preferred first so the preferred extension wins on overwrite.
    for ext in reversed(_EXTS):  # (".gz", ".xz") -> .xz overwrites .gz
        for p in _glob.glob(base + ext):
            stem, _ = _strip_known_ext(p)
            by_stem[stem] = p
    return [by_stem[stem] for stem in sorted(by_stem)]
