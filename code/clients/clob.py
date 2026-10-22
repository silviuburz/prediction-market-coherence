"""
Client for Polymarket's clob.polymarket.com endpoints.
Public, no authentication required for read endpoints.

This is the execution layer — full bid/ask depth, price history, market
metadata. The bot's `polymarket_client.py` has its own version; this is the
analysis-package version with consistent retry / rate-limit semantics.
"""

import requests
import time
import logging
import threading
from typing import Optional

from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

CLOB_API_BASE = "https://clob.polymarket.com"

_limiter = RateLimiter(min_delay=0.2)

MAX_RETRIES = 3

_counter_lock = threading.Lock()
_clob_429_count = 0
_clob_5xx_count = 0


def get_throttle_counters() -> dict[str, int]:
    """Snapshot of cumulative throttle/error counters since process start."""
    with _counter_lock:
        return {"clob_429": _clob_429_count, "clob_5xx": _clob_5xx_count}


def _bump_429() -> None:
    global _clob_429_count
    with _counter_lock:
        _clob_429_count += 1


def _bump_5xx() -> None:
    global _clob_5xx_count
    with _counter_lock:
        _clob_5xx_count += 1


def _get(url: str, params: dict | None = None, timeout: int = 15) -> requests.Response:
    """GET with rate limiting, retries, and backoff on 429/5xx."""
    for attempt in range(MAX_RETRIES + 1):
        _limiter.wait()
        try:
            short_url = url.replace(CLOB_API_BASE, "")
            logger.debug("GET %s params=%s attempt=%d", short_url, params, attempt)
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                _bump_429()
                delay = _limiter.backoff_delay(attempt)
                logger.warning("Rate limited (429). Backing off %.1fs (attempt %d)", delay, attempt + 1)
                time.sleep(delay)
                continue
            if resp.status_code >= 500:
                _bump_5xx()
                delay = _limiter.backoff_delay(attempt)
                logger.warning("Server error %d. Backing off %.1fs (attempt %d)", resp.status_code, delay, attempt + 1)
                time.sleep(delay)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.ConnectionError as e:
            if attempt < MAX_RETRIES:
                delay = _limiter.backoff_delay(attempt)
                logger.warning("Connection error: %s. Retrying in %.1fs", e, delay)
                time.sleep(delay)
            else:
                raise
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                delay = _limiter.backoff_delay(attempt)
                logger.warning("Timeout. Retrying in %.1fs", delay)
                time.sleep(delay)
            else:
                raise
    raise RuntimeError(f"Max retries exceeded for {url}")


# ── Order Book ────────────────────────────────────────────────

def get_order_book(token_id: str) -> dict:
    """Full order book for a CLOB token. Returns dict with `bids` and `asks`
    (each a list of `{"price": str, "size": str}` levels).
    """
    resp = _get(f"{CLOB_API_BASE}/book", params={"token_id": token_id})
    return resp.json()


def get_order_books(token_ids: list[str]) -> list[dict]:
    """Bulk fetch for multiple token IDs (POST /books).

    Respects 429/5xx with the same exponential backoff as `_get`. Never
    cascades to per-token fetches on rate-limit errors (which would amplify
    the request rate at the worst possible time); only falls back on
    connection/timeout errors and only as a last resort.

    On hard failure returns a list of error dicts (one per input token) so
    callers can keep going and log/skip rather than crash.
    """
    if not token_ids:
        return []
    url = f"{CLOB_API_BASE}/books"
    payload = [{"token_id": t} for t in token_ids]
    for attempt in range(MAX_RETRIES + 1):
        _limiter.wait()
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                # Polymarket POST /books may REORDER its response relative to the
                # request AND OMIT tokens with no/empty book, so we must NOT rely
                # on positional alignment (callers zip our return onto their batch).
                # Re-key each book by its own asset_id and return in the REQUESTED
                # token order; any token the API dropped is filled with an error
                # placeholder so the 1:1 input/output contract always holds.
                by_id = {}
                if isinstance(data, list):
                    for b in data:
                        if isinstance(b, dict) and b.get("asset_id") is not None:
                            by_id[str(b["asset_id"])] = b
                missing = sum(1 for t in token_ids if str(t) not in by_id)
                if missing:
                    logger.warning("Bulk /books returned %d/%d tokens; %d missing (filled with error dicts)",
                                   len(by_id), len(token_ids), missing)
                return [by_id.get(str(t), {"error": "missing_in_books_response", "token_id": t})
                        for t in token_ids]
            if resp.status_code == 429:
                _bump_429()
                delay = _limiter.backoff_delay(attempt)
                logger.warning("Bulk /books rate limited (429, %d toks). Backing off %.1fs (attempt %d)",
                               len(token_ids), delay, attempt + 1)
                time.sleep(delay)
                continue
            if resp.status_code >= 500:
                _bump_5xx()
                delay = _limiter.backoff_delay(attempt)
                logger.warning("Bulk /books server error %d (%d toks). Backing off %.1fs (attempt %d)",
                               resp.status_code, len(token_ids), delay, attempt + 1)
                time.sleep(delay)
                continue
            logger.warning("Bulk /books unexpected status %d (%d toks); skipping batch",
                           resp.status_code, len(token_ids))
            return [{"error": f"http_{resp.status_code}", "token_id": t} for t in token_ids]
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < MAX_RETRIES:
                delay = _limiter.backoff_delay(attempt)
                logger.warning("Bulk /books transient error: %s. Retrying in %.1fs", e, delay)
                time.sleep(delay)
            else:
                logger.warning("Bulk /books exhausted retries (%s); returning errors for %d toks",
                               e, len(token_ids))
                return [{"error": f"transport_{type(e).__name__}", "token_id": t} for t in token_ids]
    logger.warning("Bulk /books exhausted retries (rate-limit/5xx); returning errors for %d toks",
                   len(token_ids))
    return [{"error": "retries_exhausted", "token_id": t} for t in token_ids]


# ── Price ──────────────────────────────────────────────────────

def get_price(token_id: str, side: str = "buy") -> dict:
    """Best price (top of book) on the given side ("buy" or "sell")."""
    resp = _get(f"{CLOB_API_BASE}/price", params={"token_id": token_id, "side": side})
    return resp.json()


# ── Price History ─────────────────────────────────────────────

VALID_INTERVALS = ("1m", "5m", "15m", "1h", "6h", "1d", "1w", "max")


def get_price_history(
    market: str,
    interval: str = "max",
    fidelity: Optional[int] = None,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
) -> dict:
    """Fetch the price-history time series for a CLOB token.

    Args:
        market: CLOB token id (decimal string).
        interval: One of '1m', '5m', '15m', '1h', '6h', '1d', '1w', 'max'.
                  'max' returns the full history (downsampled by gamma).
        fidelity: Optional minutes-per-bucket override.
        start_ts, end_ts: Optional Unix-second bounds (omit when interval='max').

    Returns:
        Dict with `history`: list of `{"t": unix_seconds, "p": price}`.
    """
    if interval not in VALID_INTERVALS:
        raise ValueError(f"interval must be one of {VALID_INTERVALS}, got {interval!r}")
    params: dict = {"market": market, "interval": interval}
    if fidelity is not None:
        params["fidelity"] = fidelity
    if start_ts is not None:
        params["startTs"] = start_ts
    if end_ts is not None:
        params["endTs"] = end_ts
    resp = _get(f"{CLOB_API_BASE}/prices-history", params=params)
    return resp.json()


# ── Markets (metadata) ────────────────────────────────────────

def get_clob_market(condition_id: str) -> dict:
    """Fetch CLOB-side market metadata for a condition id."""
    resp = _get(f"{CLOB_API_BASE}/markets/{condition_id}")
    return resp.json()
