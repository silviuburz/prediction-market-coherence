"""
Client for Polymarket's gamma-api.polymarket.com endpoints.
Public, no authentication required.

Gamma exposes Markets / Events / Comments / Holders / Open-Interest. It's the
discovery and metadata layer (CLOB is the execution layer). All world-cup
collectors read from gamma to enumerate markets and capture top-of-book +
metadata.
"""

import requests
import time
import logging
from typing import Optional

from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

GAMMA_API_BASE = "https://gamma-api.polymarket.com"

_limiter = RateLimiter(min_delay=0.5)

MAX_RETRIES = 3
DEFAULT_PAGE_LIMIT = 500  # gamma's hard cap for /markets and /events


def _get(url: str, params: dict | None = None, timeout: int = 20) -> requests.Response:
    """GET with rate limiting, retries, and backoff on 429/5xx."""
    for attempt in range(MAX_RETRIES + 1):
        _limiter.wait()
        try:
            short_url = url.replace(GAMMA_API_BASE, "")
            logger.debug("GET %s params=%s attempt=%d", short_url, params, attempt)
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                delay = _limiter.backoff_delay(attempt)
                logger.warning("Rate limited (429). Backing off %.1fs (attempt %d)", delay, attempt + 1)
                time.sleep(delay)
                continue
            if resp.status_code >= 500:
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


# ── Markets ───────────────────────────────────────────────────

def get_markets(
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
    active: Optional[bool] = None,
    closed: Optional[bool] = None,
    archived: Optional[bool] = None,
    order: str = "volume24hr",
    ascending: bool = False,
    q: Optional[str] = None,
    extra: Optional[dict] = None,
) -> list[dict]:
    """Single page of /markets. Use `iter_markets` for full pagination."""
    params: dict = {
        "limit": min(limit, DEFAULT_PAGE_LIMIT),
        "offset": offset,
        "order": order,
        "ascending": str(ascending).lower(),
    }
    if active is not None:
        params["active"] = str(active).lower()
    if closed is not None:
        params["closed"] = str(closed).lower()
    if archived is not None:
        params["archived"] = str(archived).lower()
    if q:
        params["_q"] = q
    if extra:
        params.update(extra)
    resp = _get(f"{GAMMA_API_BASE}/markets", params=params)
    return resp.json()


def iter_markets(
    page_size: int = DEFAULT_PAGE_LIMIT,
    max_pages: int = 100,
    **filters,
):
    """Yield all markets matching `filters`, paginating through /markets."""
    offset = 0
    for _ in range(max_pages):
        batch = get_markets(limit=page_size, offset=offset, **filters)
        if not batch:
            return
        for m in batch:
            yield m
        if len(batch) < page_size:
            return
        offset += page_size


def get_market_by_id(market_id: str) -> dict:
    """Fetch a single market by its numeric id."""
    resp = _get(f"{GAMMA_API_BASE}/markets/{market_id}")
    return resp.json()


def get_markets_by_condition_ids(condition_ids: list[str], closed: bool = True) -> list[dict]:
    """Fetch markets by condition ID using Gamma's repeated-parameter syntax.

    Gamma declares ``condition_ids`` as an array. Its production endpoint
    accepts repeated query parameters but not a comma-separated value.
    """
    normalized = list(dict.fromkeys(str(value).lower() for value in condition_ids))
    if not normalized:
        return []
    if len(normalized) > DEFAULT_PAGE_LIMIT:
        raise ValueError(
            f"condition ID batch exceeds Gamma limit: {len(normalized)} > {DEFAULT_PAGE_LIMIT}"
        )
    params = [("condition_ids", condition_id) for condition_id in normalized]
    params.extend([
        ("closed", str(closed).lower()),
        ("limit", str(len(normalized))),
    ])
    resp = _get(f"{GAMMA_API_BASE}/markets", params=params)
    return resp.json()


# ── Events ────────────────────────────────────────────────────

def get_events(
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
    active: Optional[bool] = None,
    closed: Optional[bool] = None,
    archived: Optional[bool] = None,
    order: str = "volume24hr",
    ascending: bool = False,
    q: Optional[str] = None,
    extra: Optional[dict] = None,
) -> list[dict]:
    """Single page of /events."""
    params: dict = {
        "limit": min(limit, DEFAULT_PAGE_LIMIT),
        "offset": offset,
        "order": order,
        "ascending": str(ascending).lower(),
    }
    if active is not None:
        params["active"] = str(active).lower()
    if closed is not None:
        params["closed"] = str(closed).lower()
    if archived is not None:
        params["archived"] = str(archived).lower()
    if q:
        params["_q"] = q
    if extra:
        params.update(extra)
    resp = _get(f"{GAMMA_API_BASE}/events", params=params)
    return resp.json()


def iter_events(
    page_size: int = DEFAULT_PAGE_LIMIT,
    max_pages: int = 100,
    **filters,
):
    """Yield all events matching `filters`, paginating through /events."""
    offset = 0
    for _ in range(max_pages):
        batch = get_events(limit=page_size, offset=offset, **filters)
        if not batch:
            return
        for ev in batch:
            yield ev
        if len(batch) < page_size:
            return
        offset += page_size


def get_event_by_id(event_id: str) -> dict:
    """Fetch a single event by its numeric id."""
    resp = _get(f"{GAMMA_API_BASE}/events/{event_id}")
    return resp.json()


# ── Holders ───────────────────────────────────────────────────

# NOTE: as of 2026-06-10, the /holders endpoint actually lives on
# data-api.polymarket.com (NOT gamma-api). Gamma returns 404 for /holders.
# We hit data-api directly here — the response shape is
#   [{"token": "...", "holders": [{"proxyWallet": ..., "amount": ...}, ...]}]
# so a single conditionId query returns holders for ALL tokens in that market.
_DATA_API_BASE = "https://data-api.polymarket.com"


def get_holders(market: str, limit: int = 100) -> list[dict]:
    """Top holders for a market (by `conditionId`)."""
    params = {"market": market, "limit": limit}
    # Use a direct request since this hits a different host than gamma's _get.
    for attempt in range(MAX_RETRIES + 1):
        _limiter.wait()
        try:
            resp = requests.get(f"{_DATA_API_BASE}/holders", params=params, timeout=20)
            if resp.status_code == 429:
                time.sleep(_limiter.backoff_delay(attempt))
                continue
            if resp.status_code >= 500:
                time.sleep(_limiter.backoff_delay(attempt))
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            if attempt < MAX_RETRIES:
                time.sleep(_limiter.backoff_delay(attempt))
            else:
                raise
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                time.sleep(_limiter.backoff_delay(attempt))
            else:
                raise
    raise RuntimeError(f"Max retries exceeded for /holders?market={market}")


# ── Open Interest ─────────────────────────────────────────────

def get_open_interest(market: str) -> dict:
    """Current open interest for a market."""
    params = {"market": market}
    resp = _get(f"{GAMMA_API_BASE}/open-interest", params=params)
    return resp.json()


# ── Comments ──────────────────────────────────────────────────

COMMENTS_PAGE_LIMIT = 500


def get_comments(
    parent_entity_id: str,
    parent_entity_type: str = "Event",
    limit: int = COMMENTS_PAGE_LIMIT,
    offset: int = 0,
    order: str = "createdAt",
    ascending: bool = True,
    get_positions: bool = False,
    get_reports: bool = False,
    holders_only: bool = False,
) -> list[dict]:
    """Single page of comments for an Event (or other parent entity).

    Polymarket requires both `parent_entity_id` and `parent_entity_type`.
    For per-event comments, the type is "Event".

    `get_positions=True` enriches each comment's `profile` with a `positions`
    array ([{tokenId, positionSize}, ...]) describing the author's CURRENT
    holdings at fetch time -- captured live, this links what a wallet says to
    what it holds without reconstructing from the trade tape. `get_reports`
    adds moderation-report metadata; `holders_only` filters to authors who
    currently hold a position in the parent entity.
    """
    params = {
        "parent_entity_id": parent_entity_id,
        "parent_entity_type": parent_entity_type,
        "limit": min(limit, COMMENTS_PAGE_LIMIT),
        "offset": offset,
        "order": order,
        "ascending": str(ascending).lower(),
    }
    if get_positions:
        params["get_positions"] = "true"
    if get_reports:
        params["get_reports"] = "true"
    if holders_only:
        params["holders_only"] = "true"
    resp = _get(f"{GAMMA_API_BASE}/comments", params=params)
    return resp.json()


# Polymarket's /comments uses OFFSET pagination over a list that shifts as new
# comments arrive, so any single ordered pass silently DROPS rows. Measured on
# the WC series (id 11433, reported commentCount=870): a createdAt-ascending
# pass returns only ~678; unioning these four orderings recovers ~827. The
# residual (~43) are deleted comments counted in commentCount but no longer
# served. Page each ordering and union by id to close the gap.
DEFAULT_COMMENT_ORDERS: list[tuple[str, bool]] = [
    ("createdAt", True),
    ("createdAt", False),
    ("reactionCount", True),
    ("reactionCount", False),
]


def iter_comments(
    parent_entity_id: str,
    parent_entity_type: str = "Event",
    page_size: int = COMMENTS_PAGE_LIMIT,
    max_pages: int = 200,
    get_positions: bool = False,
    get_reports: bool = False,
    orders: Optional[list[tuple[str, bool]]] = None,
):
    """Yield all comments for a parent entity, paginating.

    NOTE: Polymarket's /comments endpoint sometimes returns MORE rows than
    `limit` (e.g. limit=100 -> 149 rows). Advancing by `page_size` instead
    of `len(batch)` would silently skip items. We use `offset += len(batch)`
    and only stop on an empty page.

    Offset pagination is also UNSTABLE: a single ordering drops a different
    subset of comments each pass. Pass `orders` (a list of (order, ascending)
    tuples, e.g. `DEFAULT_COMMENT_ORDERS`) to page each ordering and yield the
    union (deduped by id), recovering rows a single pass misses. When `orders`
    is None we do one createdAt-ascending pass (legacy behaviour).

    `get_positions=True` enriches each author profile with their current
    on-chain positions (see `get_comments`).
    """
    order_list = orders if orders is not None else [("createdAt", True)]
    seen_ids: set = set()
    for order, ascending in order_list:
        offset = 0
        for _ in range(max_pages):
            batch = get_comments(
                parent_entity_id=parent_entity_id,
                parent_entity_type=parent_entity_type,
                limit=page_size,
                offset=offset,
                order=order,
                ascending=ascending,
                get_positions=get_positions,
                get_reports=get_reports,
            )
            if not batch:
                break
            for c in batch:
                cid = c.get("id")
                if cid and cid in seen_ids:
                    continue
                if cid:
                    seen_ids.add(cid)
                yield c
            offset += len(batch)


# ── Public Profile ────────────────────────────────────────────

def get_public_profile(address: str) -> dict:
    """Fetch the public profile metadata for a wallet."""
    params = {"address": address}
    resp = _get(f"{GAMMA_API_BASE}/public-profile", params=params)
    return resp.json()
