"""
Client for Polymarket's data-api.polymarket.com endpoints.
Public, no authentication required.

Provides access to:
- Leaderboard rankings
- Trade histories
- Open / closed positions
- User activity
- Profiles (via gamma-api)
"""

import requests
import time
import logging
from typing import Optional

from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

DATA_API_BASE = "https://data-api.polymarket.com"
GAMMA_API_BASE = "https://gamma-api.polymarket.com"

_limiter = RateLimiter(min_delay=0.2)

MAX_RETRIES = 5
RETRYABLE_STATUS_CODES = {408, 425, 429}


def set_request_delay(seconds: float) -> None:
    """Set the process-wide minimum interval between Data API request starts."""
    if seconds < 0.1:
        raise ValueError("request delay must be at least 0.1 seconds")
    _limiter.min_delay = float(seconds)


def _get(url: str, params: dict | None = None, timeout: int = 20) -> requests.Response:
    """Make a GET request with rate limiting, retries, and backoff."""
    for attempt in range(MAX_RETRIES + 1):
        _limiter.wait()
        try:
            # Log at DEBUG level so we can see HTTP activity when needed
            short_url = url.replace(DATA_API_BASE, "").replace(GAMMA_API_BASE, "")
            logger.debug("GET %s params=%s attempt=%d", short_url, params, attempt)
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code in RETRYABLE_STATUS_CODES or resp.status_code >= 500:
                if attempt < MAX_RETRIES:
                    delay = _limiter.backoff_delay(attempt)
                    logger.warning(
                        "Retryable HTTP %d. Backing off %.1fs (attempt %d)",
                        resp.status_code,
                        delay,
                        attempt + 1,
                    )
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


# ── Leaderboard ───────────────────────────────────────────────

LEADERBOARD_CATEGORIES = [
    "OVERALL", "POLITICS", "SPORTS", "CRYPTO", "CULTURE",
    "MENTIONS", "WEATHER", "ECONOMICS", "TECH", "FINANCE",
]

LEADERBOARD_PERIODS = ["DAY", "WEEK", "MONTH", "ALL"]


def get_leaderboard(
    category: str = "OVERALL",
    time_period: str = "ALL",
    order_by: str = "PNL",
    limit: int = 50,
    offset: int = 0,
    user: Optional[str] = None,
) -> list[dict]:
    """Fetch trader leaderboard rankings.

    Returns list of dicts with: rank, proxyWallet, userName, vol, pnl,
    profileImage, xUsername, verifiedBadge.
    """
    params = {
        "category": category,
        "timePeriod": time_period,
        "orderBy": order_by,
        "limit": min(limit, 50),
        "offset": offset,
    }
    if user:
        params["user"] = user
    resp = _get(f"{DATA_API_BASE}/v1/leaderboard", params=params)
    return resp.json()


# ── Trades ────────────────────────────────────────────────────

def get_trades(
    user: str,
    limit: int = 1000,
    offset: int = 0,
    taker_only: bool = True,
    side: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> list[dict]:
    """Fetch trades for a user.

    Response fields per trade: proxyWallet, side, asset, conditionId,
    size, price, timestamp, title, slug, eventSlug, outcome, outcomeIndex,
    transactionHash, name, pseudonym.

    NOTE on caps (verified May 8, 2026):
    - `limit` is honored up to 1,000; values above are silently capped at 1,000.
    - The API enforces a hard ceiling at offset=3000 (returns
      HTTP 400 with `{"error": "max historical activity offset of 3000 exceeded"}`).
    - Sorting is fixed to timestamp DESC; sortDirection is not honored.
    - Combined: at most ~4,000 trades returnable per user via this endpoint.
      For wallets with more lifetime trades, use the per-market workaround
      (R10 in REBUILD_PLAN.md) — query /trades?market=conditionId for each
      market the wallet held and filter by wallet.
    """
    params = {
        "user": user,
        "limit": min(limit, 1000),
        "offset": offset,
        "takerOnly": str(taker_only).lower(),
    }
    if side:
        params["side"] = side
    if start is not None:
        params["start"] = int(start)
    if end is not None:
        params["end"] = int(end)
    resp = _get(f"{DATA_API_BASE}/trades", params=params)
    return resp.json()


def get_all_trades_in_window(
    user: str,
    start: int,
    end: int,
    taker_only: bool = True,
    side: Optional[str] = None,
    page_size: int = 1000,
    min_window_seconds: int = 1,
) -> tuple[list[dict], dict]:
    """Fetch a complete bounded trade window by recursively splitting it.

    The public API honors ``start`` and ``end`` but silently caps one response
    at 1,000 rows. A saturated interval is bisected until every leaf response
    is below the cap. Rows are deduplicated because timestamp bounds are
    inclusive and multiple fills may share a transaction hash.

    Returns ``(trades, metadata)``. ``metadata['complete']`` is false only when
    a one-second leaf still saturates, because no timestamp-only partition can
    prove completeness in that case.
    """
    if start < 0 or end < 0:
        raise ValueError("start and end must be non-negative epoch seconds")
    if end < start:
        raise ValueError("end must be greater than or equal to start")
    if page_size < 1 or page_size > 1000:
        raise ValueError("page_size must be in [1, 1000]")
    if min_window_seconds < 1:
        raise ValueError("min_window_seconds must be positive")

    pending: list[tuple[int, int]] = [(int(start), int(end))]
    rows: list[dict] = []
    saturated_leaves: list[dict] = []
    failed_leaves: list[dict] = []
    retryable_error_splits: list[dict] = []
    requests_made = 0
    max_depth = 0
    windows: list[dict] = []

    while pending:
        window_start, window_end = pending.pop()
        depth = 0
        width = max(0, window_end - window_start)
        if width:
            original_width = max(1, end - start)
            depth = max(0, (original_width // max(1, width)).bit_length() - 1)
        max_depth = max(max_depth, depth)

        try:
            batch = get_trades(
                user,
                limit=page_size,
                offset=0,
                taker_only=taker_only,
                side=side,
                start=window_start,
                end=window_end,
            )
        except (requests.exceptions.HTTPError, requests.exceptions.Timeout) as error:
            status_code = (
                error.response.status_code
                if isinstance(error, requests.exceptions.HTTPError)
                and error.response is not None
                else None
            )
            if status_code not in (None, 408):
                raise
            if window_end - window_start < min_window_seconds:
                requests_made += 1
                failed_leaves.append({
                    "start": window_start,
                    "end": window_end,
                    "status_code": status_code,
                    "error": type(error).__name__,
                })
                continue
            midpoint = window_start + (window_end - window_start) // 2
            if midpoint < window_start or midpoint >= window_end:
                requests_made += 1
                failed_leaves.append({
                    "start": window_start,
                    "end": window_end,
                    "status_code": status_code,
                    "error": type(error).__name__,
                })
                continue
            requests_made += 1
            retryable_error_splits.append({
                "start": window_start,
                "end": window_end,
                "status_code": status_code,
            })
            pending.append((midpoint + 1, window_end))
            pending.append((window_start, midpoint))
            if requests_made % 25 == 0:
                logger.info(
                    "[trades] %s requests=%d pending=%d rows=%d timeout_splits=%d",
                    user[:12],
                    requests_made,
                    len(pending),
                    len(rows),
                    len(retryable_error_splits),
                )
            continue
        requests_made += 1
        windows.append({
            "start": window_start,
            "end": window_end,
            "n": len(batch),
        })
        if requests_made % 25 == 0:
            logger.info(
                "[trades] %s requests=%d pending=%d rows=%d timeout_splits=%d",
                user[:12],
                requests_made,
                len(pending),
                len(rows),
                len(retryable_error_splits),
            )

        if len(batch) < page_size:
            rows.extend(batch)
            continue

        if window_end - window_start < min_window_seconds:
            rows.extend(batch)
            saturated_leaves.append({
                "start": window_start,
                "end": window_end,
                "n": len(batch),
            })
            continue

        midpoint = window_start + (window_end - window_start) // 2
        if midpoint < window_start or midpoint >= window_end:
            rows.extend(batch)
            saturated_leaves.append({
                "start": window_start,
                "end": window_end,
                "n": len(batch),
            })
            continue
        pending.append((midpoint + 1, window_end))
        pending.append((window_start, midpoint))

    n_rows_before_dedup = len(rows)
    seen_keys: set[tuple] = set()
    write_index = 0
    for trade in rows:
        key = (
            trade.get("transactionHash"),
            trade.get("asset"),
            trade.get("side"),
            trade.get("timestamp"),
            trade.get("price"),
            trade.get("size"),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        rows[write_index] = trade
        write_index += 1
    del rows[write_index:]
    del seen_keys
    rows.sort(key=lambda trade: int(trade.get("timestamp", 0)))
    trades = rows
    metadata = {
        "user": user,
        "start": int(start),
        "end": int(end),
        "taker_only": taker_only,
        "side": side,
        "page_size": page_size,
        "requests_made": requests_made,
        "windows_queried": windows,
        "n_rows_before_dedup": n_rows_before_dedup,
        "n_rows": len(trades),
        "n_duplicates_removed": n_rows_before_dedup - len(trades),
        "max_split_depth": max_depth,
        "retryable_error_splits": retryable_error_splits,
        "saturated_leaves": saturated_leaves,
        "failed_leaves": failed_leaves,
        "complete": not saturated_leaves and not failed_leaves,
    }
    return trades, metadata


def get_all_trades(user: str, taker_only: bool = True, page_size: int = 1000) -> list[dict]:
    """Paginate through all reachable trades for a user via /trades?user=.

    The API caps at offset=3000 (see `get_trades` docstring), so this returns
    at most ~4,000 trades per user — the user's most recent ~4,000 trades.
    For complete history on high-activity wallets, use the per-market
    workaround (R10 in REBUILD_PLAN.md).
    """
    all_trades = []
    offset = 0
    while True:
        try:
            batch = get_trades(user, limit=page_size, offset=offset, taker_only=taker_only)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 400:
                # Either we hit the offset=3000 ceiling, or the user has
                # genuinely no more trades. Either way, stop paginating.
                logger.debug("Got 400 at offset %d for user %s — likely API ceiling", offset, user[:12])
                break
            raise
        if not batch:
            break
        all_trades.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
        # The API hard-caps at offset=3000; one more page (offset=3000, limit=1000)
        # is the last possible call before HTTP 400.
        if offset > 3000:
            logger.info("Reached /trades?user= ceiling at %d trades for user %s "
                         "(API hard-caps at offset=3000)", len(all_trades), user[:12])
            break
    return all_trades


def get_trades_by_market(
    market: str,
    limit: int = 500,
    offset: int = 0,
    taker_only: bool = False,
) -> list[dict]:
    """Fetch trades for a market via condition_id (NOT token_id).

    Verified Jun 12 2026: Goldsky's orderbook-subgraph returns ~0 trades
    for active WC markets (likely indexes an old/different orderbook
    contract). The data-API's /trades?market=<conditionId> returns the
    authoritative truth. Use this for incremental polling.

    Sort: timestamp DESC (most recent first). No since/after filter is
    honored; caller must filter locally on timestamp > cursor.
    Pagination: stable; HTTP 400 at offset>3000.

    Schema per trade: proxyWallet, side (BUY/SELL), asset (token_id),
    conditionId, size (USDC), price, timestamp, title, slug, outcome,
    outcomeIndex, transactionHash, name, pseudonym.
    """
    params = {
        "market": market,
        "limit": min(limit, 500),
        "offset": offset,
        "takerOnly": str(taker_only).lower(),
    }
    resp = _get(f"{DATA_API_BASE}/trades", params=params)
    return resp.json()


def get_all_trades_by_market(
    market: str,
    since_ts: int = 0,
    taker_only: bool = False,
    page_size: int = 500,
) -> list[dict]:
    """Paginate trades for a market, stopping when we cross `since_ts` or hit
    the API's offset=3000 ceiling. Returns oldest-first (chronological).

    Polymarket's data-API returns trades DESC by timestamp; we stop
    paginating once a page's oldest trade has `timestamp < since_ts`, then
    sort the accumulated result ascending.
    """
    out: list[dict] = []
    offset = 0
    while True:
        try:
            batch = get_trades_by_market(market, limit=page_size, offset=offset, taker_only=taker_only)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 400:
                logger.debug("Got 400 at offset %d for market %s — likely API ceiling", offset, market[:16])
                break
            raise
        if not batch:
            break
        # batch is newest-first; keep only the ones strictly newer than cursor
        new_trades = [t for t in batch if int(t.get("timestamp", 0)) > since_ts]
        out.extend(new_trades)
        # If the OLDEST trade in this page is <= cursor, we've crossed the
        # boundary; no need to paginate further.
        oldest_ts = min(int(t.get("timestamp", 0)) for t in batch)
        if oldest_ts <= since_ts:
            break
        if len(batch) < page_size:
            break
        offset += page_size
        if offset > 3000:
            logger.warning("Reached /trades?market= ceiling at %d trades for market %s "
                           "(API hard-caps at offset=3000); some older trades missed", offset, market[:16])
            break
    # Return in chronological order
    out.sort(key=lambda t: int(t.get("timestamp", 0)))
    return out


# ── Positions ─────────────────────────────────────────────────

def get_positions(
    user: str,
    limit: int = 500,
    offset: int = 0,
    size_threshold: float = 1,
) -> list[dict]:
    """Fetch current open positions for a user."""
    params = {
        "user": user,
        "limit": min(limit, 500),
        "offset": offset,
        "sizeThreshold": size_threshold,
        "sortBy": "CURRENT",
        "sortDirection": "DESC",
    }
    resp = _get(f"{DATA_API_BASE}/positions", params=params)
    return resp.json()


def get_all_positions(user: str, max_positions: int = 50000) -> list[dict]:
    """Paginate through all open positions for a user.

    Args:
        max_positions: Safety cap to prevent runaway pagination on a buggy
            response. The Polymarket API itself does not enforce a hard
            ceiling at 5000 — mega-whales can have 5000–6000+ positions and
            we paginate through all of them. (Verified May 8, 2026 against
            wallet 0x8e9eedf20d which returned 5862 positions cleanly.)
            Set this to a value comfortably larger than any whale we expect.
    """
    all_positions = []
    offset = 0
    while True:
        try:
            batch = get_positions(user, limit=500, offset=offset)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 400:
                logger.warning("Got 400 at offset %d for positions user %s — stopping", offset, user[:12])
                break
            raise
        if not batch:
            break
        all_positions.extend(batch)
        logger.debug("  positions page: offset=%d got=%d total=%d", offset, len(batch), len(all_positions))
        if len(batch) < 500:
            break
        offset += 500
        if len(all_positions) >= max_positions:
            logger.warning("Hit max_positions cap (%d) for positions user %s — truncating", max_positions, user[:12])
            break
    return all_positions


# ── Closed Positions ──────────────────────────────────────────

def get_closed_positions(
    user: str,
    limit: int = 50,
    offset: int = 0,
    sort_by: str = "TIMESTAMP",
    sort_direction: str = "DESC",
) -> list[dict]:
    """Fetch closed (resolved/sold) positions for a user."""
    params = {
        "user": user,
        "limit": min(limit, 50),
        "offset": offset,
        "sortBy": sort_by,
        "sortDirection": sort_direction,
    }
    resp = _get(f"{DATA_API_BASE}/closed-positions", params=params)
    return resp.json()


def get_closed_positions_since(
    user: str,
    since_ts: int,
    through_ts: Optional[int] = None,
    page_size: int = 50,
    max_offset: int = 100000,
) -> tuple[list[dict], dict]:
    """Legacy filter over the wallet-specific closed-position timestamp.

    This field is not a market-resolution clock (WI-211), so this helper must
    not be used to build point-in-time outcome histories. It remains available
    for artifact reproduction and wallet-activity queries only.
    """
    if since_ts < 0:
        raise ValueError("since_ts must be non-negative")
    if through_ts is not None and through_ts < since_ts:
        raise ValueError("through_ts must be greater than or equal to since_ts")
    if page_size < 1 or page_size > 50:
        raise ValueError("page_size must be in [1, 50]")
    if max_offset < 0:
        raise ValueError("max_offset must be non-negative")

    rows: list[dict] = []
    offset = 0
    requests_made = 0
    reached_cutoff = False
    stop_reason = "unknown"

    while offset <= max_offset:
        batch = get_closed_positions(
            user,
            limit=page_size,
            offset=offset,
            sort_by="TIMESTAMP",
            sort_direction="DESC",
        )
        requests_made += 1
        if not batch:
            stop_reason = "empty_page"
            break

        timestamps = []
        for position in batch:
            try:
                timestamp = int(position.get("timestamp", 0))
            except (TypeError, ValueError):
                continue
            timestamps.append(timestamp)
            if timestamp < since_ts:
                continue
            if through_ts is not None and timestamp > through_ts:
                continue
            rows.append(position)

        if timestamps and min(timestamps) < since_ts:
            reached_cutoff = True
            stop_reason = "crossed_cutoff"
            break
        if len(batch) < page_size:
            stop_reason = "short_page"
            break
        offset += page_size
    else:
        stop_reason = "max_offset"

    deduped: dict[tuple, dict] = {}
    for position in rows:
        key = (
            position.get("conditionId"),
            position.get("asset"),
            position.get("outcome"),
            position.get("timestamp"),
        )
        deduped[key] = position
    positions = sorted(
        deduped.values(), key=lambda position: int(position.get("timestamp", 0))
    )
    metadata = {
        "user": user,
        "since_ts": int(since_ts),
        "through_ts": int(through_ts) if through_ts is not None else None,
        "page_size": page_size,
        "requests_made": requests_made,
        "last_offset": offset,
        "stop_reason": stop_reason,
        "reached_cutoff": reached_cutoff,
        "n_rows_before_dedup": len(rows),
        "n_rows": len(positions),
        "n_duplicates_removed": len(rows) - len(positions),
        "complete_for_requested_range": reached_cutoff or stop_reason in {"short_page", "empty_page"},
    }
    return positions, metadata


def closed_position_lifetime_complete(
    direction_metadata: dict[str, dict],
    direction_overlap_rows: int,
) -> bool:
    natural_end_reached = any(
        bool(row.get("natural_end"))
        for row in direction_metadata.values()
    )
    return bool(
        natural_end_reached
        or (
            set(direction_metadata) == {"DESC", "ASC"}
            and direction_overlap_rows > 0
        )
    )


def get_all_closed_positions_with_metadata(
    user: str,
    page_size: int = 50,
    max_positions: int = 50000,
) -> tuple[list[dict], dict]:
    """Collect both reachable tails of a wallet's closed-position history.

    The /closed-positions endpoint has a hard offset cap at ~3,000 (verified
    May 8, 2026: the API silently returns empty pages past offset=3050, no
    error). To recover more history we call it TWICE — once DESC (newest
    first) and once ASC (oldest first) — and merge. This gives us up to
    ~6,100 closed positions per wallet. For wallets with more lifetime
    closed positions, the middle of their history is unreachable via this
    endpoint and requires a Polygon RPC fallback (see R11 in REBUILD_PLAN.md).

    Completeness is established when either direction reaches a natural end,
    or when the ascending and descending traversals overlap. If both directions
    hit the API ceiling without overlap, the unreachable middle is reported and
    ``complete_lifetime_range`` is false.
    """
    if page_size < 1 or page_size > 50:
        raise ValueError("page_size must be in [1, 50]")
    if max_positions < 1:
        raise ValueError("max_positions must be positive")

    seen_keys: set[tuple] = set()
    all_positions: list[dict] = []
    direction_keys: dict[str, set[tuple]] = {}
    direction_metadata: dict[str, dict] = {}
    requests_made = 0
    hit_max_positions = False
    next_progress_rows = 5000

    for direction in ("DESC", "ASC"):
        offset = 0
        stalled_pages = 0  # consecutive pages where we added 0 new uniques
        keys_this_direction: set[tuple] = set()
        stop_reason = "unknown"
        natural_end = False
        previous_page_was_full = False
        while True:
            try:
                batch = get_closed_positions(
                    user, limit=page_size, offset=offset,
                    sort_by="TIMESTAMP", sort_direction=direction,
                )
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 400:
                    stop_reason = "http_400"
                    break
                raise
            requests_made += 1
            logger.debug(
                "[closed_positions] %s direction=%s offset=%d -> %d items (total=%d unique)",
                user[:12], direction, offset, len(batch), len(all_positions),
            )
            if not batch:
                if previous_page_was_full:
                    stop_reason = "empty_after_full_page"
                else:
                    stop_reason = "empty_page"
                    natural_end = True
                break
            page_new = 0
            page_keys: set[tuple] = set()
            for pos in batch:
                # Dedupe by (conditionId, asset, outcome, timestamp) — same
                # closed position may show up in both DESC and ASC sweeps,
                # and the API silently returns duplicate pages past its
                # offset ceiling.
                key = (
                    pos.get("conditionId"),
                    pos.get("asset"),
                    pos.get("outcome"),
                    pos.get("timestamp"),
                )
                keys_this_direction.add(key)
                page_keys.add(key)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                all_positions.append(pos)
                page_new += 1
            if len(all_positions) >= next_progress_rows:
                logger.info(
                    "[closed_positions] %s unique=%d direction=%s offset=%d",
                    user[:12],
                    len(all_positions),
                    direction,
                    offset,
                )
                while next_progress_rows <= len(all_positions):
                    next_progress_rows += 5000
            if (
                direction == "ASC"
                and page_keys & direction_keys.get("DESC", set())
            ):
                stop_reason = "direction_overlap"
                break
            # Stop conditions:
            #  - explicit short page (server signals end)
            #  - 2 consecutive pages with zero new uniques (API is past its
            #    offset ceiling and silently looping; verified May 9, 2026)
            if len(batch) < page_size:
                stop_reason = "short_page"
                natural_end = True
                break
            previous_page_was_full = True
            if page_new == 0:
                stalled_pages += 1
                if stalled_pages >= 2:
                    stop_reason = "stalled_pages"
                    break
            else:
                stalled_pages = 0
            offset += page_size
            if len(all_positions) >= max_positions:
                logger.warning("Hit max_positions cap (%d) for user %s — stopping", max_positions, user[:12])
                stop_reason = "max_positions"
                hit_max_positions = True
                break
        direction_keys[direction] = keys_this_direction
        direction_metadata[direction] = {
            "stop_reason": stop_reason,
            "natural_end": natural_end,
            "last_offset": offset,
            "n_unique_rows_seen": len(keys_this_direction),
        }
        if hit_max_positions:
            break

    overlap = len(direction_keys.get("DESC", set()) & direction_keys.get("ASC", set()))
    complete = closed_position_lifetime_complete(
        direction_metadata,
        overlap,
    )
    all_positions.sort(
        key=lambda position: int(position.get("timestamp", 0) or 0)
    )
    positions = all_positions
    metadata = {
        "user": user,
        "timestamp_semantics": (
            "wallet-specific closed-position field; not used as market resolution time"
        ),
        "page_size": page_size,
        "requests_made": requests_made,
        "n_rows": len(positions),
        "directions": direction_metadata,
        "direction_overlap_rows": overlap,
        "hit_max_positions": hit_max_positions,
        "complete_lifetime_range": complete,
        "missing_middle_history": not complete,
    }
    return positions, metadata


def get_all_closed_positions(user: str, page_size: int = 50, max_positions: int = 50000) -> list[dict]:
    """Backward-compatible list-only wrapper for dual-direction collection."""
    positions, _ = get_all_closed_positions_with_metadata(
        user,
        page_size=page_size,
        max_positions=max_positions,
    )
    return positions


# ── Activity ──────────────────────────────────────────────────

def get_activity(
    user: str,
    limit: int = 500,
    offset: int = 0,
    activity_type: Optional[str] = None,
    side: Optional[str] = None,
) -> list[dict]:
    """Fetch user activity (trades, splits, merges, redeems, etc.)."""
    params: dict = {
        "user": user,
        "limit": min(limit, 500),
        "offset": offset,
        "sortBy": "TIMESTAMP",
        "sortDirection": "DESC",
    }
    if activity_type:
        params["type"] = activity_type
    if side:
        params["side"] = side
    resp = _get(f"{DATA_API_BASE}/activity", params=params)
    return resp.json()


# ── Profile ───────────────────────────────────────────────────

def get_profile(address: str) -> dict:
    """Fetch public profile for a wallet address."""
    params = {"address": address}
    resp = _get(f"{GAMMA_API_BASE}/public-profile", params=params)
    return resp.json()
