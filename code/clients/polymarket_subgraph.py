"""
Polymarket Goldsky subgraph client — for R10 historical-trade recovery.

The data-API endpoint /trades?user= is hard-capped at offset=3000 (~4000
trades returned, most-recent only). For 256 of our 1054 leaderboard wallets
this means the API loses access to most of their backtest-window trades.

The Goldsky-hosted Polymarket orderbook subgraph indexes every OrderFilled
event from the CTF Exchange contract on Polygon. It supports timestamp-cursor
pagination, has no 4000-trade cap, and is GraphQL-queryable for free.

Endpoint:
    https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/prod/gn

Per OrderFilledEvent fields (verified May 8, 2026 via subgraph_probe.py):
    id                    -- {transactionHash}_{orderHash}
    timestamp             -- string of Unix seconds
    maker, taker          -- addresses (the two parties to the fill)
    makerAssetId          -- token id offered by maker; 0 if USDC
    takerAssetId          -- token id offered by taker; 0 if USDC
    makerAmountFilled     -- raw amount (6 decimals)
    takerAmountFilled     -- raw amount (6 decimals)
    fee                   -- raw amount (6 decimals)
    transactionHash, orderHash
"""
from __future__ import annotations

import json
import time
from typing import Any, Iterator
import requests


SUBGRAPH_URL = (
    "https://api.goldsky.com/api/public/"
    "project_cl6mb8i9h0003e201j6li0diw/"
    "subgraphs/orderbook-subgraph/prod/gn"
)

# The Graph soft-caps `first` at 1000 per query
PAGE_SIZE = 1000

# Free public subgraph — be polite
DEFAULT_RATE_LIMIT = 0.5  # seconds between calls


class SubgraphClient:
    """Minimal client for the Polymarket orderbook subgraph."""

    def __init__(self, rate_limit_seconds: float = DEFAULT_RATE_LIMIT, timeout: int = 30):
        self.rate_limit_seconds = rate_limit_seconds
        self.timeout = timeout
        self._last_request_ts = 0.0
        self.calls = 0
        self.errors = 0

    def _wait_rate_limit(self):
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self.rate_limit_seconds:
            time.sleep(self.rate_limit_seconds - elapsed)
        self._last_request_ts = time.monotonic()

    def gql(self, query: str, variables: dict | None = None, max_retries: int = 4) -> dict:
        """Single GraphQL POST with retry/backoff on transient errors.

        Polymarket's Goldsky subgraph occasionally returns statement-timeout
        errors when the underlying Postgres DB is under load. We retry up
        to `max_retries` times with exponential backoff (2, 4, 8, 16 sec).
        Non-transient errors (e.g. malformed query) raise immediately.
        """
        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables

        last_err = None
        for attempt in range(max_retries + 1):
            self._wait_rate_limit()
            self.calls += 1
            try:
                resp = requests.post(SUBGRAPH_URL, json=payload, timeout=self.timeout)
                body = resp.json()
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_err = f"transport error: {exc}"
                if attempt < max_retries:
                    backoff = 2 ** (attempt + 1)
                    print(f"    [retry {attempt+1}/{max_retries}] subgraph {last_err}; sleeping {backoff}s")
                    time.sleep(backoff)
                    continue
                self.errors += 1
                raise RuntimeError(f"subgraph error after {max_retries} retries: {last_err}") from exc
            except (ValueError, json.JSONDecodeError) as exc:
                # Non-JSON response: typically a 502/503/504 HTML error page
                # from the gateway. Treat as transient.
                snippet = ""
                try:
                    snippet = resp.text[:200].replace("\n", " ")
                except Exception:
                    pass
                last_err = f"non-json response (HTTP {resp.status_code}): {snippet}"
                if attempt < max_retries:
                    backoff = 2 ** (attempt + 1)
                    print(f"    [retry {attempt+1}/{max_retries}] subgraph {last_err[:80]}; sleeping {backoff}s")
                    time.sleep(backoff)
                    continue
                self.errors += 1
                raise RuntimeError(f"subgraph error after {max_retries} retries: {last_err}") from exc

            if "errors" in body:
                err_msg = str(body["errors"])
                # Timeouts are transient — retry. Other errors (schema
                # mismatch, syntax) are not — fail fast.
                # Goldsky has at least three timeout phrasings observed:
                #   - "canceling statement due to statement timeout" (DB)
                #   - "Query timed out" (gateway)
                #   - "request timed out" (gateway)
                transient_markers = (
                    "statement timeout",
                    "canceling statement",
                    "query timed out",
                    "request timed out",
                    "timed out",
                )
                if any(m in err_msg.lower() for m in transient_markers):
                    last_err = err_msg
                    if attempt < max_retries:
                        backoff = 2 ** (attempt + 1)
                        print(f"    [retry {attempt+1}/{max_retries}] subgraph timeout; sleeping {backoff}s")
                        time.sleep(backoff)
                        continue
                self.errors += 1
                raise RuntimeError(f"subgraph error: {body['errors']}")
            return body.get("data", {})

        # Unreachable, but for safety:
        self.errors += 1
        raise RuntimeError(f"subgraph error after {max_retries} retries: {last_err}")

    def iter_order_filled_events(
        self,
        wallet: str,
        side: str = "either",  # "maker", "taker", or "either"
        from_ts: int | None = None,
        to_ts: int | None = None,
        progress_every: int = 0,
    ) -> Iterator[dict]:
        """Yield OrderFilledEvent dicts for a wallet, ordered by timestamp asc.

        Pagination uses `timestamp_gte` cursor and de-duplicates by `id`.
        On each page we advance the cursor to `last_ts` (not `last_ts + 1`)
        so that same-timestamp events spanning a page boundary aren't lost;
        the `seen` set elides the dups.

        We can't use `id_gt` cursor because the subgraph's index on `id` is
        prohibitively slow for filtered queries (verified May 8, 2026 — the
        backing query times out within 10 seconds on the database).

        Args:
            wallet: 0x-prefixed address (lowercase preferred).
            side: "maker", "taker", or "either" (runs both queries
                and merges/dedupes).
            from_ts, to_ts: optional time-window filters (Unix seconds).
            progress_every: if >0, print a one-line progress message every
                N yielded events (useful for the high-activity whales).
        """
        wallet = wallet.lower()
        sides = ["maker", "taker"] if side == "either" else [side]
        seen: set[str] = set()
        yielded = 0

        for s in sides:
            cursor_ts = from_ts or 0
            page = 0
            stalled_pages = 0
            while True:
                page += 1
                where_parts = [f'{s}: "{wallet}"']
                if cursor_ts > 0:
                    where_parts.append(f'timestamp_gte: "{cursor_ts}"')
                if to_ts:
                    where_parts.append(f'timestamp_lte: "{to_ts}"')
                where_clause = "{" + ", ".join(where_parts) + "}"

                query = f"""{{
                  orderFilledEvents(
                    first: {PAGE_SIZE},
                    orderBy: timestamp,
                    orderDirection: asc,
                    where: {where_clause}
                  ) {{
                    id
                    timestamp
                    maker
                    taker
                    makerAssetId
                    takerAssetId
                    makerAmountFilled
                    takerAmountFilled
                    fee
                    transactionHash
                    orderHash
                  }}
                }}"""

                data = self.gql(query)
                events = data.get("orderFilledEvents", [])
                if progress_every:
                    print(f"    [{s}] page {page}: got {len(events)} events  cursor_ts={cursor_ts}  total_yielded={yielded}")
                if not events:
                    break

                last_ts = None
                page_new = 0
                for ev in events:
                    if ev["id"] in seen:
                        continue
                    seen.add(ev["id"])
                    yielded += 1
                    page_new += 1
                    if progress_every and yielded % progress_every == 0:
                        print(f"    progress: yielded={yielded}")
                    yield ev
                    last_ts = int(ev["timestamp"])

                # Detect a stalled cursor (page returned only known events)
                if page_new == 0:
                    stalled_pages += 1
                    if stalled_pages >= 1:
                        # Force-advance past the cluster by +1 second.
                        # If the events at this exact timestamp exceed PAGE_SIZE,
                        # we lose the tail beyond PAGE_SIZE — acceptably rare.
                        cursor_ts = (cursor_ts + 1) if cursor_ts else (int(events[0]["timestamp"]) + 1)
                        if progress_every:
                            print(f"    [{s}] page {page}: 0 new events; force-advancing cursor to {cursor_ts}")
                        continue

                if len(events) < PAGE_SIZE:
                    break
                if last_ts:
                    cursor_ts = last_ts
                stalled_pages = 0


def order_filled_to_trade(event: dict, wallet: str) -> dict:
    """Convert a raw OrderFilledEvent into a record matching the
    `data/traders/{wallet}.json` trade schema (best effort).

    The data-api /trades schema fields we need to populate:
      proxyWallet, side ("BUY"/"SELL"), asset, conditionId, size, price,
      timestamp, transactionHash, outcome (best-effort), outcomeIndex.

    `conditionId` is NOT in the OrderFilledEvent — it would need a separate
    lookup. For the simulator's purposes we have the asset (token id), which
    is what the simulator actually needs to resolve prices. `conditionId` is
    only used for human-readable joins, so we leave it None and let the
    consumer resolve it later if needed.
    """
    wallet_l = wallet.lower()
    maker = event["maker"].lower()
    taker = event["taker"].lower()
    maker_asset = event["makerAssetId"]
    taker_asset = event["takerAssetId"]
    maker_amt = int(event["makerAmountFilled"])
    taker_amt = int(event["takerAmountFilled"])

    # In Polymarket: makerAssetId or takerAssetId == "0" is USDC; the other
    # is the conditional token (YES or NO outcome share).
    # If wallet is the side offering USDC (asset id 0), wallet is BUYING shares.
    # If wallet is the side offering shares (non-zero asset id), wallet is SELLING shares.

    if wallet_l == maker:
        wallet_offers = maker_asset
        wallet_receives = taker_asset
        wallet_offered_amt = maker_amt
        wallet_received_amt = taker_amt
        role = "maker"
    elif wallet_l == taker:
        wallet_offers = taker_asset
        wallet_receives = maker_asset
        wallet_offered_amt = taker_amt
        wallet_received_amt = maker_amt
        role = "taker"
    else:
        # Shouldn't happen — caller filtered by maker or taker
        raise ValueError(f"wallet {wallet_l} is neither maker {maker} nor taker {taker}")

    if wallet_offers == "0":
        # Wallet paid USDC → BUYing shares
        side = "BUY"
        asset = wallet_receives
        usdc_amount = wallet_offered_amt
        share_amount = wallet_received_amt
    else:
        # Wallet paid shares → SELLing
        side = "SELL"
        asset = wallet_offers
        usdc_amount = wallet_received_amt
        share_amount = wallet_offered_amt

    # Both USDC and CTF tokens use 6-decimal raw units on Polygon.
    # Price = USDC paid / shares received (in 6-decimal raw, the ratio is correct).
    if share_amount == 0:
        price = 0.0
    else:
        price = usdc_amount / share_amount

    # `size` in the data-api schema means the share amount (USDC equivalent at $1/share).
    # Original data-api returns it in float dollar-equivalent units; we mirror that:
    size = share_amount / 1e6

    return {
        "proxyWallet": wallet_l,
        "side": side,
        "asset": asset,
        "conditionId": None,  # Not in subgraph; resolve later if needed
        "size": round(size, 6),
        "price": round(price, 6),
        "timestamp": int(event["timestamp"]),
        "transactionHash": event["transactionHash"],
        "_source": "subgraph",
        "_role": role,  # "maker" or "taker"
    }
