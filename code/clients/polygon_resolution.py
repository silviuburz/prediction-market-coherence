"""Polygon client for ConditionalTokens ``ConditionResolution`` events.

The indexed condition ID supports efficient full-history OR queries. Each
event supplies the common condition-resolution transaction and payout vector;
block timestamps are fetched in JSON-RPC batches.
"""
from __future__ import annotations

import logging
import time
from typing import Iterable

import requests


logger = logging.getLogger(__name__)

DEFAULT_RPC_URL = "https://polygon.gateway.tenderly.co"
DEFAULT_FALLBACK_RPC_URLS = (
    "https://polygon.api.onfinality.io/public",
    "https://polygon.drpc.org",
)
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
CONDITION_RESOLUTION_TOPIC = (
    "0xb44d84d3289691f71497564b85d4233648d9dbae8cbdbb4329f301c3a0185894"
)
MAX_RETRIES = 4


def normalize_bytes32(value: str) -> str:
    text = str(value).strip().lower()
    if not text.startswith("0x"):
        text = "0x" + text
    if len(text) != 66:
        raise ValueError(f"expected bytes32 hex value, got {value!r}")
    try:
        int(text[2:], 16)
    except ValueError as error:
        raise ValueError(f"expected bytes32 hex value, got {value!r}") from error
    return text


def _word(data: bytes, offset: int) -> int:
    if offset < 0 or offset + 32 > len(data):
        raise ValueError("ABI word offset is outside event data")
    return int.from_bytes(data[offset:offset + 32], "big")


def decode_condition_resolution_log(event: dict) -> dict:
    """Decode one CTF ``ConditionResolution`` log without a web3 dependency."""
    topics = event.get("topics") or []
    if len(topics) != 4:
        raise ValueError(f"ConditionResolution log has {len(topics)} topics, expected 4")
    if str(topics[0]).lower() != CONDITION_RESOLUTION_TOPIC:
        raise ValueError("unexpected event signature")
    condition_id = normalize_bytes32(topics[1])
    question_id = normalize_bytes32(topics[3])
    oracle_topic = str(topics[2]).lower().removeprefix("0x")
    if len(oracle_topic) != 64:
        raise ValueError("invalid indexed oracle topic")
    oracle = "0x" + oracle_topic[-40:]

    data_hex = str(event.get("data", ""))
    if not data_hex.startswith("0x"):
        raise ValueError("event data is not hex-prefixed")
    try:
        data = bytes.fromhex(data_hex[2:])
    except ValueError as error:
        raise ValueError("event data is not valid hex") from error
    if len(data) < 96 or len(data) % 32:
        raise ValueError("event data has invalid ABI length")

    outcome_slot_count = _word(data, 0)
    payout_offset = _word(data, 32)
    if payout_offset % 32:
        raise ValueError("payout array offset is not word-aligned")
    payout_count = _word(data, payout_offset)
    if payout_count != outcome_slot_count:
        raise ValueError(
            "outcome slot count does not match payout array length: "
            f"{outcome_slot_count} != {payout_count}"
        )
    payouts = [
        _word(data, payout_offset + 32 + index * 32)
        for index in range(payout_count)
    ]
    denominator = sum(payouts)
    if denominator <= 0:
        raise ValueError("payout vector has zero denominator")

    return {
        "condition_id": condition_id,
        "oracle": oracle,
        "question_id": question_id,
        "outcome_slot_count": outcome_slot_count,
        "payout_numerators": payouts,
        "payout_denominator": denominator,
        "outcomes_by_index": [value / denominator for value in payouts],
        "block_number": int(event["blockNumber"], 16),
        "block_hash": str(event.get("blockHash", "")).lower(),
        "transaction_hash": str(event.get("transactionHash", "")).lower(),
        "transaction_index": int(event.get("transactionIndex", "0x0"), 16),
        "log_index": int(event.get("logIndex", "0x0"), 16),
        "removed": bool(event.get("removed", False)),
    }


class PolygonResolutionClient:
    """Minimal archival JSON-RPC client for condition resolutions."""

    def __init__(
        self,
        rpc_url: str = DEFAULT_RPC_URL,
        fallback_rpc_urls: Iterable[str] = DEFAULT_FALLBACK_RPC_URLS,
        timeout: int = 90,
        min_delay: float = 0.1,
    ):
        self.rpc_url = rpc_url
        self.fallback_rpc_urls = tuple(
            url for url in fallback_rpc_urls if url and url != rpc_url
        )
        self.timeout = timeout
        self.min_delay = min_delay
        self._last_request = 0.0
        self.calls = 0

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self._last_request = time.monotonic()

    def _post(self, payload, rpc_url: str | None = None):
        target_url = rpc_url or self.rpc_url
        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            self._wait()
            self.calls += 1
            try:
                response = requests.post(
                    target_url,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                body = response.json()
                if isinstance(body, dict) and body.get("error"):
                    raise RuntimeError(str(body["error"]))
                return body
            except (requests.RequestException, ValueError, RuntimeError) as error:
                last_error = error
                if attempt >= MAX_RETRIES:
                    break
                delay = 2 ** attempt
                logger.warning(
                    "Polygon RPC attempt %d/%d failed: %s; retrying in %ds",
                    attempt + 1,
                    MAX_RETRIES + 1,
                    error,
                    delay,
                )
                time.sleep(delay)
        raise RuntimeError(f"Polygon RPC failed after retries: {last_error}")

    def _get_single_block_timestamp(self, block_number: int) -> int:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_getBlockByNumber",
            "params": [hex(block_number), False],
        }
        errors = []
        for rpc_url in (self.rpc_url, *self.fallback_rpc_urls):
            try:
                body = self._post(payload, rpc_url=rpc_url)
                result = body.get("result") if isinstance(body, dict) else None
                if result:
                    return int(result["timestamp"], 16)
                errors.append(f"{rpc_url}: missing result")
            except RuntimeError as error:
                errors.append(f"{rpc_url}: {error}")
        raise RuntimeError(
            f"no archival block response for {block_number}: {'; '.join(errors)}"
        )

    def get_resolution_logs(self, condition_ids: Iterable[str]) -> list[dict]:
        normalized = list(dict.fromkeys(normalize_bytes32(value) for value in condition_ids))
        if not normalized:
            return []
        params = {
            "address": CTF_ADDRESS,
            "fromBlock": "0x0",
            "toBlock": "latest",
            "topics": [CONDITION_RESOLUTION_TOPIC, normalized],
        }
        body = self._post({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_getLogs",
            "params": [params],
        })
        return body.get("result", [])

    def get_block_timestamps(
        self,
        block_numbers: Iterable[int],
        batch_size: int = 100,
    ) -> dict[int, int]:
        unique = list(dict.fromkeys(int(value) for value in block_numbers))
        timestamps: dict[int, int] = {}
        for start in range(0, len(unique), batch_size):
            batch = unique[start:start + batch_size]
            payload = [
                {
                    "jsonrpc": "2.0",
                    "id": index,
                    "method": "eth_getBlockByNumber",
                    "params": [hex(block_number), False],
                }
                for index, block_number in enumerate(batch)
            ]
            body = self._post(payload)
            if isinstance(body, dict):
                body = [body]
            if not isinstance(body, list):
                body = []
            by_id = {int(item["id"]): item for item in body}
            for index, block_number in enumerate(batch):
                item = by_id.get(index)
                if not item or item.get("error") or not item.get("result"):
                    timestamps[block_number] = self._get_single_block_timestamp(
                        block_number
                    )
                else:
                    timestamps[block_number] = int(item["result"]["timestamp"], 16)
        return timestamps

    def get_resolutions(self, condition_ids: Iterable[str]) -> dict[str, dict]:
        decoded = [
            decode_condition_resolution_log(event)
            for event in self.get_resolution_logs(condition_ids)
            if not event.get("removed", False)
        ]
        timestamps = self.get_block_timestamps(
            record["block_number"] for record in decoded
        )
        resolutions: dict[str, dict] = {}
        for record in decoded:
            condition_id = record["condition_id"]
            if condition_id in resolutions:
                raise RuntimeError(f"multiple ConditionResolution logs for {condition_id}")
            record["block_timestamp"] = timestamps[record["block_number"]]
            resolutions[condition_id] = record
        return resolutions