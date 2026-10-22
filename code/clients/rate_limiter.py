"""
Rate limiter for polite API access.
Enforces minimum delay between requests and exponential backoff on 429/5xx errors.
"""

import time
import threading


class RateLimiter:
    """Thread-safe rate limiter with configurable delay between requests."""

    def __init__(self, min_delay: float = 1.0):
        """
        Args:
            min_delay: Minimum seconds between requests (default 1.0).
        """
        self.min_delay = min_delay
        self._last_request_time = 0.0
        self._lock = threading.Lock()

    def wait(self):
        """Block until enough time has passed since the last request."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self.min_delay:
                time.sleep(self.min_delay - elapsed)
            self._last_request_time = time.monotonic()

    def backoff_delay(self, attempt: int, base: float = 2.0, max_delay: float = 60.0) -> float:
        """Compute exponential backoff delay.

        Args:
            attempt: Zero-based attempt number (0 = first retry).
            base: Base multiplier.
            max_delay: Cap on the delay.

        Returns:
            Seconds to sleep.
        """
        delay = min(base ** (attempt + 1), max_delay)
        return delay
