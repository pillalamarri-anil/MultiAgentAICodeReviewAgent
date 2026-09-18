"""Thread-safe token-bucket limiter.

Specialist agents fire up to 4 concurrent requests per file, each carrying a
multi-thousand-token context; bursting those against the account's tokens-per-minute
cap is what produces 429s. Pacing requests proactively (this module) is cheaper and
more reliable than firing them all and leaning on reactive retries after a 429.
"""

from __future__ import annotations

import threading
import time


class TokenBucket:
    def __init__(self, tokens_per_minute: int):
        self.capacity = max(int(tokens_per_minute), 1)
        self._rate_per_sec = self.capacity / 60.0
        self._available = float(self.capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        self._available = min(self.capacity, self._available + elapsed * self._rate_per_sec)

    def acquire(self, tokens: int) -> None:
        """Block until ``tokens`` worth of budget is available, then deduct it."""
        tokens = min(tokens, self.capacity)  # a single call can never exceed the whole bucket
        while True:
            with self._lock:
                self._refill_locked()
                if self._available >= tokens:
                    self._available -= tokens
                    return
                deficit = tokens - self._available
                wait_s = deficit / self._rate_per_sec
            time.sleep(min(wait_s, 1.0))

    def adjust(self, estimated: int, actual: int) -> None:
        """Correct the bucket once real usage is known -- the pre-call estimate is a
        rough approximation, so this keeps the bucket accurate as a run progresses."""
        diff = estimated - actual
        if not diff:
            return
        with self._lock:
            self._refill_locked()
            self._available = min(self.capacity, self._available + diff)
