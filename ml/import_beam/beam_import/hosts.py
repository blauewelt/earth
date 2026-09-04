"""LaneState — everything that makes one lane polite.

A LANE is one sequential stream of requests to one host. Beam's GroupByKey
hands all the items of one (host, lane) key to one worker as one iterable, so
the number of lanes per host IS the maximum number of simultaneous connections
that host will ever see from us. This module holds the per-lane state that
bounds what happens inside a lane:

  pace             sleep at least `min_gap_s` before every request
  backoff ladder   60 s, 5 min, 15 min, 60 min, each multiplied by a random
                   0.8-1.2 jitter, and a server's own Retry-After always wins
  circuit breaker  5 consecutive failed items and the lane stops; everything
                   it has not reached yet is appended to the RETRY QUEUE and
                   reported `queued`, so a later run picks it up. There is no
                   `failed` state anywhere in this package (DESIGN §2)
  counters         requests, bytes, backoffs, breaker trips — the politeness
                   audit that ends up in summary.md

Nothing here talks to the network. It is pure state plus sleeps, which is what
makes it testable with a fake clock (see tests/test_hosts.py).
"""
from __future__ import annotations

import random
import time
from typing import Any, Callable, Dict, List, Optional

# The ladder, in seconds. Four rungs means at most five attempts per item.
DEFAULT_BACKOFF_LADDER_S: List[float] = [60.0, 300.0, 900.0, 3600.0]

# Multiplied onto every ladder rung so a fleet of lanes does not retry in step.
JITTER_LO, JITTER_HI = 0.8, 1.2

# Consecutive failed items that stop a lane.
BREAKER_TRIP_AFTER = 5


class TransientError(Exception):
    """Something that might work later: 429, 5xx, a timeout, a reset socket,
    a short transfer. Retried up the ladder, then the item is `failed`."""


class PermanentError(Exception):
    """Something no amount of waiting fixes: a 403, a file that is not the
    format it claims, a wrong password. Not retried inside the run — but the
    item still goes to the RETRY QUEUE with the reason attached, because
    DESIGN §2 allows no state in which work is silently dropped."""


class NotFound(Exception):
    """The server answered 404 or 410 for this exact thing.

    This is NOT yet `absent`. DESIGN §2: a source has to say no TWICE, on
    runs at least six hours apart, before we believe it — one 404 is a bad
    afternoon, and "a truncated transfer raised no exception" is the reason
    this project does not take a single answer as evidence. The response is
    recorded by `sinks.record_not_found` and the item stays `queued` until
    the second sighting.
    """

    def __init__(self, message: str, status: int = 404, url: str = "",
                 headers: Optional[Dict[str, str]] = None) -> None:
        super().__init__(message)
        self.status = status
        self.url = url
        self.headers = headers or {}

    def evidence(self) -> Dict[str, Any]:
        return {"status": self.status, "url": self.url,
                "headers": self.headers, "message": str(self)}


class BlockedError(Exception):
    """A gated source with no credentials (ERA5 without a CDS key). Reported
    `blocked`; nothing is requested and nothing is retried."""


class CircuitOpen(Exception):
    """Raised inside a lane when the breaker has tripped. The LaneWorker
    catches it and puts everything the lane has not reached on the retry
    queue — `queued`, never dropped."""


class LaneState:
    """The politeness state of ONE lane on ONE host.

    Args:
        host:      the host name, for counters and messages
        cfg:       that host's row from sources.yaml
        sleep_fn:  injected so tests can run the ladder without waiting
        rand_fn:   injected so tests get a deterministic jitter
        clock_fn:  injected so tests can control elapsed time
    """

    def __init__(self, host: str, cfg: Dict[str, Any],
                 sleep_fn: Callable[[float], None] = time.sleep,
                 rand_fn: Callable[[float, float], float] = random.uniform,
                 clock_fn: Callable[[], float] = time.monotonic) -> None:
        self.host = host
        self.min_gap_s = float(cfg.get("min_gap_s", 0.0))
        self.max_lanes = int(cfg.get("max_lanes", 1))
        self.head_poll_s = float(cfg.get("head_poll_s", 0.0))
        self.ladder: List[float] = [float(x) for x in cfg.get(
            "backoff_ladder_s", DEFAULT_BACKOFF_LADDER_S)]
        self._sleep = sleep_fn
        self._rand = rand_fn
        self._clock = clock_fn

        self._last_request_at: Optional[float] = None
        self.consecutive_failures = 0
        self.open = False                       # True once the breaker trips

        self.counters: Dict[str, float] = {
            "requests": 0, "bytes": 0, "backoffs": 0, "trips": 0,
            "backoff_seconds": 0.0, "paced_seconds": 0.0,
        }

    # -- pacing ------------------------------------------------------------
    def pace(self) -> None:
        """Sleep until `min_gap_s` has passed since the previous request."""
        now = self._clock()
        if self._last_request_at is not None:
            wait = self.min_gap_s - (now - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
                self.counters["paced_seconds"] += wait
        self._last_request_at = self._clock()
        self.counters["requests"] += 1

    def note_bytes(self, n: int) -> None:
        self.counters["bytes"] += int(n)

    # -- the backoff ladder ------------------------------------------------
    def backoff_seconds(self, attempt: int,
                        retry_after: Optional[float] = None) -> float:
        """How long to wait before attempt number `attempt` (1-based).

        A server's own Retry-After always wins — if it tells us 20 minutes, we
        wait 20 minutes even where our ladder would have said 60 s.
        """
        rung = self.ladder[min(attempt - 1, len(self.ladder) - 1)]
        wait = rung * self._rand(JITTER_LO, JITTER_HI)
        if retry_after is not None:
            wait = max(wait, float(retry_after))
        return wait

    def backoff(self, attempt: int, retry_after: Optional[float] = None) -> float:
        """Sleep one rung of the ladder. Returns the seconds slept."""
        wait = self.backoff_seconds(attempt, retry_after)
        self.counters["backoffs"] += 1
        self.counters["backoff_seconds"] += wait
        self._sleep(wait)
        return wait

    def attempts_allowed(self) -> int:
        """Attempts per item: the first try plus one per ladder rung."""
        return len(self.ladder) + 1

    # -- the circuit breaker ----------------------------------------------
    def check_open(self) -> None:
        """Raise CircuitOpen if this lane has already stopped."""
        if self.open:
            raise CircuitOpen(
                f"{self.host}: lane stopped after {BREAKER_TRIP_AFTER} "
                "consecutive failures")

    def record_success(self) -> None:
        self.consecutive_failures = 0

    def record_failure(self) -> bool:
        """Count one failed item. Returns True if that tripped the breaker."""
        self.consecutive_failures += 1
        if self.consecutive_failures >= BREAKER_TRIP_AFTER and not self.open:
            self.open = True
            self.counters["trips"] += 1
            return True
        return False

    # -- running one attempt-loop -----------------------------------------
    def run_with_backoff(self, fn: Callable[[], Any], on_note=None) -> Any:
        """Call `fn` until it succeeds or the ladder runs out.

        TransientError climbs the ladder. PermanentError, NotFound and
        BlockedError come straight back out — retrying them would be rudeness
        with no possible payoff. Anything else (a bug in our code) also comes
        straight out, because DESIGN §2 says only programming errors may
        propagate out of the DoFn and the LaneWorker is the one place that
        decides what to do with them.
        """
        self.check_open()
        last: Optional[BaseException] = None
        for attempt in range(1, self.attempts_allowed() + 1):
            try:
                self.pace()
                return fn()
            except TransientError as exc:
                last = exc
                if attempt >= self.attempts_allowed():
                    break
                retry_after = getattr(exc, "retry_after", None)
                slept = self.backoff(attempt, retry_after)
                if on_note:
                    on_note(f"attempt {attempt} failed ({exc}); "
                            f"slept {slept:.0f}s")
        raise TransientError(f"gave up after {self.attempts_allowed()} "
                             f"attempts: {last}")


def lane_states(reg, sleep_fn=time.sleep) -> Dict[str, LaneState]:
    """One LaneState per host — what a worker builds in DoFn.setup()."""
    return {name: LaneState(name, cfg, sleep_fn=sleep_fn)
            for name, cfg in reg.hosts.items()}
