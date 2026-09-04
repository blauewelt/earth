"""The politeness machinery, with a fake clock — no test may actually sleep."""
from __future__ import annotations

import pytest

from beam_import import hosts
from beam_import.hosts import (BREAKER_TRIP_AFTER, CircuitOpen, LaneState,
                               TransientError)


class Clock:
    """A stand-in for time.sleep + time.monotonic that records the sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept = []

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.now += s

    def monotonic(self) -> float:
        return self.now


def _lane(**cfg) -> tuple:
    clock = Clock()
    base = {"max_lanes": 1, "min_gap_s": 0}
    base.update(cfg)
    lane = LaneState("t", base, sleep_fn=clock.sleep,
                     rand_fn=lambda a, b: 1.0, clock_fn=clock.monotonic)
    return lane, clock


def test_pace_sleeps_the_gap_and_no_more():
    lane, clock = _lane(min_gap_s=20)
    lane.pace()                     # the first request never waits
    assert clock.slept == []
    lane.pace()
    assert clock.slept == [20.0]
    assert lane.counters["requests"] == 2


def test_backoff_ladder_is_60s_5min_15min_60min():
    lane, _clock = _lane()
    got = [lane.backoff_seconds(i) for i in range(1, 6)]
    assert got == [60.0, 300.0, 900.0, 3600.0, 3600.0]


def test_jitter_stays_inside_0_8_to_1_2():
    for r in (0.8, 1.0, 1.2):
        lane = LaneState("t", {"max_lanes": 1, "min_gap_s": 0},
                         sleep_fn=lambda s: None, rand_fn=lambda a, b: r)
        assert lane.backoff_seconds(1) == pytest.approx(60.0 * r)


def test_retry_after_always_wins():
    lane, _clock = _lane()
    assert lane.backoff_seconds(1, retry_after=1800) == 1800.0
    # ... but only when it is LONGER than our own rung.
    assert lane.backoff_seconds(4, retry_after=10) == 3600.0


def test_run_with_backoff_climbs_the_ladder_then_gives_up():
    lane, clock = _lane()
    calls = {"n": 0}

    def always_fails():
        calls["n"] += 1
        raise TransientError("nope")

    with pytest.raises(TransientError):
        lane.run_with_backoff(always_fails)
    assert calls["n"] == 5                      # first try + four rungs
    assert clock.slept == [60.0, 300.0, 900.0, 3600.0]
    assert lane.counters["backoffs"] == 4


def test_run_with_backoff_returns_on_a_later_attempt():
    lane, _clock = _lane()
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientError("later")
        return "ok"

    assert lane.run_with_backoff(flaky) == "ok"
    assert lane.counters["backoffs"] == 2


def test_breaker_trips_after_five_consecutive_failures():
    lane, _clock = _lane()
    for i in range(BREAKER_TRIP_AFTER - 1):
        assert lane.record_failure() is False
        assert lane.open is False
    assert lane.record_failure() is True
    assert lane.open is True
    assert lane.counters["trips"] == 1
    with pytest.raises(CircuitOpen):
        lane.check_open()


def test_a_success_resets_the_breaker_count():
    lane, _clock = _lane()
    for _ in range(BREAKER_TRIP_AFTER - 1):
        lane.record_failure()
    lane.record_success()
    for _ in range(BREAKER_TRIP_AFTER - 1):
        assert lane.record_failure() is False
    assert lane.open is False


def test_a_tripped_lane_refuses_more_work():
    lane, _clock = _lane()
    for _ in range(BREAKER_TRIP_AFTER):
        lane.record_failure()
    with pytest.raises(CircuitOpen):
        lane.run_with_backoff(lambda: "never reached")


def test_lane_states_are_built_from_the_registry(real_registry):
    from beam_import import registry
    reg = registry.load(real_registry)
    lanes = hosts.lane_states(reg)
    assert lanes["psl"].min_gap_s == 20
    assert lanes["psl"].head_poll_s == 900
    assert lanes["ncei"].max_lanes == 6
