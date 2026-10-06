#!/usr/bin/env python3
"""scripts/family1_queue.mjs --selftest, in the suite (it pins the queue
keeper's transitions, including the #1040 fix: a lane whose run was once seen
is never declared lost)."""
import os
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_queue_keeper_selftest():
    r = subprocess.run(["node", "scripts/family1_queue.mjs", "--selftest"],
                       cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "selftest: all passed" in r.stdout
    assert "a tick whose listings miss it keeps the lane in flight" in r.stdout
