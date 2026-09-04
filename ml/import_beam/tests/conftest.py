"""Shared test setup: make the package importable and build the fixtures."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                     # the handover directory
GENERATED = os.path.join(HERE, "fixtures", "generated")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def fixtures() -> str:
    """The generated fixture directory, built once per test session."""
    subprocess.check_call([sys.executable,
                           os.path.join(HERE, "fixtures", "make_fixtures.py"),
                           GENERATED])
    return GENERATED


@pytest.fixture(scope="session")
def test_registry(fixtures: str) -> str:
    return os.path.join(fixtures, "sources_test.yaml")


@pytest.fixture(scope="session")
def real_registry() -> str:
    return os.path.join(ROOT, "sources.yaml")
