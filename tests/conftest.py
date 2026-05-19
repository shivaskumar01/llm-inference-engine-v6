"""pytest configuration for llmengine tests.

Adds the in-tree ``python/`` dir to sys.path so that the installed-extension
``import llmengine`` resolves to the local source package, regardless of the
caller's cwd.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "tests"))   # for fixtures.py imports


def pytest_configure(config):
    # Allow tests to opt into the correctness build via env var.
    if os.environ.get("LLMENGINE_BUILD") == "correctness":
        sys.path.insert(0, str(REPO_ROOT / "build" / "correctness" / "python"))
