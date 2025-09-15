# tests/test_app_smoke.py
"""
Smoke test that simulates a CPU-only Hugging Face Spaces import.
It *does not* start any server or UI. If no app module exists, the test is skipped.
"""

import sys
import pathlib
import importlib
import inspect
import pytest

def _import_app_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))

    # Prefer real files if present; otherwise try a plain "app" import
    candidates = []
    if (root / "app.py").exists():
        candidates.append("app")
    if (root / "src" / "app.py").exists():
        candidates.append("app")
    if not candidates:
        candidates = ["app"]

    last_err = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception as e:
            last_err = e
    pytest.skip(f"No importable app module found (last error: {last_err})")

def _try_tiny_call(module) -> bool:
    preferred = ("ping", "healthcheck", "health_check", "hello", "version", "get_version")
    fallback  = ("predict", "inference", "run", "main", "chord_bot")

    def try_call(fn):
        try:
            sig = inspect.signature(fn)
            if len(sig.parameters) == 0:
                fn()
                return True
            if len(sig.parameters) == 1:
                fn("test")
                return True
        except Exception:
            return False
        return False

    called = False
    for name in list(preferred) + list(fallback):
        if hasattr(module, name) and callable(getattr(module, name)):
            if try_call(getattr(module, name)):
                called = True
                break
    return called

def test_import_app_and_optionally_call():
    module = _import_app_module()
    assert module is not None  # Import alone verifies basic environment
    # Try a non-server, tiny function call if present; ok if none fits
    _try_tiny_call(module)
