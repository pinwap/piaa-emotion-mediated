"""Suites Registry and Discovery Module."""
from __future__ import annotations

import importlib
from pathlib import Path
from src.utils.suite_engine import Suite

SUITES_DIR = Path(__file__).resolve().parent


def get_all_suites() -> dict[str, Suite]:
    """Scan suites/ directory and return dict of suite_name -> Suite object."""
    import importlib.util
    suites = {}
    for p in sorted(SUITES_DIR.glob("*.py")):
        if p.name.startswith("__"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(p.stem, p)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "SUITE") and isinstance(mod.SUITE, Suite):
                    suites[mod.SUITE.name] = mod.SUITE
        except Exception as e:
            print(f"[Warning] Could not load suite {p.name}: {e}")
    return suites


def get_suite(name: str) -> Suite | None:
    """Retrieve a suite by its codename."""
    all_suites = get_all_suites()
    return all_suites.get(name.lower())
