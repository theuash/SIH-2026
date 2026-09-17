"""Native core loader. `NETRADR_IMPL=py` forces the Python twins (parity tests).
`auto` (default) uses C++ when built, else Python. API unchanged either way.
"""
import os

IMPL = os.environ.get("NETRADR_IMPL", "auto")

try:
    if IMPL == "py":
        raise ImportError("forced python")
    from . import netradr_core as mod
except ImportError:
    mod = None


def available():
    return mod is not None


def want_native():
    return IMPL in ("auto", "cxx") and mod is not None
