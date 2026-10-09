"""Keep stdlib corpus CI usable without optional neural-training packages."""

import importlib.util
import unittest


def require(*modules):
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        raise unittest.SkipTest("optional test dependencies: " + ", ".join(missing))
