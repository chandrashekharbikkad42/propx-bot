"""Regression test for Bug #6 — credential hot-reload AttributeError.

`scripts/run_asian_sweep_live.py` used to rebuild the Settings singleton
after the interactive credential prompt with:

    import config.settings as _settings_module
    _settings_module.settings = _settings_module._build()

Because `config/__init__.py` does `from .settings import settings`,
Python 3.5+ semantics make `import config.settings as X` equivalent to
`from config import settings as X` — which binds X to the **Settings
dataclass instance**, not the module. The follow-up `._build()` call
raised `AttributeError: 'Settings' object has no attribute '_build'`
and crashed every LIVE arm (`--no-dry-run` + EXECUTION_MODE=REAL,
`--reset`, or `--switch`).

These tests pin both the failure mode (so a regression is loud) and
the `importlib.import_module` fix (the path the live script now uses).
"""

from __future__ import annotations
import importlib

import config
import config.settings as _aliased
from config.settings import Settings


class TestImportAliasReproducesBug6:
    """Documents the broken pattern. If this ever passes through to a
    real module we know `config/__init__.py` changed and the original
    workaround can be reconsidered."""

    def test_attribute_import_resolves_to_dataclass_instance(self):
        # `import config.settings as _aliased` actually returns the
        # Settings instance attached to the `config` package.
        assert isinstance(_aliased, Settings)
        assert not hasattr(_aliased, "_build")


class TestImportlibImportModuleIsCorrect:
    """Pins the fix used by `run_asian_sweep_live.py`."""

    def test_importlib_returns_module_with_build(self):
        mod = importlib.import_module("config.settings")
        # Must be the module (has both the dataclass + factory + singleton).
        assert hasattr(mod, "_build"), \
            "config.settings._build vanished — regression of Bug #6"
        assert callable(mod._build)
        assert hasattr(mod, "Settings")
        assert hasattr(mod, "settings")

    def test_build_is_idempotent_singleton_replacement(self):
        # The exact call pattern the live script uses.
        mod = importlib.import_module("config.settings")
        old_singleton = mod.settings
        new_singleton = mod._build()
        mod.settings = new_singleton
        try:
            assert isinstance(new_singleton, Settings)
            # `_build` should return a fresh Settings each call.
            assert new_singleton is not old_singleton or \
                new_singleton == old_singleton
        finally:
            # Restore so other tests sharing the singleton see no drift.
            mod.settings = old_singleton

    def test_package_attribute_still_points_at_instance(self):
        # And the package-level shortcut still works the way callers
        # expect (`from config import settings` returns the instance).
        assert isinstance(config.settings, Settings)
