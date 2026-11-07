"""Configuration from environment variables. Moved to ``hubkit.config`` (Agent Hub ADR-H01); this name is kept as an alias so imports and patches still work."""
import sys

import hubkit.config as _impl

sys.modules[__name__] = _impl
