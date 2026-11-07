"""TypeSafe Jev client. Moved to ``hubkit.evals.jev`` (Agent Hub ADR-H01); this name is kept as an alias so imports and patches still work."""
import sys

import hubkit.evals.jev as _impl

sys.modules[__name__] = _impl
