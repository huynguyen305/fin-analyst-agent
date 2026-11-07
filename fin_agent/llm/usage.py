"""Token usage and cost accounting. Moved to ``hubkit.llm.usage`` (Agent Hub ADR-H01); this name is kept as an alias so imports and patches still work."""
import sys

import hubkit.llm.usage as _impl

sys.modules[__name__] = _impl
