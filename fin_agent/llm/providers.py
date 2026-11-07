"""LLM providers. Moved to ``hubkit.llm.providers`` (Agent Hub ADR-H01); this name is kept as an alias so imports and patches still work."""
import sys

import hubkit.llm.providers as _impl

sys.modules[__name__] = _impl
