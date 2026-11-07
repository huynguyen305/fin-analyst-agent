"""LLM provider base types. Moved to ``hubkit.llm.base`` (Agent Hub ADR-H01); this name is kept as an alias so imports and patches still work."""
import sys

import hubkit.llm.base as _impl

sys.modules[__name__] = _impl
