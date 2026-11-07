"""Vault knowledge base (GraphRAG store). Moved to ``hubkit.knowledge.store`` (Agent Hub ADR-H01); this name is kept as an alias so imports and patches still work."""
import sys

import hubkit.knowledge.store as _impl

sys.modules[__name__] = _impl
