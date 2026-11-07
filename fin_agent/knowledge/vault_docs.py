"""Vault notes to chunk Documents. Moved to ``hubkit.knowledge.vault_docs`` (Agent Hub ADR-H01); this name is kept as an alias so imports and patches still work."""
import sys

import hubkit.knowledge.vault_docs as _impl

sys.modules[__name__] = _impl
