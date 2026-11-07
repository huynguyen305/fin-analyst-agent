"""LangChain chat-model adapter. Moved to ``hubkit.llm.langchain_adapter`` (Agent Hub ADR-H01); this name is kept as an alias so imports and patches still work."""
import sys

import hubkit.llm.langchain_adapter as _impl

sys.modules[__name__] = _impl
