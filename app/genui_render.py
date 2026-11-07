"""Streamlit renderer for agentic UI specs. Moved to ``hubkit.genui.render`` (Agent Hub ADR-H01); this name is kept as an alias so imports and patches still work."""
import sys

import hubkit.genui.render as _impl

sys.modules[__name__] = _impl
