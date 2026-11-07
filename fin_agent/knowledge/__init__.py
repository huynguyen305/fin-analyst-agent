"""Vault knowledge base: GraphRAG over Obsidian notes (langchain-graph-retriever + Chroma).

``vault_docs`` turns notes into chunk Documents whose metadata carries the graph
(links, tags, ticker, lens); ``store`` embeds them locally and retrieves with
graph traversal; ``retrieval`` is the agent node that runs after intake.
"""

from pathlib import Path

from hubkit.config import env

# Fin Analyst's index stays in this repo (hubkit's default is the working directory)
FIN_KB_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge"


def kb_dir() -> Path:
    """``HUB_KB_DIR`` / ``FIN_AGENT_KB_DIR`` if set, else ``<repo>/data/knowledge``."""
    return Path(env("KB_DIR") or FIN_KB_DIR)
