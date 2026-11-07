"""Vault knowledge base: GraphRAG over Obsidian notes (langchain-graph-retriever + Chroma).

``vault_docs`` turns notes into chunk Documents whose metadata carries the graph
(links, tags, ticker, lens); ``store`` embeds them locally and retrieves with
graph traversal; ``retrieval`` is the agent node that runs after intake.
"""
