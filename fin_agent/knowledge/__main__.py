"""Build and inspect the vault knowledge base.

    python -m fin_agent.knowledge index            # incremental (only changed chunks are embedded)
    python -m fin_agent.knowledge index --full     # rebuild from scratch
    python -m fin_agent.knowledge search "is FPT overbought?" --ticker FPT
    python -m fin_agent.knowledge stats
"""

from __future__ import annotations

import argparse
import os
import sys

from fin_agent.config import load_env
from fin_agent.knowledge import kb_dir
from fin_agent.knowledge.store import KnowledgeBase, KnowledgeError
from fin_agent.knowledge.vault_docs import DEFAULT_INCLUDE


def include_dirs() -> tuple[str, ...]:
    raw = os.getenv("FIN_AGENT_KB_INCLUDE")
    return tuple(x.strip() for x in raw.split(",") if x.strip()) if raw else DEFAULT_INCLUDE


def main(argv: list[str] | None = None) -> int:
    load_env()
    ap = argparse.ArgumentParser(description="Vault knowledge base (GraphRAG)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ix = sub.add_parser("index", help="embed vault notes into the local vector store")
    ix.add_argument("--full", action="store_true", help="drop the collection and re-embed everything")
    se = sub.add_parser("search", help="run a graph retrieval and show what comes back")
    se.add_argument("query")
    se.add_argument("--ticker", default=None, help="only notes about this company")
    se.add_argument("-k", type=int, default=6)
    se.add_argument("--depth", type=int, default=2, help="max graph hops (0 = plain similarity search)")
    se.add_argument("--reference", action="store_true", help="reference notes only (as the agent's lenses do)")
    sub.add_parser("stats", help="show what is indexed")
    args = ap.parse_args(argv)

    kb = KnowledgeBase(directory=kb_dir())
    try:
        if args.cmd == "index":
            vault = os.getenv("VAULT_PATH")
            if not vault:
                print("ERROR: VAULT_PATH is not set", file=sys.stderr)
                return 2
            print(f"Indexing {', '.join(include_dirs())} from {vault} with {kb.spec} ...")
            s = kb.sync(vault, include_dirs(), progress=lambda m: print("  " + m), full=args.full)
            print(f"{s.notes} notes, {s.chunks} chunks: embedded {s.embedded}, metadata updated {s.metadata_updated}, "
                  f"deleted {s.deleted}{' (rebuilt)' if s.rebuilt else ''} in {s.seconds:.0f}s")
        elif args.cmd == "search":
            for h in kb.search(args.query, k=args.k, ticker=args.ticker, max_depth=args.depth, reference_only=args.reference):
                print(f"[hop {h.depth} · sim {h.score:.3f} · {h.kind}] {h.path} > {h.heading}")
                print("   " + h.text.replace("\n", " ")[:220])
        else:
            m = kb.manifest()
            print(f"store {kb.store_kind} at {kb.dir}\nembeddings {m.get('embedding_model')}\nbuilt {m.get('built_at')}\n"
                  f"notes {len(m.get('files', {}))}, chunks {len(m.get('chunks', {}))}, in store {kb.count()}")
    except KnowledgeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
