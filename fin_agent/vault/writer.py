"""Vault Writer: commit generated notes to a branch and open a PR (ADR-006).

The implementation lives in ``hubkit.vault.writer`` (Agent Hub ADR-H01). This module
pins Fin Analyst's path allowlist (``render.ALLOWED_PREFIXES``) and commit author, so
callers keep using ``VaultWriter(vault)`` and ``check_paths(files)``.
"""

from __future__ import annotations

from pathlib import Path

from hubkit.vault import writer as _w
from hubkit.vault.writer import SECRET_PATTERNS, WIKILINK, SyncResult, VaultSyncError, broken_links, scan_secrets  # noqa: F401

from fin_agent.vault.render import ALLOWED_PREFIXES


def check_paths(files: dict[str, str], allowed_prefixes: tuple[str, ...] = ALLOWED_PREFIXES) -> None:
    _w.check_paths(files, allowed_prefixes)


class VaultWriter(_w.VaultWriter):
    def __init__(self, vault_path: str | Path, base: str = "main", remote: str = "origin", push: bool = True):
        super().__init__(vault_path, ALLOWED_PREFIXES, base=base, remote=remote, push=push,
                         author="Fin Analyst Agent", email="fin-analyst-agent@localhost")
