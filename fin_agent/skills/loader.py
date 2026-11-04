"""Skill loader: vault notes -> skill packs (Vault Sync Contract, section 1).

Reads rule notes tagged ``fa-skill`` from ``$VAULT_PATH/Intelligence/Investment_Analysis/Analyst_Skills/``
and selects them by ``fa_lens`` and ``fa_sector`` frontmatter. Built-in defaults
cover any lens the vault does not have yet, so the agent works before the vault
notes exist. The full graph-neighborhood loader (wikilink expansion to concept
notes) arrives in step 3.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from fin_agent.data.schema import Sector
from fin_agent.skills.defaults import DEFAULT_SKILLS

SKILLS_SUBDIR = Path("Intelligence/Investment_Analysis/Analyst_Skills")


@dataclass
class SkillPack:
    lens: str
    text: str
    sources: list[str]  # vault paths (or "builtin:<lens>") used, for citation in reports


def _parse_note(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
    if not m:
        return {}, raw
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, m.group(2)


def load_skill_packs(lenses: list[str], sector: Sector, vault_path: str | None = None) -> dict[str, SkillPack]:
    vault_path = vault_path or os.getenv("VAULT_PATH")
    vault_notes: dict[str, list[tuple[str, str]]] = {}
    if vault_path:
        root = Path(vault_path)
        for p in sorted((root / SKILLS_SUBDIR).glob("*.md")):
            meta, body = _parse_note(p)
            tags = meta.get("tags") or []
            if "fa-skill" not in tags:
                continue
            note_sector = str(meta.get("fa_sector", "all"))
            if note_sector not in ("all", sector.value):
                continue
            vault_notes.setdefault(str(meta.get("fa_lens", "")), []).append((str(p.relative_to(root)), body.strip()))

    packs: dict[str, SkillPack] = {}
    for lens in lenses + ["recommendation", "qa"]:
        notes = vault_notes.get(lens, [])
        if notes:
            packs[lens] = SkillPack(lens, "\n\n".join(b for _, b in notes), [p for p, _ in notes])
        else:
            key = (lens, sector.value) if (lens, sector.value) in DEFAULT_SKILLS else (lens, "all")
            text = DEFAULT_SKILLS.get(key, "")
            packs[lens] = SkillPack(lens, text, [f"builtin:{lens}"] if text else [])
    return packs
