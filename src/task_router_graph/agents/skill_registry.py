from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from ..protocol_constants import FM_ALLOWED_TOOLS, SKILL_FILENAME

REQUIRED_FRONTMATTER_FIELDS = ("name", "description", "when_to_use", FM_ALLOWED_TOOLS)
SKILL_MODE_FIELD = "skill-mode"
SKILL_MODE_SYNC = "sync"
SKILL_MODE_PYSKILL = "pyskill"


class SkillRegistryError(ValueError):
    pass


def normalize_skill_key(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower())
    return normalized.strip("-")


def _parse_frontmatter(skill_path: Path) -> tuple[dict[str, Any], str]:
    raw = skill_path.read_text(encoding="utf-8")
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    if not text.startswith("---\n"):
        raise SkillRegistryError(f"skill file missing frontmatter header: {skill_path}")

    end = text.find("\n---\n", 4)
    if end < 0:
        raise SkillRegistryError(f"skill file missing frontmatter closing marker: {skill_path}")

    frontmatter_text = text[4:end]
    content_text = text[end + 5 :].strip()

    try:
        frontmatter = yaml.safe_load(frontmatter_text)
    except Exception as exc:
        raise SkillRegistryError(f"skill frontmatter yaml parse failed: {skill_path}: {exc}") from exc

    if not isinstance(frontmatter, dict):
        raise SkillRegistryError(f"skill frontmatter must be a mapping: {skill_path}")

    return dict(frontmatter), content_text


def _validate_allowed_tools(raw: Any, *, skill_path: Path) -> list[str]:
    if not isinstance(raw, list):
        raise SkillRegistryError(f"{FM_ALLOWED_TOOLS} must be a string array: {skill_path}")

    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            raise SkillRegistryError(f"{FM_ALLOWED_TOOLS} must be a string array: {skill_path}")
        tool_name = item.strip()
        if not tool_name:
            raise SkillRegistryError(f"{FM_ALLOWED_TOOLS} contains empty tool name: {skill_path}")
        if tool_name in seen:
            raise SkillRegistryError(f"{FM_ALLOWED_TOOLS} contains duplicate tool name '{tool_name}': {skill_path}")
        seen.add(tool_name)
        result.append(tool_name)

    return result


def _resolve_script_for_tool(*, skill_dir: Path, tool_name: str) -> Path:
    scripts_dir = skill_dir / "scripts"
    sh_path = scripts_dir / f"{tool_name}.sh"
    py_path = scripts_dir / f"{tool_name}.py"

    if sh_path.exists() and sh_path.is_file():
        return sh_path
    if py_path.exists() and py_path.is_file():
        return py_path

    raise SkillRegistryError(
        f"skill tool script not found for '{tool_name}', expected {sh_path.name} or {py_path.name}"
        f" under {scripts_dir}"
    )


def _validate_skill_mode(raw: Any, *, skill_path: Path) -> str:
    if raw is None:
        return SKILL_MODE_SYNC
    mode = str(raw).strip().lower()
    if mode in {SKILL_MODE_SYNC, SKILL_MODE_PYSKILL}:
        return mode
    raise SkillRegistryError(
        f"{SKILL_MODE_FIELD} must be one of ['{SKILL_MODE_SYNC}', '{SKILL_MODE_PYSKILL}']: {skill_path}"
    )


def load_skill_catalog(*, workspace_root: Path, skills_root: str, agent: str) -> dict[str, dict[str, Any]]:
    agent_root = (workspace_root / skills_root / agent).resolve()
    if not agent_root.exists() or not agent_root.is_dir():
        return {}

    catalog: dict[str, dict[str, Any]] = {}

    for skill_dir in sorted(agent_root.iterdir(), key=lambda item: item.as_posix()):
        if not skill_dir.is_dir() or skill_dir.name.startswith("."):
            continue

        skill_file = skill_dir / SKILL_FILENAME
        if not skill_file.exists() or not skill_file.is_file():
            continue

        frontmatter, _ = _parse_frontmatter(skill_file)

        for field in REQUIRED_FRONTMATTER_FIELDS:
            if field not in frontmatter:
                raise SkillRegistryError(f"missing required field '{field}': {skill_file}")

        name = str(frontmatter.get("name", "")).strip()
        description = str(frontmatter.get("description", "")).strip()
        when_to_use = str(frontmatter.get("when_to_use", "")).strip()
        skill_mode = _validate_skill_mode(frontmatter.get(SKILL_MODE_FIELD), skill_path=skill_file)
        allowed_tools = _validate_allowed_tools(frontmatter.get(FM_ALLOWED_TOOLS), skill_path=skill_file)

        if not name:
            raise SkillRegistryError(f"field 'name' must be non-empty: {skill_file}")
        if not description:
            raise SkillRegistryError(f"field 'description' must be non-empty: {skill_file}")
        if not when_to_use:
            raise SkillRegistryError(f"field 'when_to_use' must be non-empty: {skill_file}")
        if skill_mode == SKILL_MODE_PYSKILL and len(allowed_tools) != 1:
            raise SkillRegistryError(
                f"{SKILL_MODE_FIELD}=pyskill requires exactly one allowed tool: {skill_file}"
            )

        normalized_key = normalize_skill_key(name)
        if not normalized_key:
            raise SkillRegistryError(f"skill name is not normalizable: {skill_file}")
        if normalized_key in catalog:
            other = catalog[normalized_key].get("path", "")
            raise SkillRegistryError(f"duplicate skill name: {name!r}; conflicts with {other}")

        script_map: dict[str, str] = {}
        script_map_abs: dict[str, str] = {}
        for tool_name in allowed_tools:
            script_path = _resolve_script_for_tool(skill_dir=skill_dir, tool_name=tool_name)
            if skill_mode == SKILL_MODE_PYSKILL and script_path.suffix != ".py":
                raise SkillRegistryError(
                    f"{SKILL_MODE_FIELD}=pyskill requires python script entry (.py): {script_path}"
                )
            script_map[tool_name] = script_path.relative_to(workspace_root).as_posix()
            script_map_abs[tool_name] = str(script_path.resolve())

        catalog[normalized_key] = {
            "key": normalized_key,
            "name": name,
            "description": description,
            "when_to_use": when_to_use,
            "skill_mode": skill_mode,
            "allowed_tools": list(allowed_tools),
            "path": skill_file.relative_to(workspace_root).as_posix(),
            "skill_dir": skill_dir.relative_to(workspace_root).as_posix(),
            "skill_file_abs": str(skill_file.resolve()),
            "skill_dir_abs": str(skill_dir.resolve()),
            "scripts": script_map,
            "scripts_abs": script_map_abs,
        }

    return catalog


def build_skill_registry_text(*, catalog: dict[str, dict[str, Any]], agent: str) -> str:
    title = f"### {agent.capitalize()} Skill Registry (Metadata For Selection)"
    entries: list[dict[str, Any]] = []

    for entry in sorted(catalog.values(), key=lambda item: str(item.get("name", "")).lower()):
        entries.append(
            {
                "name": str(entry.get("name", "")).strip(),
                "description": str(entry.get("description", "")).strip(),
                "when_to_use": str(entry.get("when_to_use", "")).strip(),
                SKILL_MODE_FIELD: str(entry.get("skill_mode", SKILL_MODE_SYNC)).strip() or SKILL_MODE_SYNC,
                "path": str(entry.get("path", "")).strip(),
                FM_ALLOWED_TOOLS: list(entry.get("allowed_tools", [])),
            }
        )

    return "\n\n".join([title, json.dumps(entries, ensure_ascii=False, indent=2)]).strip()
