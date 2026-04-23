from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from task_router_graph.schema import Environment

PACKAGE_ROOT = Path(__file__).resolve().parent
DOCS_ROOT = PACKAGE_ROOT / "docs"
ASSETS_ROOT = PACKAGE_ROOT / "assets"
CONFIGS_ROOT = PACKAGE_ROOT / "configs"
REPO_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_RUNTIME_SKILLS_ROOT = "src/task_router_graph/skills"


def build_controller_state_input(
    *,
    user_input: str,
    environment_payload: dict[str, Any],
    workspace_root: Path | None = None,
    skills_root: str = DEFAULT_RUNTIME_SKILLS_ROOT,
) -> dict[str, Any]:
    # controller 训练态真正看到的 state 真源就在这里。
    # 它把 runtime Environment 收敛成训练时可比较、可重复构造的正式输入。
    runtime_root = _resolve_runtime_root(workspace_root)
    environment = Environment.from_dict(copy.deepcopy(environment_payload))
    return {
        "USER_INPUT": str(user_input),
        "ENVIRONMENT_JSON": environment.build_controller_context(
            default_task_limit=5,
            compress=False,
        ),
        "SKILLS_INDEX": _build_skill_registry_preview(
            workspace_root=runtime_root,
            skills_root=skills_root,
            agent="controller",
        ),
    }


def build_reply_state_input(
    *,
    user_input: str,
    environment_payload: dict[str, Any],
    final_task: dict[str, Any],
) -> dict[str, Any]:
    # reply 只看 formal state 视图和最终任务，不应直接接触 verifier sidecar 扩展信息。
    environment = Environment.from_dict(copy.deepcopy(environment_payload))
    return {
        "USER_INPUT": str(user_input),
        "FINAL_TASK_JSON": copy.deepcopy(final_task),
        "ENVIRONMENT_JSON": environment.build_context_view(
            task_limit=None,
            include_user_input=True,
            include_task=True,
            include_reply=True,
            include_trace=False,
            compress=False,
        ),
    }


def _resolve_runtime_root(workspace_root: Path | None) -> Path:
    if workspace_root is None:
        return REPO_ROOT.resolve()
    return workspace_root.resolve()


def _build_skill_registry_preview(*, workspace_root: Path, skills_root: str, agent: str) -> str:
    # skill registry preview 只是训练态的技能元信息摘要，
    # 用来帮助模型理解“有哪些候选技能”，不是运行时真实执行结果。
    agent_root = (workspace_root / skills_root / agent).resolve()
    entries: list[dict[str, Any]] = []
    if not agent_root.exists():
        return "[]"

    for skill_dir in sorted(agent_root.iterdir(), key=lambda item: item.as_posix()):
        if not skill_dir.is_dir() or skill_dir.name.startswith("."):
            continue
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue
        frontmatter = _parse_skill_frontmatter(skill_file)
        entries.append(
            {
                "name": str(frontmatter.get("name", "")).strip(),
                "description": str(frontmatter.get("description", "")).strip(),
                "when_to_use": str(frontmatter.get("when_to_use", "")).strip(),
                "skill-mode": str(frontmatter.get("skill-mode", "sync")).strip() or "sync",
                "path": skill_file.relative_to(workspace_root).as_posix(),
                "allowed-tools": list(frontmatter.get("allowed-tools", [])),
            }
        )

    title = f"### {agent.capitalize()} Skill Registry (Metadata For Selection)"
    return "\n\n".join([title, json.dumps(entries, ensure_ascii=False, indent=2)]).strip()


def _parse_skill_frontmatter(skill_path: Path) -> dict[str, Any]:
    text = skill_path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}
    payload: dict[str, Any] = {}
    for raw_line in text[4:end].splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        normalized_key = key.strip()
        value = raw_value.strip()
        if normalized_key == "allowed-tools":
            payload[normalized_key] = [] if not value else json.loads(value)
            continue
        payload[normalized_key] = value.strip('"')
    return payload
