from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"jsonl row must be an object: {path}")
        rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: list[Any]) -> None:
    serialized_rows = [_normalize_row(row) for row in rows]
    lines = [json.dumps(row, ensure_ascii=False) for row in serialized_rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _normalize_row(row: Any) -> dict[str, Any]:
    if is_dataclass(row):
        payload = asdict(row)
    elif hasattr(row, "to_dict") and callable(getattr(row, "to_dict")):
        payload = row.to_dict()
    else:
        payload = row
    if not isinstance(payload, dict):
        raise ValueError("jsonl rows must serialize to object payloads")
    return payload
