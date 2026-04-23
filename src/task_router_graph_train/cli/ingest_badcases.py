from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..dataset import sanitize_environment_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest production bad-case traces into a standardized feedback pool.")
    parser.add_argument("--input", required=True, help="Path to source trajectory jsonl.")
    parser.add_argument("--output", required=True, help="Path to normalized bad-case pool jsonl.")
    parser.add_argument("--source", default="production_sampled", help="Bad-case source label.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for raw_line in input_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            continue
        sample_id = str(payload.get("sample_id", "")).strip()
        user_input = str(payload.get("user_input", "")).strip()
        environment = payload.get("environment", {})
        if not isinstance(environment, dict):
            environment = {}
        tags = payload.get("error_tags", payload.get("tags", []))
        if not isinstance(tags, list):
            tags = []
        rows.append(
            {
                "sample_id": sample_id or f"badcase_{len(rows)+1:06d}",
                "source": str(args.source),
                "user_input": user_input,
                "environment_raw": environment,
                "environment_formal": sanitize_environment_payload(environment)[0],
                "environment": environment,
                "policy_output_text": str(payload.get("policy_output_text", payload.get("prediction", payload.get("response", "")))),
                "policy_output_action": payload.get("policy_output_action", {}),
                "error_code": str(payload.get("error_code", "")).strip(),
                "error_tags": [str(item).strip() for item in tags if str(item).strip()],
                "trace_id": str(payload.get("trace_id", payload.get("run_id", ""))).strip(),
                "metadata": payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {},
            }
        )

    lines = [json.dumps(item, ensure_ascii=False) for item in rows]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(
        json.dumps(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "record_count": len(rows),
                "source": str(args.source),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
