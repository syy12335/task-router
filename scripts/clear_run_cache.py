from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_run_root(*, config_path: Path, run_root_arg: str | None) -> Path:
    # 优先使用 --run-root；否则回退到 config.paths.run_root。
    if run_root_arg:
        candidate = Path(run_root_arg)
        return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()

    run_root_rel = "var/runs"
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        paths = config.get("paths", {}) if isinstance(config, dict) else {}
        run_root_rel = str(paths.get("run_root", run_root_rel))

    candidate = Path(run_root_rel)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def collect_run_dirs(run_root: Path) -> list[Path]:
    if not run_root.exists():
        return []
    return sorted(
        [p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("run_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear run-level cache folders (run_*)")
    parser.add_argument("--config", default="configs/graph.yaml", help="Path to graph config")
    parser.add_argument("--run-root", default=None, help="Override run root path")
    parser.add_argument("--keep", type=int, default=0, help="Keep latest N runs")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed")
    args = parser.parse_args()

    if args.keep < 0:
        raise ValueError("--keep must be >= 0")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()

    run_root = resolve_run_root(config_path=config_path, run_root_arg=args.run_root)
    run_dirs = collect_run_dirs(run_root)

    to_remove = run_dirs[args.keep :]

    print(f"Run root: {run_root}")
    print(f"Detected run dirs: {len(run_dirs)}")
    print(f"Keep latest: {args.keep}")
    print(f"To remove: {len(to_remove)}")

    if not to_remove:
        print("Nothing to remove.")
        return

    removed = 0
    for run_dir in to_remove:
        if args.dry_run:
            print(f"[DRY-RUN] remove {run_dir}")
            continue
        shutil.rmtree(run_dir)
        removed += 1
        print(f"Removed {run_dir}")

    if args.dry_run:
        print("Dry-run complete.")
    else:
        print(f"Done. Removed {removed} run directories.")


if __name__ == "__main__":
    main()
