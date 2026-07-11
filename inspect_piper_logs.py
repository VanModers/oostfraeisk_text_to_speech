#!/usr/bin/env python3
"""Create a small, shareable report from Piper/Lightning TensorBoard logs."""

import argparse
import csv
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Optional


DEFAULT_LOGS_DIR = Path("piper_training_multi_logs")
DEFAULT_OUTPUT_DIR = Path("piper_log_exports")


def package_version(package_name: str) -> str:
    """Return an installed package version without failing the report."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not installed"


def last_modified(path: Path) -> float:
    """Return the newest mtime anywhere inside a run directory."""
    mtimes = [path.stat().st_mtime]
    mtimes.extend(item.stat().st_mtime for item in path.rglob("*") if item.exists())
    return max(mtimes)


def find_run(logs_dir: Path, requested_run: Optional[str]) -> Path:
    """Resolve an explicit run or select the most recently modified run."""
    if requested_run:
        requested_path = Path(requested_run).expanduser()
        candidates = [
            requested_path,
            logs_dir / requested_run,
            logs_dir / "lightning_logs" / requested_run,
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate.resolve()
        raise FileNotFoundError(f"Lightning run not found: {requested_run}")

    candidates = {
        path.resolve()
        for pattern in (
            logs_dir / "lightning_logs" / "version_*",
            logs_dir / "version_*",
        )
        for path in pattern.parent.glob(pattern.name)
        if path.is_dir()
    }
    if not candidates:
        raise FileNotFoundError(
            f"No version_* runs found below {logs_dir.resolve()}"
        )

    return max(candidates, key=last_modified)


def git_revision() -> str:
    """Return the current Git revision when available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def environment_info() -> dict[str, Any]:
    """Collect only the environment details relevant to Piper training."""
    info: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "git_revision": git_revision(),
        "piper_tts": package_version("piper-tts"),
        "torch": package_version("torch"),
        "lightning": package_version("lightning"),
        "tensorboard": package_version("tensorboard"),
        "librosa": package_version("librosa"),
    }

    try:
        import torch

        info["cuda_build"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        info["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception as exc:  # Environment reporting must not block extraction.
        info["torch_runtime_error"] = repr(exc)

    return info


def safe_name(value: str) -> str:
    """Make a filesystem-safe report name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "run"


def write_report(
    report_path: Path,
    run_dir: Path,
    tags: list[str],
    scalar_events: dict[str, list[Any]],
    env: dict[str, Any],
    last_points: int,
) -> dict[str, Any]:
    """Write the human-readable report and return its scalar summary."""
    summaries: dict[str, Any] = {}

    with open(report_path, "w", encoding="utf-8", newline="\n") as report:
        def write(line: str = "") -> None:
            print(line, file=report)

        write("Piper training log report")
        write("=" * 80)
        write(f"Created (UTC): {datetime.now(timezone.utc).isoformat()}")
        write(f"Run directory: {run_dir}")
        write()
        write("Environment")
        write("-" * 80)
        for key, value in env.items():
            write(f"{key}: {value}")

        write()
        write("Run files (inventory only; large files are not included in the archive)")
        write("-" * 80)
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                write(f"{path.relative_to(run_dir)}: {path.stat().st_size} bytes")

        write()
        write(f"Scalar tags: {len(tags)}")
        write("=" * 80)

        for tag in tags:
            events = scalar_events[tag]
            finite = [event for event in events if math.isfinite(event.value)]
            nonfinite_count = len(events) - len(finite)
            summary: dict[str, Any] = {
                "points": len(events),
                "nonfinite": nonfinite_count,
            }

            write()
            write(f"[{tag}]")
            write(f"points: {len(events)}")
            write(f"non-finite values: {nonfinite_count}")

            if finite:
                first = finite[0]
                last = finite[-1]
                minimum = min(finite, key=lambda event: event.value)
                maximum = max(finite, key=lambda event: event.value)
                summary.update(
                    {
                        "first": {"step": first.step, "value": first.value},
                        "last": {"step": last.step, "value": last.value},
                        "min": {"step": minimum.step, "value": minimum.value},
                        "max": {"step": maximum.step, "value": maximum.value},
                    }
                )
                write(f"first: step={first.step}, value={first.value:.9g}")
                write(f"last:  step={last.step}, value={last.value:.9g}")
                write(f"min:   step={minimum.step}, value={minimum.value:.9g}")
                write(f"max:   step={maximum.step}, value={maximum.value:.9g}")
                write(f"last {min(last_points, len(finite))} finite points:")
                for event in finite[-last_points:]:
                    write(f"  step={event.step}, value={event.value:.9g}")

            summaries[tag] = summary

    return summaries


def create_export(args: argparse.Namespace) -> tuple[Path, Path]:
    """Extract one Lightning run and return its directory and archive paths."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard is not installed. Activate the Piper training environment."
        ) from exc

    logs_dir = args.logs_dir.expanduser().resolve()
    run_dir = find_run(logs_dir, args.run)

    accumulator = EventAccumulator(
        str(run_dir),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    tags = sorted(accumulator.Tags().get("scalars", []))
    if not tags:
        raise RuntimeError(f"No TensorBoard scalar metrics found in {run_dir}")

    scalar_events = {tag: accumulator.Scalars(tag) for tag in tags}
    export_name = safe_name(f"{logs_dir.name}_{run_dir.name}")
    output_root = args.output_dir.expanduser().resolve()
    export_dir = output_root / export_name
    export_dir.mkdir(parents=True, exist_ok=True)

    report_path = export_dir / "training_log_report.txt"
    csv_path = export_dir / "training_scalars.csv"
    manifest_path = export_dir / "manifest.json"

    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["tag", "step", "wall_time", "value"])
        for tag in tags:
            for event in scalar_events[tag]:
                writer.writerow([tag, event.step, event.wall_time, event.value])

    env = environment_info()
    scalar_summaries = write_report(
        report_path,
        run_dir,
        tags,
        scalar_events,
        env,
        args.last_points,
    )

    hparams_path = run_dir / "hparams.yaml"
    if hparams_path.is_file():
        shutil.copy2(hparams_path, export_dir / "hparams.yaml")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(run_dir),
        "environment": env,
        "scalars": scalar_summaries,
    }
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)

    archive_path = output_root / f"{export_name}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(export_dir.iterdir()):
            if path.is_file():
                archive.add(path, arcname=f"{export_name}/{path.name}")

    return export_dir, archive_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract compact CSV/report files from Piper Lightning logs without "
            "including checkpoints or raw TensorBoard event files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        help="Piper/Lightning log root",
    )
    parser.add_argument(
        "--run",
        help="Specific version directory name or path; newest run if omitted",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for compact exports",
    )
    parser.add_argument(
        "--last-points",
        type=int,
        default=20,
        help="Recent finite points shown per scalar in the text report",
    )
    args = parser.parse_args()
    if args.last_points < 1:
        parser.error("--last-points must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    try:
        export_dir, archive_path = create_export(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Exported report: {export_dir}")
    print(f"Share this archive: {archive_path}")
    print(f"Archive size: {archive_path.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
