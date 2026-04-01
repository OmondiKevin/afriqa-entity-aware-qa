#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

import gdown


ALLOWED_TOP_LEVEL_DIRS = ("logs", "metrics", "predictions")
METRIC_EXTS = {".json", ".csv"}


@dataclass(frozen=True)
class PlannedDownload:
    file_id: str
    drive_path: PurePosixPath
    local_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pull experiment artifacts from a shared Google Drive folder into "
            "outputs_colab while preserving logs/metrics/predictions structure."
        )
    )
    parser.add_argument(
        "--drive-folder-url",
        required=True,
        help="Google Drive shared folder URL (root that contains logs/metrics/predictions).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_colab",
        help="Local target directory (default: outputs_colab).",
    )
    parser.add_argument(
        "--organize-existing",
        action="store_true",
        help=(
            "Move misplaced files from output-dir root into logs/metrics/predictions "
            "based on extension before pulling."
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing logs/metrics/predictions directories before pulling.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading files.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce gdown progress output.",
    )
    return parser.parse_args()


def ensure_structure(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ALLOWED_TOP_LEVEL_DIRS:
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def clean_structure(output_dir: Path) -> None:
    for name in ALLOWED_TOP_LEVEL_DIRS:
        d = output_dir / name
        if d.exists():
            shutil.rmtree(d)
    ensure_structure(output_dir)


def _classify_root_file(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix == ".log":
        return "logs"
    if suffix in METRIC_EXTS:
        return "metrics"
    if suffix == ".jsonl":
        return "predictions"
    return None


def organize_existing_root_files(output_dir: Path) -> tuple[int, int]:
    moved = 0
    skipped = 0
    for item in output_dir.iterdir():
        if item.is_dir():
            continue
        destination_group = _classify_root_file(item)
        if destination_group is None:
            skipped += 1
            continue
        target_path = output_dir / destination_group / item.name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(item), str(target_path))
        moved += 1
    return moved, skipped


def list_drive_files(drive_folder_url: str, quiet: bool = False) -> list:
    files = gdown.download_folder(
        url=drive_folder_url,
        skip_download=True,
        quiet=quiet,
        use_cookies=False,
        remaining_ok=True,
    )
    if not files:
        raise RuntimeError("No files discovered from the provided Google Drive folder.")
    return files


def plan_downloads(files: Iterable, output_dir: Path) -> list[PlannedDownload]:
    planned: list[PlannedDownload] = []
    for f in files:
        drive_path = PurePosixPath(f.path)
        if not drive_path.parts:
            continue
        top = drive_path.parts[0]
        if top not in ALLOWED_TOP_LEVEL_DIRS:
            continue
        if "checkpoints" in drive_path.parts:
            continue
        local_path = output_dir / top / PurePosixPath(*drive_path.parts[1:])
        planned.append(PlannedDownload(file_id=f.id, drive_path=drive_path, local_path=local_path))
    return planned


def execute_downloads(planned: list[PlannedDownload], quiet: bool = False, dry_run: bool = False) -> tuple[int, int]:
    downloaded = 0
    failed = 0
    for item in planned:
        if dry_run:
            print(f"[dry-run] {item.drive_path} -> {item.local_path}")
            continue
        item.local_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://drive.google.com/uc?id={item.file_id}"
        try:
            gdown.download(
                url=url,
                output=str(item.local_path),
                quiet=quiet,
                use_cookies=False,
                fuzzy=True,
                resume=True,
            )
            downloaded += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[warn] failed: {item.drive_path} ({exc})")
    return downloaded, failed


def verify_structure(output_dir: Path) -> dict[str, list[Path]]:
    invalid: dict[str, list[Path]] = {name: [] for name in ALLOWED_TOP_LEVEL_DIRS}

    for p in (output_dir / "logs").rglob("*"):
        if p.is_file() and p.suffix.lower() != ".log":
            invalid["logs"].append(p)

    for p in (output_dir / "metrics").rglob("*"):
        if p.is_file() and p.suffix.lower() not in METRIC_EXTS:
            invalid["metrics"].append(p)

    for p in (output_dir / "predictions").rglob("*"):
        if p.is_file() and p.suffix.lower() != ".jsonl":
            invalid["predictions"].append(p)

    return invalid


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    ensure_structure(output_dir)
    if args.clean:
        clean_structure(output_dir)

    if args.organize_existing:
        moved, skipped = organize_existing_root_files(output_dir)
        print(f"Organized root files: moved={moved}, skipped={skipped}")

    files = list_drive_files(args.drive_folder_url, quiet=args.quiet)
    planned = plan_downloads(files, output_dir)
    if not planned:
        raise RuntimeError(
            "No downloadable files matched logs/metrics/predictions in the provided Drive folder."
        )

    counts = {name: 0 for name in ALLOWED_TOP_LEVEL_DIRS}
    for item in planned:
        counts[item.drive_path.parts[0]] += 1

    print("Planned downloads:")
    for name in ALLOWED_TOP_LEVEL_DIRS:
        print(f"  {name}: {counts[name]} files")

    downloaded, failed = execute_downloads(planned, quiet=args.quiet, dry_run=args.dry_run)
    if args.dry_run:
        print("Dry-run complete. No files downloaded.")
        return

    print(f"Downloaded files: {downloaded}")
    print(f"Failed downloads: {failed}")

    invalid = verify_structure(output_dir)
    print("Verification:")
    for name in ALLOWED_TOP_LEVEL_DIRS:
        if invalid[name]:
            print(f"  {name}: invalid files={len(invalid[name])}")
            for p in invalid[name][:5]:
                print(f"    - {p}")
        else:
            print(f"  {name}: OK")


if __name__ == "__main__":
    main()
