"""Validate a bounded sample of NIH ChestX-ray14 images.

This module is intended for educational dataset QA. It does not modify images
and does not perform diagnosis.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

REQUIRED_MANIFEST_COLUMNS = {"image_index", "image_path", "split"}


def read_manifest(manifest_path: Path, limit: int | None = None) -> list[dict[str, str]]:
    """Read a manifest CSV and return up to `limit` rows in file order."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows: list[dict[str, str]] = []

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = REQUIRED_MANIFEST_COLUMNS.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Manifest is missing required columns: {sorted(missing_columns)}"
            )

        for row in reader:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break

    if not rows:
        raise ValueError(f"Manifest contains no image rows: {manifest_path}")

    return rows


def validate_image(image_path: Path) -> dict[str, object]:
    """Validate one image and return a structured result."""
    result: dict[str, object] = {
        "image_index": image_path.name,
        "image_path": image_path.as_posix(),
        "valid": False,
        "width": None,
        "height": None,
        "mode": None,
        "error": None,
    }

    if not image_path.is_file():
        result["error"] = "file_not_found"
        return result

    if image_path.stat().st_size == 0:
        result["error"] = "empty_file"
        return result

    try:
        with Image.open(image_path) as image:
            image.load()
            result["valid"] = True
            result["width"] = image.width
            result["height"] = image.height
            result["mode"] = image.mode
    except (UnidentifiedImageError, OSError, ValueError) as error:
        result["error"] = f"unreadable_image: {type(error).__name__}"

    return result


def validate_manifest_images(
    manifest_path: Path,
    output_path: Path,
    limit: int | None = None,
) -> dict[str, object]:
    """Validate manifest images and write a JSON inventory report."""
    rows = read_manifest(manifest_path, limit=limit)

    results = [validate_image(Path(row["image_path"])) for row in rows]
    valid_count = sum(result["valid"] is True for result in results)

    report = {
        "manifest_path": manifest_path.as_posix(),
        "requested_limit": limit,
        "validated_count": len(results),
        "valid_count": valid_count,
        "invalid_count": len(results) - valid_count,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/nih_chestxray14_manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/nih_image_validation.json"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=64,
        help="Maximum number of manifest rows to validate.",
    )
    parser.add_argument(
        "--fail-on-invalid",
        action="store_true",
        help="Return a non-zero exit code if any sampled image is invalid.",
    )
    args = parser.parse_args()

    if args.limit <= 0:
        print("FAIL: --limit must be greater than zero.", file=sys.stderr)
        return 1

    try:
        report = validate_manifest_images(
            manifest_path=args.manifest,
            output_path=args.output,
            limit=args.limit,
        )
    except (OSError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print(
        "PASS: validated "
        f"{report['valid_count']}/{report['validated_count']} images; "
        f"{report['invalid_count']} invalid."
    )
    print(f"Report written to {args.output}")

    if args.fail_on_invalid and report["invalid_count"]:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())