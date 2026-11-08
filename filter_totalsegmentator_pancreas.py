#!/usr/bin/env python3
"""Audit TotalSegmentator case folders and select nonempty pancreas masks.

Standalone dependencies: numpy and nibabel. Original data are read-only.
CT headers are checked; only the much smaller pancreas-mask voxel arrays
are loaded. Output manifests work with this project's read_manifest().
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

# Keep this audit runnable without installing the training dependencies.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pancreas_seg.geometry import compare_grids


MANIFEST_FIELDS = ["case_id", "patient_id", "image", "label", "official_split"]
AUDIT_FIELDS = [
    "case_id", "patient_id", "patient_id_source", "metadata_row_found", "official_split", "status", "reason",
    "foreground_voxels", "mask_values", "mask_shape", "ct_shape", "voxel_spacing",
    "spatial_unit", "touches_volume_border", "bbox_start", "bbox_end_exclusive",
    "max_grid_offset_voxels", "grid_header_adjusted",
    "image", "label",
]


def read_metadata(path: Path | None, case_column: str | None = None,
                  split_column: str = "split", patient_column: str = "patient_id") -> dict:
    """Read comma/semicolon/tab CSV. Preserve source split labels verbatim."""
    if path is None:
        return {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(8192)
        stream.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(stream, dialect=dialect)
        columns = reader.fieldnames or []
        chosen = case_column or next((name for name in ("image_id", "case_id", "id") if name in columns), None)
        if chosen not in columns:
            raise ValueError(f"Cannot find metadata case ID column in {columns}; use --metadata-case-column")
        metadata = {}
        for row in reader:
            case_id = (row.get(chosen) or "").strip()
            if not case_id:
                raise ValueError("Metadata contains an empty case ID")
            if case_id in metadata:
                raise ValueError(f"Duplicate case ID in metadata: {case_id}")
            metadata[case_id] = {
                "official_split": (row.get(split_column) or "").strip(),
                "patient_id": (row.get(patient_column) or "").strip(),
            }
    return metadata


def audit_case(case: Path, metadata: dict | None = None, min_voxels: int = 1,
               foreground_value: int = 1, affine_atol: float = 1e-4,
               grid_tolerance_voxels: float = 0.0) -> dict:
    """Classify one case; missing, unreadable and empty masks are different."""
    metadata_row_found = metadata is not None
    metadata = metadata or {}
    image_path = case / "ct.nii.gz"
    mask_path = case / "segmentations" / "pancreas.nii.gz"
    patient_id = metadata.get("patient_id") or case.name
    row = {field: "" for field in AUDIT_FIELDS}
    row.update(case_id=case.name, patient_id=patient_id,
               patient_id_source="metadata" if metadata.get("patient_id") else "case_id_proxy",
               metadata_row_found=metadata_row_found,
               official_split=metadata.get("official_split", ""),
               image=str(image_path.resolve()), label=str(mask_path.resolve()))

    def finish(status, reason=""):
        row.update(status=status, reason=reason)
        return row

    if not image_path.is_file() and not mask_path.is_file():
        return finish("missing_both", "ct.nii.gz and segmentations/pancreas.nii.gz are missing")
    if not image_path.is_file():
        return finish("missing_ct", "ct.nii.gz is missing")
    if not mask_path.is_file():
        return finish("missing_mask", "Pancreas mask is missing; this is not an empty annotation")
    try:
        image = nib.load(image_path)
        row["ct_shape"] = json.dumps(list(image.shape))
    except Exception as error:
        return finish("unreadable_ct", f"{type(error).__name__}: {error}")
    try:
        mask_image = nib.load(mask_path)
        row["mask_shape"] = json.dumps(list(mask_image.shape))
        row["voxel_spacing"] = json.dumps([float(v) for v in mask_image.header.get_zooms()[:3]])
        row["spatial_unit"] = mask_image.header.get_xyzt_units()[0]
    except Exception as error:
        return finish("unreadable_mask", f"{type(error).__name__}: {error}")
    geometry = compare_grids(image.shape, image.affine, mask_image.shape, mask_image.affine,
                             tolerance_voxels=grid_tolerance_voxels, affine_atol=affine_atol)
    row["max_grid_offset_voxels"] = geometry.max_grid_offset_voxels
    row["grid_header_adjusted"] = geometry.header_adjusted
    if not geometry.compatible:
        return finish(geometry.status, geometry.reason)
    image_unit = image.header.get_xyzt_units()[0]
    mask_unit = mask_image.header.get_xyzt_units()[0]
    if image_unit != "unknown" and mask_unit != "unknown" and image_unit != mask_unit:
        return finish("unit_mismatch", "CT and mask declare different known spatial units")
    try:
        values = np.asanyarray(mask_image.dataobj)
        if not np.isfinite(values).all():
            return finish("invalid_mask_values", "Mask contains NaN or infinity")
        unique = np.unique(values)
        row["mask_values"] = json.dumps([float(value) for value in unique[:32]])
        # These files are individual organ masks, not merged multiclass labels.
        # Do not silently interpret arbitrary nonzero values as pancreas.
        if not np.isin(unique, [0, foreground_value]).all():
            return finish("invalid_mask_values", f"Expected only 0 and {foreground_value}; inspect mask encoding")
        foreground = values == foreground_value
        count = int(np.count_nonzero(foreground))
        row["foreground_voxels"] = count
    except Exception as error:
        return finish("unreadable_mask", f"{type(error).__name__}: {error}")
    if count == 0:
        row["touches_volume_border"] = False
        return finish("empty_mask", "No annotated pancreas foreground voxels")
    start, end = [], []
    for axis in range(3):
        occupied = np.flatnonzero(foreground.any(axis=tuple(a for a in range(3) if a != axis)))
        start.append(int(occupied[0]))
        end.append(int(occupied[-1]) + 1)
    row["bbox_start"], row["bbox_end_exclusive"] = json.dumps(start), json.dumps(end)
    row["touches_volume_border"] = any(a == 0 or b == size for a, b, size in zip(start, end, foreground.shape))
    if count < min_voxels:
        return finish("below_min_voxels", f"Foreground count is below the requested minimum {min_voxels}")
    return finish("kept")


def relative_or_absolute(path: str, parent: Path) -> str:
    try:
        return Path(os.path.relpath(path, parent)).as_posix()
    except ValueError:  # e.g. Windows paths on different drives
        return path


def write_outputs(rows: list[dict], output_dir: Path, *, root: Path,
                   metadata_path: Path | None, min_voxels: int, foreground_value: int,
                   grid_tolerance_voxels: float = 0.0) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    kept = [row for row in rows if row["status"] == "kept"]
    valid = [row for row in rows if row["status"] in {"kept", "empty_mask", "below_min_voxels"}]
    for filename, selected in (("manifest.csv", kept), ("valid_manifest.csv", valid)):
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for row in selected:
                record = {field: row[field] for field in MANIFEST_FIELDS}
                for field in ("image", "label"):
                    record[field] = relative_or_absolute(record[field], output_dir.resolve())
                writer.writerow(record)
    with (output_dir / "audit.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "kept_case_ids.txt").open("w", encoding="utf-8") as stream:
        stream.write("".join(row["case_id"] + "\n" for row in kept))
    summary = {
        "dataset": "TotalSegmentator separate-organ NIfTI layout",
        "root": str(root.resolve()), "metadata": str(metadata_path) if metadata_path else None,
        "total_cases_found": len(rows), "kept_cases": len(kept),
        "valid_cases_including_empty": len(valid), "status_counts": dict(Counter(row["status"] for row in rows)),
        "min_voxels": min_voxels, "foreground_value": foreground_value,
        "grid_tolerance_voxels": grid_tolerance_voxels,
        "grid_header_adjusted_cases": sum(row["grid_header_adjusted"] is True for row in valid),
        "border_touching_nonempty_cases": sum(row["touches_volume_border"] is True for row in rows),
        "patient_id_sources": dict(Counter(row["patient_id_source"] for row in rows)),
        "metadata_rows_matched": sum(row["metadata_row_found"] for row in rows),
        "cases_without_metadata_row": [row["case_id"] for row in rows if not row["metadata_row_found"]] if metadata_path else [],
        "official_split_counts_kept": dict(Counter(row["official_split"] or "unspecified" for row in kept)),
        "ct_voxel_payload_checked": False,
        "notes": [
            "Counts come from decompressed mask voxel values, not .nii.gz byte sizes.",
            "A zero annotation does not prove anatomical absence; missing masks have a separate status.",
            "Nonempty does not guarantee the entire pancreas is in the field of view.",
            "A case ID is only a patient-group proxy when no patient mapping was supplied.",
            "The trainer preserves official_split assignments when freezing the curated manifest's split JSON.",
            "Accepted approximate grids use the CT affine for the mask in memory; original source files are never modified.",
            "Positive-only evaluation is conditional on annotated pancreas presence; retain empty cases if evaluating absence handling.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Directory containing sXXXX folders")
    parser.add_argument("--output-dir", type=Path, default=Path("data/totalsegmentator_filter"))
    parser.add_argument("--case-glob", default="s[0-9]*")
    parser.add_argument("--recursive", action="store_true", help="Find nested case directories; duplicate IDs are rejected")
    parser.add_argument("--min-voxels", type=int, default=1, help="1 keeps every valid nonempty mask")
    parser.add_argument("--foreground-value", type=int, default=1, help="Expected positive value in separate binary pancreas masks")
    parser.add_argument("--grid-tolerance-voxels", type=float, default=0.0,
                        help="Allow this maximum CT-voxel coordinate offset at volume corners; default 0 is strict, must be <0.5")
    parser.add_argument("--metadata", type=Path, help="Optional meta.csv; root/meta.csv is detected automatically")
    parser.add_argument("--metadata-case-column")
    parser.add_argument("--metadata-split-column", default="split")
    parser.add_argument("--metadata-patient-column", default="patient_id")
    parser.add_argument("--overwrite", action="store_true", help="Replace only this script's existing output reports")
    args = parser.parse_args(argv)
    if args.min_voxels < 1 or args.foreground_value <= 0:
        parser.error("min-voxels and foreground-value must be positive")
    if not np.isfinite(args.grid_tolerance_voxels) or not 0 <= args.grid_tolerance_voxels < 0.5:
        parser.error("--grid-tolerance-voxels must be in [0, 0.5)")
    if not args.root.is_dir():
        parser.error(f"Dataset root is not a directory: {args.root}")
    names = ["manifest.csv", "valid_manifest.csv", "audit.csv", "summary.json", "kept_case_ids.txt"]
    if not args.overwrite and any((args.output_dir / name).exists() for name in names):
        parser.error("Output reports already exist; choose a new --output-dir or pass --overwrite")
    paths = args.root.rglob(args.case_glob) if args.recursive else args.root.glob(args.case_glob)
    cases = sorted(path for path in paths if path.is_dir())
    if not cases:
        parser.error("No case directories found; --root must contain the sXXXX folders, or use --recursive")
    if len({case.name for case in cases}) != len(cases):
        parser.error("Duplicate case IDs found in nested directories; select one dataset version/root")
    metadata_path = args.metadata
    if metadata_path is None and (args.root / "meta.csv").is_file():
        metadata_path = args.root / "meta.csv"
    try:
        metadata = read_metadata(metadata_path, args.metadata_case_column,
                                  args.metadata_split_column, args.metadata_patient_column)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    rows = []
    for index, case in enumerate(cases, 1):
        row = audit_case(case, metadata.get(case.name), args.min_voxels, args.foreground_value,
                         grid_tolerance_voxels=args.grid_tolerance_voxels)
        rows.append(row)
        if index % 25 == 0 or index == len(cases):
            print(f"Audited {index}/{len(cases)} cases", file=sys.stderr)
    summary = write_outputs(rows, args.output_dir, root=args.root, metadata_path=metadata_path,
                             min_voxels=args.min_voxels, foreground_value=args.foreground_value,
                             grid_tolerance_voxels=args.grid_tolerance_voxels)
    print(json.dumps(summary, indent=2))
    print(f"Selected-cohort manifest: {(args.output_dir / 'manifest.csv').resolve()}")
    return 0 if summary["kept_cases"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
