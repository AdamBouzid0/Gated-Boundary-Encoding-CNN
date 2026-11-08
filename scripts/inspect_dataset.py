import argparse
import json
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

from pancreas_seg.data import read_manifest
from pancreas_seg.geometry import compare_grids


def range_summary(values):
    values = np.asarray(values)
    return {"min": values.min(axis=0).tolist(),
            "median": np.median(values, axis=0).tolist(),
            "max": values.max(axis=0).tolist()}


def main():
    parser = argparse.ArgumentParser(description="Inspect NIfTI shapes, spacing, label IDs and alignment")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--limit", type=int, default=5, help="0 inspects every case")
    parser.add_argument("--headers-only", action="store_true", help="Inspect geometry without loading mask voxels")
    parser.add_argument("--grid-tolerance-voxels", type=float, default=0.0,
                        help="Maximum allowed header offset in CT voxels; strict by default, must be <0.5")
    parser.add_argument("--output", type=Path, help="Write case reports and aggregate header statistics as JSON")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if not np.isfinite(args.grid_tolerance_voxels) or not 0 <= args.grid_tolerance_voxels < 0.5:
        parser.error("--grid-tolerance-voxels must be in [0, 0.5)")
    records = read_manifest(args.manifest)
    grouping_ids = len({row["patient_id"] for row in records})
    print(f"Cases: {len(records)}; patient grouping IDs: {grouping_ids} (may be case-ID proxies)")
    reports = []
    for record in records[:args.limit or None]:
        image, label = nib.load(record["image"]), nib.load(record["label"])
        geometry = compare_grids(image.shape, image.affine, label.shape, label.affine,
                                 tolerance_voxels=args.grid_tolerance_voxels)
        report = {"case_id": record["case_id"], "patient_id": record["patient_id"],
                  "image_shape": list(image.shape), "label_shape": list(label.shape),
                  "header_voxel_spacing": [float(value) for value in image.header.get_zooms()[:3]],
                  "header_spatial_unit": image.header.get_xyzt_units()[0],
                  "axis_codes": list(nib.aff2axcodes(image.affine)),
                  "image_affine": image.affine.tolist(), "image_dtype": str(image.get_data_dtype()),
                  "label_header_spatial_unit": label.header.get_xyzt_units()[0],
                  "aligned": geometry.compatible,
                  "max_grid_offset_voxels": geometry.max_grid_offset_voxels,
                  "grid_header_adjusted": geometry.header_adjusted}
        if not args.headers_only:
            ids, counts = np.unique(np.asarray(label.dataobj), return_counts=True)
            report["label_voxel_counts"] = {str(value): int(count) for value, count in zip(ids, counts)}
        reports.append(report)
        print(json.dumps(report))
        if not geometry.compatible:
            raise ValueError(f"Fix image/label dimensionality or alignment before training: {geometry.reason}")
    units = sorted({row["header_spatial_unit"] for row in reports})
    summary = {
        "image_shape": range_summary([row["image_shape"] for row in reports]),
        "header_voxel_spacing": range_summary([row["header_voxel_spacing"] for row in reports]),
        "header_voxel_spacing_by_unit": {
            unit: range_summary([row["header_voxel_spacing"] for row in reports if row["header_spatial_unit"] == unit])
            for unit in units},
        "axis_code_counts": dict(Counter("".join(code or "?" for code in row["axis_codes"]) for row in reports)),
        "header_spatial_unit_counts": dict(Counter(row["header_spatial_unit"] for row in reports)),
        "image_dtype_counts": dict(Counter(row["image_dtype"] for row in reports)),
    }
    print(json.dumps({"inspected_cases": len(reports), "summary": summary}))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "manifest": str(Path(args.manifest).resolve()), "manifest_cases": len(records),
            "patient_grouping_ids": grouping_ids, "headers_only": args.headers_only,
            "grid_tolerance_voxels": args.grid_tolerance_voxels,
            "inspected_cases": len(reports), "cases": reports, "summary": summary,
            "spacing_note": "Spacing values are raw header values in declared units; unknown units are not assumed to be mm.",
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Inspection report: {args.output.resolve()}")
    if not args.headers_only:
        print("Use the dataset documentation to identify which printed class ID(s) denote pancreas.")


if __name__ == "__main__":
    main()

