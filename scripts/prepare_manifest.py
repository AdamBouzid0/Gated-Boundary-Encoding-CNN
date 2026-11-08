"""Pair NIfTI scans/masks and declare patient grouping explicitly."""

import argparse
import csv
import os
import re
from pathlib import Path


def stem(path: Path) -> str:
    name = path.name.removesuffix(".gz").removesuffix(".nii")
    return name.removesuffix("_0000")  # optional nnU-Net single-modality suffix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/manifest.csv"))
    grouping = parser.add_mutually_exclusive_group(required=True)
    grouping.add_argument("--patient-regex", help="Regex on case ID; first capture group is patient ID")
    grouping.add_argument("--one-case-per-patient", action="store_true",
                          help="Explicitly assert every volume belongs to a different patient")
    parser.add_argument("--expected-cases", type=int,
                        help="Optional expected count; no dataset size is assumed")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error("Output already exists; choose a new path or pass --overwrite")
    images = sorted(path for path in args.images.iterdir() if path.name.endswith((".nii", ".nii.gz")))
    if not images:
        parser.error("No .nii/.nii.gz image files found")
    rows, seen_ids = [], set()
    for image in images:
        case_id = stem(image)
        if case_id in seen_ids:
            parser.error(f"Multiple image files resolve to case ID {case_id}; this loader expects one scalar modality")
        seen_ids.add(case_id)
        candidates = {args.labels / image.name, args.labels / f"{case_id}.nii.gz", args.labels / f"{case_id}.nii"}
        candidates = [path for path in candidates if path.is_file()]
        if len(candidates) != 1:
            parser.error(f"Need exactly one matching label for {image.name}; found {len(candidates)}")
        if args.patient_regex:
            match = re.search(args.patient_regex, case_id)
            if match is None or match.lastindex is None:
                parser.error(f"patient-regex must capture a patient ID for {case_id}")
            patient_id = match.group(1)
        else:
            patient_id = case_id
        rows.append({"case_id": case_id, "patient_id": patient_id,
                     "image": os.path.relpath(image.resolve(), args.output.parent.resolve()),
                     "label": os.path.relpath(candidates[0].resolve(), args.output.parent.resolve())})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_id", "patient_id", "image", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} cases from {len({row['patient_id'] for row in rows})} patients to {args.output}")
    if args.expected_cases is not None and len(rows) != args.expected_cases:
        print(f"NOTE: expected {args.expected_cases} examples; verify what an example means in this dataset.")


if __name__ == "__main__":
    main()

