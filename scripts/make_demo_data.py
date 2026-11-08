"""Create SYNTHETIC ellipsoid volumes to test plumbing, never benchmark data."""

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np


def make_demo(root: Path, cases: int = 6, shape=(40, 40, 40), seed: int = 123) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    coordinates = np.indices(shape)
    rows = []
    for index in range(cases):
        case_id = f"synthetic_{index:03d}"
        center = np.asarray(shape) / 2 + rng.uniform(-3, 3, size=3)
        radii = rng.uniform(5, 9, size=3)
        label = (((coordinates - center[:, None, None, None]) / radii[:, None, None, None])**2).sum(0) <= 1
        image = rng.normal(0, 0.1, size=shape).astype(np.float32) + label.astype(np.float32)
        affine = np.diag([1.5, 1.5, 2.5, 1.0])
        image_path, label_path = root / f"{case_id}.nii.gz", root / f"{case_id}_label.nii.gz"
        if image_path.exists() or label_path.exists():
            raise FileExistsError("Demo files already exist; choose a fresh demo directory")
        image_file = nib.Nifti1Image(image, affine)
        label_file = nib.Nifti1Image(label.astype(np.uint8), affine)
        image_file.header.set_xyzt_units("mm")
        label_file.header.set_xyzt_units("mm")
        nib.save(image_file, image_path)
        nib.save(label_file, label_path)
        rows.append({"case_id": case_id, "patient_id": case_id,
                     "image": image_path.name, "label": label_path.name})
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/demo"))
    parser.add_argument("--cases", type=int, default=6)
    args = parser.parse_args()
    if args.cases < 3:
        parser.error("At least three synthetic patients are needed")
    print(make_demo(args.output, args.cases))


if __name__ == "__main__":
    main()

