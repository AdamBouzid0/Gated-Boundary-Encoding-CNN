"""NIfTI pairs, patient-grouped splits, and paired 3D patch augmentation.

    full image/label -> preprocessing -> enlarged patch -> paired spatial warp
    -> boundary target -> final center crop -> image-only intensity changes

The halo ensures the FINAL crop cannot create an artificial organ surface.
Ground-truth labels are used for training targets / patch sampling only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.transforms import Compose, Rand3DElasticd, RandAffined, RandFlipd
from nibabel.processing import resample_from_to, resample_to_output
from torch.utils.data import Dataset, get_worker_info

from .boundaries import inner_boundary
from .geometry import compare_grids


def read_manifest(path: str | Path) -> list[dict]:
    path = Path(path).resolve()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"case_id", "patient_id", "image", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest needs columns {sorted(required)}")
        records = []
        seen_ids, seen_images = set(), set()
        for row in reader:
            if any(not (row.get(key) or "").strip() for key in required):
                raise ValueError("Every case needs nonempty case_id, patient_id, image, label")
            record = {key: row[key].strip() for key in required}
            # Keep legacy records/digests unchanged when this column is absent.
            if "official_split" in (reader.fieldnames or []):
                record["official_split"] = (row.get("official_split") or "").strip()
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", record["case_id"]):
                raise ValueError("case_id must start with a letter/digit and use only letters, digits, _, -, .")
            if record["case_id"] in seen_ids:
                raise ValueError(f"Duplicate case_id: {record['case_id']}")
            for key in ("image", "label"):
                value = Path(record[key]).expanduser()
                record[key] = str((path.parent / value).resolve() if not value.is_absolute() else value.resolve())
            if record["image"] in seen_images:
                raise ValueError(f"Image appears more than once: {record['image']}")
            seen_ids.add(record["case_id"])
            seen_images.add(record["image"])
            records.append(record)
    if not records:
        raise ValueError("The manifest is empty")
    return records


def manifest_digest(records: list[dict]) -> str:
    canonical = json.dumps(sorted(records, key=lambda row: row["case_id"]), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def official_partitions(records: list[dict]) -> dict | None:
    """Read complete source partitions; an entirely blank column is optional."""
    if not any(record.get("official_split") for record in records):
        return None
    partitions = {name: [] for name in ("train", "val", "test")}
    patient_partitions = {}
    for record in records:
        name = record.get("official_split", "")
        if name not in partitions:
            raise ValueError(f"{record['case_id']}: official_split must be train, val or test for every case")
        patient = record["patient_id"]
        if patient in patient_partitions and patient_partitions[patient] != name:
            raise ValueError(f"Patient leakage in official partitions: {patient} has conflicting split labels")
        patient_partitions[patient] = name
        partitions[name].append(record["case_id"])
    if not partitions["train"] or not partitions["test"]:
        raise ValueError("Official partitions must include nonempty train and test sets")
    return {name: sorted(case_ids) for name, case_ids in partitions.items()}


def validate_splits(records: list[dict], splits: dict) -> None:
    official = official_partitions(records)
    by_id = {record["case_id"]: record for record in records}
    seen_cases, seen_patients = set(), set()
    for name in ("train", "val", "test"):
        case_ids = splits.get(name, [])
        if not case_ids:
            raise ValueError(f"Split {name!r} is empty")
        if len(case_ids) != len(set(case_ids)) or seen_cases.intersection(case_ids):
            raise ValueError("A case occurs in multiple splits or occurs more than once")
        if not set(case_ids).issubset(by_id):
            raise ValueError("A split contains unknown case IDs")
        patients = {by_id[case_id]["patient_id"] for case_id in case_ids}
        if patients & seen_patients:
            raise ValueError("Patient leakage: a patient occurs in multiple splits")
        seen_cases.update(case_ids)
        seen_patients.update(patients)
    if seen_cases != set(by_id):
        raise ValueError("Every manifest case must belong to exactly one split")
    if splits.get("manifest_digest") != manifest_digest(records):
        raise ValueError("Manifest changed since these splits were made; use a new split file explicitly")
    if official is not None:
        # With train/test-only metadata, validation may contain source training
        # cases. Source test cases remain exactly the official test cohort.
        if set(splits["test"]) != set(official["test"]):
            raise ValueError("Frozen splits contradict official test assignments")
        if official["val"] and any(set(splits[name]) != set(official[name]) for name in ("train", "val")):
            raise ValueError("Frozen splits contradict official train/val assignments")


def make_or_load_splits(records: list[dict], path: str | Path,
                        fractions=(0.8, 0.1, 0.1), seed: int = 42) -> dict:
    path = Path(path)
    if path.exists():
        splits = json.loads(path.read_text())
        validate_splits(records, splits)
        return splits
    official = official_partitions(records)
    if official is not None:
        splits = official
        if not splits["val"]:
            groups = sorted({record["patient_id"] for record in records if record["official_split"] == "train"})
            if len(groups) < 2:
                raise ValueError("Need at least two official training patient groups to derive validation")
            random.Random(seed).shuffle(groups)
            val_fraction = fractions[1] / (fractions[0] + fractions[1])
            n_val = min(max(1, int(len(groups) * val_fraction)), len(groups) - 1)
            val_patients = set(groups[:n_val])
            splits["train"] = sorted(record["case_id"] for record in records
                                     if record["official_split"] == "train" and record["patient_id"] not in val_patients)
            splits["val"] = sorted(record["case_id"] for record in records
                                   if record["official_split"] == "train" and record["patient_id"] in val_patients)
    else:
        groups = sorted({record["patient_id"] for record in records})
        if len(groups) < 3:
            raise ValueError("At least three independent patient groups are needed for train/val/test")
        random.Random(seed).shuffle(groups)
        n_train = min(max(1, int(len(groups) * fractions[0])), len(groups) - 2)
        n_val = min(max(1, int(len(groups) * fractions[1])), len(groups) - n_train - 1)
        patient_sets = dict(zip(("train", "val", "test"),
                           (set(groups[:n_train]), set(groups[n_train:n_train + n_val]),
                            set(groups[n_train + n_val:]))))
        splits = {name: sorted(record["case_id"] for record in records if record["patient_id"] in patients)
                  for name, patients in patient_sets.items()}
    splits.update(seed=seed, manifest_digest=manifest_digest(records))
    validate_splits(records, splits)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects a split file if two runs start together.
    try:
        with path.open("x") as stream:
            json.dump(splits, stream, indent=2)
    except FileExistsError:
        splits = json.loads(path.read_text())
        validate_splits(records, splits)
    return splits


def select_records(records: list[dict], splits: dict, name: str) -> list[dict]:
    ids = set(splits[name])
    return [record for record in records if record["case_id"] in ids]


def in_millimetres(image, unknown_unit: str | None):
    unit = image.header.get_xyzt_units()[0]
    if unit == "unknown":
        unit = unknown_unit
    factors = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}
    if unit not in factors:
        raise ValueError("Unknown NIfTI spatial units: confirm them and set data.spacing_unit_if_unknown")
    if factors[unit] == 1:
        return image
    affine = image.affine.copy()
    affine[:3] *= factors[unit]
    header = image.header.copy()
    header.set_xyzt_units("mm")
    return image.__class__(np.asanyarray(image.dataobj), affine, header=header)


def load_pair(record: dict, config: dict) -> dict:
    """Validate geometry before selecting a pancreas label or resampling."""
    image_obj = nib.load(record["image"])
    label_obj = nib.load(record["label"])
    if len(image_obj.shape) != 3 or len(label_obj.shape) != 3:
        raise ValueError(f"{record['case_id']}: expected 3D scalar image and label, got "
                         f"{image_obj.shape} and {label_obj.shape}")
    image_obj = in_millimetres(image_obj, config.get("spacing_unit_if_unknown", "mm"))
    label_obj = in_millimetres(label_obj, config.get("spacing_unit_if_unknown", "mm"))
    geometry = compare_grids(image_obj.shape, image_obj.affine, label_obj.shape, label_obj.affine,
                             tolerance_voxels=config.get("grid_tolerance_voxels", 0.0))
    if not geometry.compatible:
        raise ValueError(f"{record['case_id']}: image/label grids are not aligned: {geometry.reason}")
    if geometry.header_adjusted:
        # Retain voxel-index correspondence for every compatible header
        # discrepancy, including tiny differences that can change canonical
        # orientation at a rotation tie. No interpolation or source edit occurs.
        label_obj = label_obj.__class__(label_obj.dataobj, image_obj.affine, header=label_obj.header.copy())
    image_obj, label_obj = nib.as_closest_canonical(image_obj), nib.as_closest_canonical(label_obj)
    spacing = config["target_spacing_mm"]
    if spacing is not None:
        image_obj = resample_to_output(image_obj, voxel_sizes=spacing, order=1)
        label_obj = resample_from_to(label_obj, (image_obj.shape, image_obj.affine), order=0)
    image = image_obj.get_fdata(dtype=np.float32)
    labels = label_obj.get_fdata(dtype=np.float32)
    if not np.isfinite(image).all() or not np.isfinite(labels).all():
        raise ValueError(f"{record['case_id']}: non-finite intensities or labels")
    if not np.allclose(labels, np.round(labels), atol=1e-4):
        raise ValueError(f"{record['case_id']}: segmentation labels are not discrete class IDs")
    pancreas_ids = config["pancreas_label_ids"]
    if not pancreas_ids or 0 in pancreas_ids:
        raise ValueError("Set data.pancreas_label_ids after inspecting the dataset; zero is background")
    mask = np.isin(labels, pancreas_ids).astype(np.float32)
    window = config["intensity_clip"]
    if window is not None:
        image = np.clip(image, *window)
    mean, std = float(image.mean()), float(image.std())
    image = (image - mean) / max(std, 1e-6)
    return {"image": np.ascontiguousarray(image[None], dtype=np.float32),
            "label": np.ascontiguousarray(mask[None], dtype=np.float32),
            "spacing": np.asarray(image_obj.header.get_zooms()[:3], dtype=np.float64),
            "affine": image_obj.affine.copy(),
            "grid_header_adjusted": geometry.header_adjusted,
            "max_grid_offset_voxels": geometry.max_grid_offset_voxels}


def extract_patch(array: np.ndarray, start: np.ndarray, size: np.ndarray,
                  pad_value: float = 0) -> np.ndarray:
    """Copy a channel-first patch, with explicit padding outside the volume."""
    spatial = np.asarray(array.shape[1:])
    output = np.full((array.shape[0], *size), pad_value, dtype=array.dtype)
    source_low, source_high = np.maximum(start, 0), np.minimum(start + size, spatial)
    target_low = source_low - start
    target_high = target_low + source_high - source_low
    if np.all(source_high > source_low):
        output[(slice(None), *[slice(a, b) for a, b in zip(target_low, target_high)])] = array[
            (slice(None), *[slice(a, b) for a, b in zip(source_low, source_high)])]
    return output


def spatial_augmentation(config: dict) -> Compose:
    keys = ["image", "label"]
    transforms = [RandAffined(
        keys=keys, prob=config["affine_probability"],
        rotate_range=(float(np.deg2rad(config["rotation_degrees"])),) * 3,
        scale_range=(config["scale_fraction"],) * 3,
        mode=("bilinear", "nearest"), padding_mode="border")]
    for axis in config["flip_axes"]:
        transforms.append(RandFlipd(keys=keys, prob=0.5, spatial_axis=axis))
    if config["elastic_probability"] > 0:
        transforms.append(Rand3DElasticd(
            keys=keys, prob=config["elastic_probability"],
            sigma_range=tuple(config["elastic_sigma_range"]),
            magnitude_range=tuple(config["elastic_magnitude_range"]),
            mode=("bilinear", "nearest"), padding_mode="border"))
    return Compose(transforms)


class PancreasDataset(Dataset):
    def __init__(self, records: list[dict], config: dict, training: bool):
        self.records, self.config, self.training = records, config, training
        aug = config["augmentation"]
        spatial_enabled = bool(aug["affine_probability"] > 0 or aug["elastic_probability"] > 0
                               or aug["flip_axes"])
        self.augmentation = spatial_augmentation(aug) if training and spatial_enabled else None
        self.set_seed(config["seed"])

    def set_seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        if self.augmentation is not None:
            self.augmentation.set_random_state(seed=int(seed) % 2**32)

    def __len__(self) -> int:
        return len(self.records) * (self.config["data"]["patches_per_case"] if self.training else 1)

    def __getitem__(self, index: int) -> dict:
        data_config = self.config["data"]
        record = self.records[index // data_config["patches_per_case"] if self.training else index]
        pair = load_pair(record, data_config)
        result = {"case_id": record["case_id"], "patient_id": record["patient_id"],
                  "spacing": torch.tensor(pair["spacing"]), "affine": torch.tensor(pair["affine"])}
        if not self.training:
            result.update(image=torch.from_numpy(pair["image"]), label=torch.from_numpy(pair["label"]))
            return result
        size = np.asarray(data_config["patch_size"])
        halo = data_config["boundary_halo"]
        enlarged = size + 2 * halo
        foreground = np.flatnonzero(pair["label"][0])
        spatial = np.asarray(pair["image"].shape[1:])
        if foreground.size and self.rng.random() < data_config["foreground_probability"]:
            center = np.asarray(np.unravel_index(self.rng.choice(foreground), spatial))
        else:
            center = np.asarray([self.rng.integers(n) for n in spatial])
        start = center - enlarged // 2
        patch = {key: extract_patch(pair[key], start, enlarged) for key in ("image", "label")}
        if self.augmentation is not None:
            patch = self.augmentation(patch)
        image = np.asarray(patch["image"], dtype=np.float32)
        mask = np.asarray(patch["label"], dtype=np.float32)
        boundary = inner_boundary(mask[0])[None]
        # Derive on the ENLARGED, augmented mask; only then cut to network size.
        crop = (slice(None), *[slice(halo, halo + n) for n in size])
        image, mask, boundary = image[crop].copy(), mask[crop].copy(), boundary[crop].copy()
        aug = self.config["augmentation"]
        if aug["noise_probability"] > 0 and self.rng.random() < aug["noise_probability"]:
            image += self.rng.normal(0, aug["noise_std"], image.shape).astype(np.float32)
        result.update(image=torch.from_numpy(image), label=torch.from_numpy(mask),
                      boundary=torch.from_numpy(boundary))
        return result


def seed_worker(worker_id: int) -> None:
    info = get_worker_info()
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    info.dataset.set_seed(seed)
