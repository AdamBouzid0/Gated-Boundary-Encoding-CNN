import copy
import csv
import json

import nibabel as nib
import numpy as np
import pytest

from pancreas_seg.benchmark import benchmark_training
from pancreas_seg.config import ROOT, load_config
from pancreas_seg.data import make_or_load_splits, read_manifest


def synthetic_config(tmp_path):
    rng = np.random.default_rng(51)
    rows = []
    for index, partition in enumerate(("train", "train", "train", "train", "val", "test")):
        mask = np.zeros((36, 36, 36), dtype=np.uint8)
        mask[10:24, 11:25, 9:23] = 1
        image = rng.normal(0, 0.1, mask.shape).astype(np.float32) + mask * (index + 1)
        paths = {"image": tmp_path / f"case_{index}_ct.nii.gz", "label": tmp_path / f"case_{index}_mask.nii.gz"}
        for key, array in (("image", image), ("label", mask)):
            volume = nib.Nifti1Image(array, np.diag([1.5, 1.5, 1.5, 1.0]))
            volume.header.set_xyzt_units("mm")
            nib.save(volume, paths[key])
        rows.append({"case_id": f"case_{index}", "patient_id": f"patient_{index}",
                     **{key: str(value) for key, value in paths.items()}, "official_split": partition})
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = load_config(ROOT / "configs/boundary_attention.yaml")
    config["data"].update(manifest=str(manifest), splits=str(tmp_path / "splits.json"), expected_cases=6,
                           pancreas_label_ids=[1], patch_size=[32, 32, 32], boundary_halo=1,
                           patches_per_case=1, target_spacing_mm=None, foreground_probability=1.0)
    config["model"].update(base_channels=2, boundary_base_channels=1)
    config["training"].update(num_workers=0, cpu_threads=1, batch_size=1, accumulation_steps=2, amp=False)
    config["augmentation"].update(affine_probability=0, elastic_probability=0, noise_probability=0, flip_axes=[])
    records = read_manifest(manifest)
    make_or_load_splits(records, config["data"]["splits"], seed=config["seed"])
    return config, {row["case_id"] for row in rows if row["official_split"] == "train"}


def test_two_batches_use_real_input_pipeline_same_patches_and_training_cases_only(tmp_path, monkeypatch):
    from pancreas_seg import data

    config, training_ids = synthetic_config(tmp_path)
    original_config = copy.deepcopy(config)
    loads = []
    actual_load_pair = data.load_pair

    def checked_load_pair(record, data_config):
        assert record["case_id"] in training_ids
        loads.append(record["case_id"])
        return actual_load_pair(record, data_config)

    monkeypatch.setattr(data, "load_pair", checked_load_pair)
    output = tmp_path / "benchmark"
    report = benchmark_training(config, batches=2, device_name="cpu", output_dir=output)
    assert report["status"] == "completed"
    assert report["full_train_cases"] == report["full_train_batches"] == 4
    assert len(report["sampled_case_ids"]) == 2
    assert set(report["sampled_case_ids"]) <= training_ids
    assert loads == report["sampled_case_ids"] * 2
    assert report["patches_identical_across_variants"] is True
    for result in report["variants"].values():
        assert result["batches_completed"] == 2
        assert result["optimizer_steps"] == result["optimizer_steps_attempted"] == 1
        assert result["optimizer_steps_skipped"] == 0
        assert result["finite_losses"] is True
        assert len(result["batch_rows"]) == 2
        assert all(np.isfinite(row["loss"]) for row in result["batch_rows"])
        assert result["projected_train_epoch_seconds"] == pytest.approx(result["mean_batch_seconds"] * 4)
        assert result["peak_cuda_allocated_gib"] is None
        assert result["total_wall_seconds"] > result["compute_seconds"] > 0
    assert config == original_config
    assert json.loads((output / "report.json").read_text()) == report
    with (output / "batches.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 4
    assert not list(output.rglob("*.pt"))
    with pytest.raises(FileExistsError, match="fresh output_dir"):
        benchmark_training(config, batches=2, device_name="cpu", output_dir=output)
    assert len(loads) == 4


def test_benchmark_refuses_unbounded_request_and_repeated_patches(tmp_path):
    config, _ = synthetic_config(tmp_path)
    with pytest.raises(ValueError, match="full training epoch has only 4"):
        benchmark_training(config, batches=5, device_name="cpu", output_dir=tmp_path / "too_many")
    config["data"]["patches_per_case"] = 2
    with pytest.raises(ValueError, match="patches_per_case=1"):
        benchmark_training(config, batches=2, device_name="cpu", output_dir=tmp_path / "repeated")
    assert not (tmp_path / "too_many").exists()
    assert not (tmp_path / "repeated").exists()
