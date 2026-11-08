import csv
import hashlib
import json

import pytest

from pancreas_seg.data import (make_or_load_splits, manifest_digest,
                               read_manifest, validate_splits)


def records_for(*labels):
    return [{"case_id": f"case_{index}", "patient_id": f"patient_{index}",
             "image": f"image_{index}", "label": f"label_{index}", "official_split": split}
            for index, split in enumerate(labels)]


def test_manifest_preserves_official_labels_and_legacy_digest(tmp_path):
    path = tmp_path / "manifest.csv"
    columns = ["case_id", "patient_id", "image", "label"]
    row = {"case_id": "case_a", "patient_id": "patient_a", "image": "ct.nii.gz", "label": "mask.nii.gz"}
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerow(row)
    legacy = read_manifest(path)
    expected = dict(row, image=str(tmp_path / "ct.nii.gz"), label=str(tmp_path / "mask.nii.gz"))
    assert legacy == [expected]
    digest = hashlib.sha256(json.dumps([expected], sort_keys=True).encode()).hexdigest()
    assert manifest_digest(legacy) == digest
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns + ["official_split"])
        writer.writeheader()
        writer.writerow(dict(row, official_split=" val "))
    assert read_manifest(path) == [dict(expected, official_split="val")]


def test_official_train_val_test_are_preserved_and_reused(tmp_path):
    records = records_for("test", "train", "val", "train", "test")
    path = tmp_path / "splits.json"
    splits = make_or_load_splits(records, path, seed=42)
    assert splits["train"] == ["case_1", "case_3"]
    assert splits["val"] == ["case_2"]
    assert splits["test"] == ["case_0", "case_4"]
    assert make_or_load_splits(records, path, seed=7) == splits


@pytest.mark.parametrize("labels", [("train", "", "test"), ("train", "validation", "test"),
                                   ("train", None, "test"), ("train", "val")])
def test_incomplete_or_unrecognized_official_labels_fail(tmp_path, labels):
    with pytest.raises(ValueError, match="official_split|Official partitions"):
        make_or_load_splits(records_for(*labels), tmp_path / "splits.json")
    assert not (tmp_path / "splits.json").exists()


def test_patient_cannot_cross_official_partitions(tmp_path):
    records = records_for("train", "val", "test")
    records[1]["patient_id"] = records[0]["patient_id"]
    with pytest.raises(ValueError, match="Patient leakage in official"):
        make_or_load_splits(records, tmp_path / "splits.json")


def test_train_test_only_derives_grouped_validation_without_moving_test(tmp_path):
    records = records_for(*(["train"] * 20), "test", "test")
    for index in range(20):
        records[index]["patient_id"] = f"patient_{index // 2}"
    splits = make_or_load_splits(records, tmp_path / "first.json", fractions=(0.6, 0.3, 0.1), seed=19)
    repeated = make_or_load_splits(records, tmp_path / "second.json", fractions=(0.6, 0.3, 0.1), seed=19)
    assert splits == repeated
    assert splits["test"] == ["case_20", "case_21"]
    # Ten training patients, floor(10 * 0.3 / (0.6 + 0.3)) = three val patients.
    assert len(splits["val"]) == 6
    assert set(splits["train"] + splits["val"]) == {f"case_{index}" for index in range(20)}
    validate_splits(records, splits)


def test_train_test_only_needs_two_training_patients(tmp_path):
    records = records_for("train", "train", "test")
    records[1]["patient_id"] = records[0]["patient_id"]
    with pytest.raises(ValueError, match="two official training patient groups"):
        make_or_load_splits(records, tmp_path / "splits.json")


@pytest.mark.parametrize("swap", [("train", "val"), ("train", "test")])
def test_existing_splits_cannot_override_official_labels(tmp_path, swap):
    records = records_for("train", "val", "test")
    path = tmp_path / "splits.json"
    splits = make_or_load_splits(records, path)
    a, b = swap
    splits[a], splits[b] = splits[b], splits[a]
    path.write_text(json.dumps(splits))
    with pytest.raises(ValueError, match="contradict official"):
        make_or_load_splits(records, path)


def test_official_label_changes_invalidate_frozen_digest(tmp_path):
    records = records_for("train", "train", "val", "test")
    path = tmp_path / "splits.json"
    make_or_load_splits(records, path)
    records[0]["official_split"] = "val"
    with pytest.raises(ValueError, match="Manifest changed"):
        make_or_load_splits(records, path)


def test_all_blank_official_column_keeps_generic_grouping(tmp_path):
    records = records_for(*([""] * 12))
    splits = make_or_load_splits(records, tmp_path / "blank.json", seed=3)
    generic = [{key: value for key, value in record.items() if key != "official_split"} for record in records]
    baseline = make_or_load_splits(generic, tmp_path / "generic.json", seed=3)
    assert {name: splits[name] for name in ("train", "val", "test")} == {
        name: baseline[name] for name in ("train", "val", "test")}
