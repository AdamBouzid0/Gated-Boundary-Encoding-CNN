import hashlib
import importlib.util

import nibabel as nib
import numpy as np
import pytest

from pancreas_seg.config import ROOT, load_config, validate_config
from pancreas_seg.data import load_pair
from pancreas_seg.geometry import compare_grids


spec = importlib.util.spec_from_file_location("pancreas_filter", ROOT / "filter_totalsegmentator_pancreas.py")
filtering = importlib.util.module_from_spec(spec)
spec.loader.exec_module(filtering)


def make_pair(tmp_path, image_affine=None, mask_affine=None, mask_shape=(12, 14, 16)):
    case = tmp_path / "s0001"
    (case / "segmentations").mkdir(parents=True)
    image_affine = np.diag([2.0, 2.0, 2.0, 1.0]) if image_affine is None else image_affine
    mask_affine = image_affine.copy() if mask_affine is None else mask_affine
    image = nib.Nifti1Image(np.arange(12 * 14 * 16, dtype=np.float32).reshape(12, 14, 16), image_affine)
    image.header.set_xyzt_units("mm")
    image.set_sform(image_affine, code=1)
    # An uncoded qform must never replace the CT's authoritative sform.
    uncoded_qform = np.eye(4)
    uncoded_qform[:3, 3] = 100
    image.set_qform(uncoded_qform, code=0)
    mask = np.zeros(mask_shape, dtype=np.uint8)
    mask[3:7, 4:8, 5:9] = 1
    label = nib.Nifti1Image(mask, mask_affine)
    label.header.set_xyzt_units("mm")
    nib.save(image, case / "ct.nii.gz")
    nib.save(label, case / "segmentations/pancreas.nii.gz")
    record = {"case_id": case.name, "patient_id": case.name,
              "image": str(case / "ct.nii.gz"), "label": str(case / "segmentations/pancreas.nii.gz")}
    config = load_config(ROOT / "configs/boundary_attention.yaml")["data"]
    config.update(pancreas_label_ids=[1], target_spacing_mm=None, intensity_clip=None)
    return case, record, config, mask


def test_offset_is_maximum_corner_coordinate_in_ct_voxels():
    ct = np.diag([2.0, 3.0, 4.0, 1.0])
    mask = ct.copy()
    mask[:3, 3] = [0.4, -0.3, 0.4]
    comparison = compare_grids((9, 11, 13), ct, (9, 11, 13), mask, tolerance_voxels=0.25)
    assert comparison.compatible and comparison.header_adjusted
    assert comparison.max_grid_offset_voxels == pytest.approx(0.2)
    # A shear accumulates over the grid: checking only the origin is insufficient.
    mask = ct.copy()
    mask[0, 1] = 0.06
    comparison = compare_grids((9, 11, 13), ct, (9, 11, 13), mask, tolerance_voxels=0.25)
    assert not comparison.compatible
    assert comparison.max_grid_offset_voxels == pytest.approx(0.3)


def test_mild_shear_uses_ct_affine_in_memory_and_keeps_labels_and_sources(tmp_path, monkeypatch):
    ct = np.diag([2.0, 2.0, 2.0, 1.0])
    ct[0, 1] = 0.01
    mask_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    case, record, config, mask = make_pair(tmp_path, ct, mask_affine)
    original_bytes = {key: hashlib.sha256(open(record[key], "rb").read()).hexdigest() for key in ("image", "label")}
    assert filtering.audit_case(case)["status"] == "affine_mismatch"
    config.pop("grid_tolerance_voxels", None)
    with pytest.raises(ValueError, match="not aligned"):
        load_pair(record, config)
    config["grid_tolerance_voxels"] = 0.25
    row = filtering.audit_case(case, grid_tolerance_voxels=0.25)
    assert row["status"] == "kept" and row["grid_header_adjusted"] is True
    assert row["max_grid_offset_voxels"] < 0.25
    canonical_inputs = []
    original_canonical = nib.as_closest_canonical

    def capture_canonical(image):
        canonical_inputs.append(image.affine.copy())
        return original_canonical(image)

    monkeypatch.setattr(nib, "as_closest_canonical", capture_canonical)
    pair = load_pair(record, config)
    np.testing.assert_array_equal(pair["label"][0], mask)
    np.testing.assert_array_equal(canonical_inputs[0], canonical_inputs[1])
    np.testing.assert_allclose(pair["affine"], ct, atol=1e-8)
    assert pair["grid_header_adjusted"] is True
    assert pair["max_grid_offset_voxels"] == row["max_grid_offset_voxels"]
    assert nib.load(record["image"]).header["qform_code"] == 0
    np.testing.assert_array_equal(nib.load(record["label"]).affine, mask_affine)
    assert {key: hashlib.sha256(open(record[key], "rb").read()).hexdigest() for key in original_bytes} == original_bytes
    summary = filtering.write_outputs([row], tmp_path / "reports", root=tmp_path, metadata_path=None,
                                       min_voxels=1, foreground_value=1, grid_tolerance_voxels=0.25)
    assert summary["grid_tolerance_voxels"] == 0.25
    assert summary["grid_header_adjusted_cases"] == 1


@pytest.mark.parametrize("problem", ["shift", "shape"])
def test_filter_and_loader_reject_real_misalignment(tmp_path, problem):
    ct = np.diag([2.0, 2.0, 2.0, 1.0])
    mask = ct.copy()
    if problem == "shift":
        mask[0, 3] = 2  # one complete CT voxel
    case, record, config, _ = make_pair(tmp_path, ct, mask,
                                       mask_shape=(10, 14, 16) if problem == "shape" else (12, 14, 16))
    config["grid_tolerance_voxels"] = 0.25
    assert filtering.audit_case(case, grid_tolerance_voxels=0.25)["status"] == {
        "shift": "affine_mismatch", "shape": "shape_mismatch"}[problem]
    with pytest.raises(ValueError, match="not aligned"):
        load_pair(record, config)


def test_exact_grids_keep_legacy_behavior(tmp_path):
    case, record, config, mask = make_pair(tmp_path)
    config.pop("grid_tolerance_voxels", None)
    row = filtering.audit_case(case)
    pair = load_pair(record, config)
    assert row["status"] == "kept"
    assert row["grid_header_adjusted"] is False
    assert pair["grid_header_adjusted"] is False
    np.testing.assert_array_equal(pair["label"][0], mask)


def test_almost_identical_rotation_headers_cannot_separate_paired_voxels(tmp_path):
    mask = np.zeros((32, 32, 32), dtype=np.uint8)
    mask[3:8, 10:19, 20:25] = 1

    def rotation(degrees):
        angle = np.deg2rad(degrees)
        affine = np.eye(4)
        affine[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        return affine

    image_path, label_path = tmp_path / "image.nii.gz", tmp_path / "label.nii.gz"
    nib.save(nib.Nifti1Image(mask.astype(np.float32), rotation(44.99999)), image_path)
    nib.save(nib.Nifti1Image(mask, rotation(45.00001)), label_path)
    image, label = nib.load(image_path), nib.load(label_path)
    assert np.allclose(image.affine, label.affine, rtol=0, atol=1e-4)
    assert nib.aff2axcodes(image.affine) != nib.aff2axcodes(label.affine)
    # Independent canonicalization would swap/flip different voxel axes.
    assert not np.array_equal(np.asanyarray(nib.as_closest_canonical(image).dataobj),
                              np.asanyarray(nib.as_closest_canonical(label).dataobj))
    config = load_config(ROOT / "configs/boundary_attention.yaml")["data"]
    config.update(pancreas_label_ids=[1], target_spacing_mm=None, intensity_clip=None, grid_tolerance_voxels=0)
    pair = load_pair({"case_id": "tie", "image": str(image_path), "label": str(label_path)}, config)
    assert pair["grid_header_adjusted"] is True
    assert pair["max_grid_offset_voxels"] < 0.0001
    # The image was generated from the very same foreground pattern. Their
    # support must coincide after both follow the CT's canonical orientation.
    np.testing.assert_array_equal(pair["image"] > 0, pair["label"] > 0)


@pytest.mark.parametrize("which", ["image", "mask"])
@pytest.mark.parametrize("problem", ["singular", "nan", "infinity"])
def test_invalid_affines_are_rejected_even_when_equal(which, problem):
    image, mask = np.eye(4), np.eye(4)
    bad = image if which == "image" else mask
    bad[0, 0] = {"singular": 0, "nan": np.nan, "infinity": np.inf}[problem]
    comparison = compare_grids((8, 8, 8), image, (8, 8, 8), mask, tolerance_voxels=0.25)
    assert not comparison.compatible and comparison.status == "invalid_geometry"
    comparison = compare_grids((8, 8, 8), bad, (8, 8, 8), bad, tolerance_voxels=0.25)
    assert not comparison.compatible and comparison.status == "invalid_geometry"


def test_tolerance_boundary_and_exact_header_threshold():
    shifted = np.eye(4)
    shifted[0, 3] = 0.25
    assert compare_grids((8, 8, 8), np.eye(4), (8, 8, 8), shifted, tolerance_voxels=0.25).compatible
    shifted[0, 3] += 0.00001
    assert not compare_grids((8, 8, 8), np.eye(4), (8, 8, 8), shifted, tolerance_voxels=0.25).compatible
    shifted[0, 3] = 0.00009
    exact = compare_grids((8, 8, 8), np.eye(4), (8, 8, 8), shifted)
    assert exact.compatible and exact.header_adjusted


@pytest.mark.parametrize("tolerance", [-0.1, 0.5, 1.0, float("nan"), float("inf")])
def test_tolerance_must_be_finite_and_below_half_voxel(tolerance):
    with pytest.raises(ValueError, match="grid_tolerance_voxels"):
        compare_grids((8, 8, 8), np.eye(4), (8, 8, 8), np.eye(4), tolerance_voxels=tolerance)
    config = load_config(ROOT / "configs/boundary_attention.yaml")
    config["data"]["grid_tolerance_voxels"] = tolerance
    with pytest.raises(ValueError, match="grid_tolerance_voxels"):
        validate_config(config)
