import copy

import numpy as np
import pytest
import torch

from pancreas_seg.boundaries import inner_boundary
from pancreas_seg.config import ROOT, load_config
from pancreas_seg.data import (PancreasDataset, make_or_load_splits, read_manifest,
                               spatial_augmentation, validate_splits)
from pancreas_seg.losses import JointLoss
from pancreas_seg.metrics import segmentation_metrics
from pancreas_seg.models import PancreasUNet3d


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def test_boundary_is_surface_and_rejects_channel_axis():
    mask = np.zeros((9, 9, 9), dtype=bool)
    mask[2:7, 2:7, 2:7] = True
    boundary = inner_boundary(mask)
    assert boundary.sum() == 5**3 - 3**3
    assert boundary[2, 4, 4] == 1 and boundary[4, 4, 4] == 0
    with pytest.raises(ValueError):
        inner_boundary(mask[None])


@pytest.mark.parametrize("variant", ["unet", "attention_unet", "boundary_attention_unet"])
def test_shapes_attention_and_backward(variant):
    torch.manual_seed(3)
    model = PancreasUNet3d(variant=variant, base_channels=2, boundary_base_channels=1)
    x = torch.randn(1, 1, 32, 32, 32)
    output = model(x, return_attention=True)
    assert output["seg_logits"].shape == (1, 2, 32, 32, 32)
    for alpha in output["attention_maps"]:
        assert torch.all((alpha >= 0) & (alpha <= 1))
    config = load_config(ROOT / "configs/boundary_attention.yaml")
    label = torch.zeros(1, 1, 32, 32, 32)
    label[:, :, 12:20, 12:20, 12:20] = 1
    boundary = torch.from_numpy(inner_boundary(label[0, 0].numpy()))[None, None]
    loss, _ = JointLoss(config["loss"])(output, label, boundary)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.encoder.parameters())
    if variant == "boundary_attention_unet":
        assert output["boundary_logits"].shape == label.shape
        assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.boundary_encoder.parameters())
    else:
        assert output["boundary_logits"] is None


def test_boundary_loss_reaches_boundary_encoder_before_fusion_only():
    model = PancreasUNet3d(base_channels=2, boundary_base_channels=1)
    output = model(torch.randn(1, 1, 32, 32, 32))
    torch.nn.functional.binary_cross_entropy_with_logits(
        output["boundary_logits"], torch.zeros_like(output["boundary_logits"])).backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    deepest = model.boundary_encoder.blocks[-1].conv1[0].weight.grad
    assert deepest is not None and torch.count_nonzero(deepest) > 0


def test_segmentation_loss_uses_boundary_encoder_at_inference():
    model = PancreasUNet3d(base_channels=2, boundary_base_channels=1)
    output = model(torch.randn(1, 1, 32, 32, 32), compute_boundary=False)
    output["seg_logits"].square().mean().backward()
    assert output["boundary_logits"] is None
    deepest = model.boundary_encoder.blocks[-1].conv1[0].weight.grad
    assert deepest is not None and torch.count_nonzero(deepest) > 0


def test_patient_splits_are_reused_and_leakage_is_rejected(tmp_path):
    records = [{"case_id": f"case_{i}", "patient_id": f"patient_{i // 2}",
                "image": f"image_{i}", "label": f"label_{i}"} for i in range(20)]
    path = tmp_path / "splits.json"
    splits = make_or_load_splits(records, path, seed=42)
    assert make_or_load_splits(records, path, seed=123) == splits
    corrupted = copy.deepcopy(splits)
    moved = corrupted["train"].pop()
    corrupted["test"].append(moved)
    with pytest.raises(ValueError, match="Patient leakage"):
        validate_splits(records, corrupted)


def test_metrics_are_physical_and_empty_cases_explicit():
    a = np.zeros((12, 12, 12), dtype=bool)
    a[3:8, 3:8, 3:8] = True
    assert segmentation_metrics(a, a)["dice"] == 1
    shifted = np.roll(a, 1, axis=2)
    metrics = segmentation_metrics(shifted, a, spacing=(1, 1, 3), tolerance_mm=2)
    assert metrics["hd95_mm"] == 3
    assert metrics["boundary_f1"] < 1
    assert segmentation_metrics(np.zeros_like(a), a)["hd95_mm"] is None
    assert segmentation_metrics(np.zeros_like(a), np.zeros_like(a))["dice"] == 1


def test_paired_elastic_warp_preserves_alignment_and_labels():
    config = load_config(ROOT / "configs/boundary_attention.yaml")["augmentation"]
    config.update(affine_probability=0, elastic_probability=1,
                  elastic_sigma_range=[2, 3], elastic_magnitude_range=[10, 20])
    transform = spatial_augmentation(config)
    transform.set_random_state(seed=19)
    volume = np.zeros((1, 24, 24, 24), dtype=np.float32)
    volume[:, 6:18, 6:18, 6:18] = 1
    # For this alignment check, use nearest for both image and mask.
    transform.transforms[-1].mode = ("nearest", "nearest")
    output = transform({"image": volume.copy(), "label": volume.copy()})
    assert np.array_equal(output["image"], output["label"])
    assert set(np.unique(output["label"])) <= {0, 1}
    assert not np.array_equal(output["label"], volume)


def test_invalid_image_grid_fails_early(tmp_path):
    import nibabel as nib
    from pancreas_seg.data import load_pair
    image, label = tmp_path / "image.nii.gz", tmp_path / "label.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8)), np.eye(4)), image)
    affine = np.eye(4)
    affine[0, 3] = 5
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8)), affine), label)
    config = load_config(ROOT / "configs/boundary_attention.yaml")["data"]
    config["pancreas_label_ids"] = [1]
    with pytest.raises(ValueError, match="not aligned"):
        load_pair({"image": str(image), "label": str(label), "case_id": "bad"}, config)


def test_dataset_halo_avoids_fake_crop_boundaries(tmp_path):
    import nibabel as nib
    path = tmp_path / "volume.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((64, 64, 64), dtype=np.float32), np.eye(4)), path)
    config = load_config(ROOT / "configs/boundary_attention.yaml")
    config["data"].update(pancreas_label_ids=[1], patch_size=[32, 32, 32], boundary_halo=2)
    config["augmentation"].update(affine_probability=0, noise_probability=0, elastic_probability=0)
    record = {"case_id": "solid", "patient_id": "solid", "image": str(path), "label": str(path)}
    dataset = PancreasDataset([record], config, training=True)
    # Fix the enlarged patch centre inside the solid organ.
    class CenterRng:
        def random(self): return 0.0
        def choice(self, values): return np.ravel_multi_index((32, 32, 32), (64, 64, 64))
    dataset.rng = CenterRng()
    sample = dataset[0]
    assert sample["label"].shape == (1, 32, 32, 32)
    assert torch.all(sample["label"] == 1)
    assert sample["boundary"].sum() == 0


def test_nifti_spatial_units_are_converted_to_millimetres(tmp_path):
    import nibabel as nib
    from pancreas_seg.data import load_pair
    image_path, label_path = tmp_path / "image.nii.gz", tmp_path / "label.nii.gz"
    image = nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.float32), np.diag([0.002, 0.002, 0.003, 1]))
    label = nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.uint8), image.affine)
    image.header.set_xyzt_units("meter")
    label.header.set_xyzt_units("meter")
    nib.save(image, image_path)
    nib.save(label, label_path)
    config = load_config(ROOT / "configs/boundary_attention.yaml")["data"]
    config["pancreas_label_ids"] = [1]
    pair = load_pair({"image": str(image_path), "label": str(label_path), "case_id": "units"}, config)
    np.testing.assert_allclose(pair["spacing"], [2, 2, 3], atol=1e-5)
    assert np.all(pair["label"] == 1)


def test_unknown_units_can_be_rejected(tmp_path):
    import nibabel as nib
    from pancreas_seg.data import load_pair
    path = tmp_path / "unknown.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((8, 8, 8)), np.eye(4)), path)
    config = load_config(ROOT / "configs/boundary_attention.yaml")["data"]
    config.update(pancreas_label_ids=[1], spacing_unit_if_unknown=None)
    with pytest.raises(ValueError, match="Unknown NIfTI spatial units"):
        load_pair({"image": str(path), "label": str(path), "case_id": "unknown"}, config)
