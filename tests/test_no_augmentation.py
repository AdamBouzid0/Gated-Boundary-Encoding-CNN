import numpy as np
import nibabel as nib

from pancreas_seg.boundaries import inner_boundary
from pancreas_seg.config import ROOT, load_config
from pancreas_seg.data import PancreasDataset, load_pair


def test_disabled_augmentation_preserves_voxels_and_boundary_context(tmp_path):
    image = np.indices((64, 64, 64), dtype=np.float32).sum(axis=0)
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[8:56, 20:45, 25:55] = 1
    image_path, mask_path = tmp_path / "ct.nii.gz", tmp_path / "pancreas.nii.gz"
    nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
    config = load_config(ROOT / "configs/boundary_attention.yaml")
    config["data"].update(patch_size=[32, 32, 32], boundary_halo=1, foreground_probability=1.0)
    config["augmentation"].update(affine_probability=0, elastic_probability=0,
                                  flip_axes=[], noise_probability=0)
    record = {"case_id": "example", "patient_id": "example",
              "image": str(image_path), "label": str(mask_path)}
    dataset = PancreasDataset([record], config, training=True)

    class FixedCenter:
        def random(self):
            return 0.0

        def choice(self, values):
            return np.ravel_multi_index((32, 32, 32), image.shape)

    dataset.rng = FixedCenter()
    sample = dataset[0]
    pair = load_pair(record, config["data"])
    crop = (slice(16, 48),) * 3
    assert dataset.augmentation is None
    np.testing.assert_array_equal(sample["image"][0].numpy(), pair["image"][0][crop])
    np.testing.assert_array_equal(sample["label"][0].numpy(), mask[crop])
    np.testing.assert_array_equal(sample["boundary"][0].numpy(), inner_boundary(mask)[crop])
