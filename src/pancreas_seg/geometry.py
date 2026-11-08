"""NumPy-only comparison of image and mask voxel grids."""

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class GridComparison:
    compatible: bool
    status: str = ""
    reason: str = ""
    max_grid_offset_voxels: float | None = None
    header_adjusted: bool = False


def compare_grids(image_shape, image_affine, mask_shape, mask_affine,
                  tolerance_voxels: float = 0.0, affine_atol: float = 1e-4) -> GridComparison:
    """Bound header disagreement at all eight voxel-centre corners.

    Offset is the largest absolute coordinate component of
    (inv(image_affine) @ mask_affine - I) @ [i, j, k, 1], in CT voxels.
    A positive tolerance below half a voxel permits index correspondence
    despite small header rounding/shear differences. It is an explicit
    dataset assumption, not a substitute for inspecting image/mask pairs.
    """
    if not np.isfinite(tolerance_voxels) or not 0 <= tolerance_voxels < 0.5:
        raise ValueError("grid_tolerance_voxels must be in [0, 0.5)")
    if len(image_shape) != 3 or len(mask_shape) != 3:
        return GridComparison(False, "not_3d", "Expected scalar 3D CT and mask")
    if tuple(image_shape) != tuple(mask_shape):
        return GridComparison(False, "shape_mismatch", "CT and mask have different array shapes")
    image_affine, mask_affine = np.asarray(image_affine), np.asarray(mask_affine)
    for affine in (image_affine, mask_affine):
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            return GridComparison(False, "invalid_geometry", "Invalid or non-finite NIfTI affine")
        if abs(np.linalg.det(affine[:3, :3])) < 1e-12:
            return GridComparison(False, "invalid_geometry", "Singular spatial affine")
        if not np.allclose(affine[3], [0, 0, 0, 1], rtol=0, atol=1e-12):
            return GridComparison(False, "invalid_geometry", "Invalid affine homogeneous row")
    try:
        difference = np.linalg.inv(image_affine) @ mask_affine - np.eye(4)
    except np.linalg.LinAlgError:
        return GridComparison(False, "invalid_geometry", "Singular spatial affine")
    corners = np.asarray([(*corner, 1) for corner in product(*[(0, n - 1) for n in image_shape])]).T
    offset = float(np.max(np.abs((difference @ corners)[:3])))
    if not np.isfinite(offset):
        return GridComparison(False, "invalid_geometry", "Non-finite voxel-grid displacement")
    exact = bool(np.allclose(image_affine, mask_affine, rtol=0, atol=affine_atol))
    compatible = exact or (tolerance_voxels > 0 and offset <= tolerance_voxels)
    return GridComparison(
        compatible, "" if compatible else "affine_mismatch",
        "" if compatible else "CT and mask do not occupy the same voxel grid within the configured tolerance",
        offset, header_adjusted=compatible and not np.array_equal(image_affine, mask_affine))
