"""Boundary TARGET generation, outside the differentiable model."""

import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure


def inner_boundary(mask: np.ndarray) -> np.ndarray:
    """Return a six-connected inner surface for one 3D binary organ mask.

    No batch/channel axes are accepted. The outside of this array is treated
    as background; derive the target before the final patch crop to avoid
    inventing surfaces at patch cuts. Width is in voxels, not millimetres.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask without channel/batch axes, got {mask.shape}")
    eroded = binary_erosion(mask, structure=generate_binary_structure(3, 1), border_value=0)
    return (mask & ~eroded).astype(np.float32)

