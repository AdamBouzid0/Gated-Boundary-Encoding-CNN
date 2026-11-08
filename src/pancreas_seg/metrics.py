"""Per-volume segmentation scores and physical-distance boundary metrics."""

import numpy as np
from scipy.ndimage import distance_transform_edt

from .boundaries import inner_boundary


def segmentation_metrics(prediction: np.ndarray, target: np.ndarray,
                         spacing=(1.0, 1.0, 1.0), tolerance_mm: float = 2.0) -> dict:
    prediction, target = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Prediction and target must have the same 3D shape")
    tp = int(np.count_nonzero(prediction & target))
    pred_n, target_n = int(prediction.sum()), int(target.sum())
    both_empty = pred_n == target_n == 0
    dice = 1.0 if both_empty else 2 * tp / max(pred_n + target_n, 1)
    iou = 1.0 if both_empty else tp / max(pred_n + target_n - tp, 1)
    precision = tp / pred_n if pred_n else float(target_n == 0)
    recall = tp / target_n if target_n else 1.0
    if both_empty:
        boundary_f1, hd95 = 1.0, 0.0
    elif pred_n == 0 or target_n == 0:
        boundary_f1, hd95 = 0.0, None  # distance is undefined for a missing surface
    else:
        pred_surface, target_surface = inner_boundary(prediction).astype(bool), inner_boundary(target).astype(bool)
        to_target = distance_transform_edt(~target_surface, sampling=spacing)[pred_surface]
        to_prediction = distance_transform_edt(~pred_surface, sampling=spacing)[target_surface]
        bp = float(np.mean(to_target <= tolerance_mm))
        br = float(np.mean(to_prediction <= tolerance_mm))
        boundary_f1 = 2 * bp * br / (bp + br) if bp + br else 0.0
        hd95 = float(max(np.percentile(to_target, 95), np.percentile(to_prediction, 95)))
    return {"dice": float(dice), "iou": float(iou), "precision": float(precision),
            "recall": float(recall), "boundary_f1": float(boundary_f1), "hd95_mm": hd95}


def aggregate_metrics(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot aggregate an empty evaluation split")
    result = {"cases": len(rows)}
    for name in ("dice", "iou", "precision", "recall", "boundary_f1", "hd95_mm"):
        values = [row[name] for row in rows if row[name] is not None]
        result[name] = float(np.mean(values)) if values else None
        result[f"{name}_std"] = float(np.std(values)) if values else None
    result["hd95_undefined_cases"] = sum(row["hd95_mm"] is None for row in rows)
    return result
