"""Exercise the full pipeline on synthetic NIfTI volumes, on CPU."""

import argparse
import copy
import json
import tempfile
from pathlib import Path

import torch

from make_demo_data import make_demo
from pancreas_seg.config import ROOT, load_config
from pancreas_seg.engine import evaluate_checkpoint, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Optional persistent JSON verification report")
    args = parser.parse_args()
    (ROOT / "runs").mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="smoke_", dir=ROOT / "runs"))
    manifest = make_demo(work / "synthetic", cases=6)
    config = load_config(ROOT / "configs/boundary_attention.yaml")
    config["data"].update(manifest=str(manifest), splits=str(work / "splits.json"), expected_cases=6,
                          pancreas_label_ids=[1], patch_size=[32, 32, 32], boundary_halo=2)
    config["model"].update(base_channels=2, boundary_base_channels=1)
    config["training"].update(output_dir=str(work / "runs"), epochs=1, num_workers=0,
                              cpu_threads=1, validate_every=1, amp=False)
    config["augmentation"].update(affine_probability=0, elastic_probability=0,
                                  noise_probability=0, flip_axes=[])
    report = {"purpose": "Synthetic pipeline verification; NOT pancreas benchmark results",
              "torch": torch.__version__, "models": {}}
    for variant in ("attention_unet", "boundary_attention_unet"):
        current = copy.deepcopy(config)
        current["model"]["variant"] = variant
        run = train(current, device_name="cpu")
        if variant == "boundary_attention_unet":
            # Check optimizer/scaler/scheduler reload and one additional epoch.
            current["training"]["epochs"] = 2
            train(current, device_name="cpu", resume=str(run / "last.pt"))
        summary = evaluate_checkpoint(run / "best.pt", device_name="cpu", save_predictions=True)
        assert 0 <= summary["dice"] <= 1
        report["models"][variant] = {
            "status": "passed", "parameters_at_smoke_width": summary["parameters"],
            "train_completed": True, "checkpoint_reloaded": True,
            "full_volume_evaluated": True, "nifti_prediction_saved": True,
            "resumed_training_checked": variant == "boundary_attention_unet",
        }
    assert (work / "splits.json").is_file()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"PASS: both variants trained, checkpointed and evaluated. Artifacts: {work}")


if __name__ == "__main__":
    main()

