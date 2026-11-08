"""Run a bounded training benchmark without touching production checkpoints."""

import argparse
import json

from pancreas_seg.benchmark import benchmark_training
from pancreas_seg.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/boundary_attention.yaml")
    parser.add_argument("--variants", nargs="+", choices=["unet", "attention_unet", "boundary_attention_unet"],
                        default=["attention_unet", "boundary_attention_unet"])
    parser.add_argument("--batches", type=int, default=20, help="Loader batches per variant, not optimizer steps")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="runs/runtime_smoke", help="Fresh directory; existing reports are preserved")
    args = parser.parse_args()
    report = benchmark_training(load_config(args.config), args.variants, args.batches, args.device, args.output_dir)
    print(json.dumps({"status": report["status"], "output_dir": args.output_dir,
                      "variants": {name: {key: result.get(key) for key in (
                          "status", "batches_completed", "optimizer_steps", "mean_batch_seconds",
                          "peak_cuda_allocated_gib", "projected_train_epoch_seconds", "error")}
                                   for name, result in report["variants"].items()}}, indent=2))
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
