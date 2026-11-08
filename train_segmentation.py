"""Configuration-driven entry point, analogous to the 2018 training script."""

import argparse

from pancreas_seg.config import load_config
from pancreas_seg.engine import train


def main():
    parser = argparse.ArgumentParser(description="Train 3D pancreas segmentation")
    parser.add_argument("-c", "--config", default="configs/boundary_attention.yaml")
    parser.add_argument("--variant", choices=["unet", "attention_unet", "boundary_attention_unet"])
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--resume", help="Your last.pt checkpoint")
    parser.add_argument("--output-dir", help="Override training.output_dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--label-id", type=int, nargs="+", help="Pancreas class ID(s) in the source masks")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.variant:
        config["model"]["variant"] = args.variant
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.label_id:
        config["data"]["pancreas_label_ids"] = args.label_id
    run_dir = train(config, args.device, args.resume)
    print(f"Completed. Best checkpoint: {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

