import argparse

from pancreas_seg.engine import evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate a frozen checkpoint on full volumes")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.split, args.device, args.save_predictions)


if __name__ == "__main__":
    main()

