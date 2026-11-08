"""Freeze or validate patient-grouped splits without loading scans or training."""

import argparse

from pancreas_seg.config import load_config
from pancreas_seg.data import make_or_load_splits, official_partitions, read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/boundary_attention.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    records = read_manifest(config["data"]["manifest"])
    official = official_partitions(records)
    splits = make_or_load_splits(records, config["data"]["splits"],
                                 config["data"]["split_fractions"], config["seed"])
    if official is None:
        source = "reproducible patient-grouped random partitions (no official labels)"
    elif official["val"]:
        source = "official train/val/test partitions"
    else:
        source = "official test partition, with validation derived from training patients"
    print(f"Validated {source}")
    print({name: len(splits[name]) for name in ("train", "val", "test")})
    print(f"Frozen splits: {config['data']['splits']}")


if __name__ == "__main__":
    main()
