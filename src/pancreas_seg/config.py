from __future__ import annotations

from pathlib import Path
from math import isfinite

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    """Resolve project-relative paths against the project, not the shell cwd."""
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    for section, key in (("data", "manifest"), ("data", "splits"), ("training", "output_dir")):
        value = Path(config[section][key]).expanduser()
        config[section][key] = str(value if value.is_absolute() else ROOT / value)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    tolerance = config["data"].get("grid_tolerance_voxels", 0.0)
    if not isinstance(tolerance, (int, float)) or not isfinite(tolerance) or not 0 <= tolerance < 0.5:
        raise ValueError("data.grid_tolerance_voxels must be in [0, 0.5)")
    patch = config["data"]["patch_size"]
    if len(patch) != 3 or any(not isinstance(x, int) or x < 32 or x % 16 for x in patch):
        raise ValueError("data.patch_size must have three multiples of 16, each >=32")
    if config["data"]["boundary_halo"] < 1:
        raise ValueError("Keep at least one halo voxel for boundary target generation")
    fractions = config["data"]["split_fractions"]
    if len(fractions) != 3 or any(x <= 0 for x in fractions) or abs(sum(fractions) - 1) > 1e-6:
        raise ValueError("split_fractions must be three positive fractions summing to one")
    if not 0 <= config["data"]["foreground_probability"] <= 1:
        raise ValueError("foreground_probability must be in [0,1]")
    spacing = config["data"]["target_spacing_mm"]
    if spacing is not None and (len(spacing) != 3 or any(x <= 0 for x in spacing)):
        raise ValueError("target_spacing_mm must be null or three positive values")
    if config["training"]["accumulation_steps"] < 1 or config["training"]["epochs"] < 1:
        raise ValueError("epochs and accumulation_steps must be positive")
    if config["training"]["validate_every"] < 1 or config["training"]["batch_size"] < 1:
        raise ValueError("validate_every and batch_size must be positive")
    if config["data"]["patches_per_case"] < 1 or config["training"]["num_workers"] < 0:
        raise ValueError("patches_per_case must be positive and num_workers nonnegative")
    if len(config["model"]["attention_subsample"]) != 3 or any(
            x not in (1, 2) for x in config["model"]["attention_subsample"]):
        raise ValueError("attention_subsample must contain three factors, each 1 or 2")
    if min(config["model"]["base_channels"], config["model"]["boundary_base_channels"]) < 1:
        raise ValueError("Encoder widths must be positive")
    if not 0 <= config["evaluation"]["overlap"] < 1 or config["evaluation"]["boundary_tolerance_mm"] < 0:
        raise ValueError("Invalid sliding-window overlap or boundary tolerance")
    if config["loss"]["dice_weight"] + config["loss"]["ce_weight"] <= 0:
        raise ValueError("At least one segmentation objective must have positive weight")
    if config["loss"]["boundary_weight"] < 0 or config["loss"]["boundary_pos_weight"] <= 0:
        raise ValueError("Boundary loss weights must be nonnegative / positive")

