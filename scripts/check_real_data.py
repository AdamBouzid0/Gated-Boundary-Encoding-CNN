"""Inspect a few training cases and check both models on fixed real patches.

This is a data/gradient check, not a validation score or a full training run.
Run from an installed project, or with PYTHONPATH=src.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import resource
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F

from pancreas_seg.boundaries import inner_boundary
from pancreas_seg.config import load_config
from pancreas_seg.data import PancreasDataset, read_manifest, validate_splits
from pancreas_seg.engine import set_seed
from pancreas_seg.losses import JointLoss
from pancreas_seg.models import build_model, count_parameters


def percentiles(values):
    levels = [0, 0.5, 1, 50, 99, 99.5, 100]
    return {str(q): float(v) for q, v in zip(levels, np.percentile(values, levels))}


def overlay(image, mask, boundary, output, title, limits=None):
    """Use slices through foreground medians in a canonical RAS array."""
    coordinates = np.argwhere(mask)
    center = np.median(coordinates, axis=0).astype(int) if len(coordinates) else np.array(mask.shape) // 2
    if limits is None:
        limits = np.percentile(image, [1, 99])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, ax in enumerate(axes):
        section = [slice(None)] * 3
        section[axis] = int(center[axis])
        section = tuple(section)
        ax.imshow(image[section].T, cmap="gray", origin="lower", vmin=limits[0], vmax=limits[1])
        organ, edge = mask[section].T, boundary[section].T
        colors = np.zeros((*organ.shape, 4))
        colors[organ.astype(bool)] = [1, 0.8, 0, 0.25]
        colors[edge.astype(bool)] = [1, 0.15, 0.1, 0.85]
        ax.imshow(colors, origin="lower")
        ax.set_title(f"{('Sagittal', 'Coronal', 'Axial')[axis]} — index {center[axis]}")
        ax.set_axis_off()
    fig.suptitle(title + " | yellow: pancreas; red: inner boundary")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def select_training(records, splits, count, audit_path):
    training = [r for r in records if r["case_id"] in set(splits["train"])]
    # Mask sizes only select diagnostic training examples; no test images load.
    sizes_by_id = {}
    if audit_path.is_file():
        with audit_path.open(newline="") as stream:
            sizes_by_id = {row["case_id"]: int(row["foreground_voxels"])
                           for row in csv.DictReader(stream) if row["status"] == "kept"}
    sizes = [(sizes_by_id[r["case_id"]] if r["case_id"] in sizes_by_id else
              int(np.count_nonzero(np.asanyarray(nib.load(r["label"]).dataobj))), r) for r in training]
    sizes.sort(key=lambda item: (item[0], item[1]["case_id"]))
    positions = np.linspace(0.1, 0.9, min(count, len(sizes))) * (len(sizes) - 1)
    return [sizes[i][1] for i in sorted(set(np.round(positions).astype(int)))]


def inspect_cases(records, config, output_dir, sanity_patch):
    diagnostic = copy.deepcopy(config)
    diagnostic["data"]["foreground_probability"] = 1.0
    diagnostic["augmentation"]["affine_probability"] = 1.0
    dataset = PancreasDataset(records, diagnostic, training=True)
    summaries, patches = [], []
    for index, record in enumerate(records):
        image_obj = nib.load(record["image"])
        raw = image_obj.get_fdata(dtype=np.float32)
        mask = np.asanyarray(nib.load(record["label"]).dataobj) > 0
        if not np.isfinite(raw).all():
            raise ValueError(f"{record['case_id']}: non-finite CT payload")
        summary = {"case_id": record["case_id"], "split": "train",
                   "shape": list(raw.shape), "spacing": list(map(float, image_obj.header.get_zooms()[:3])),
                   "spatial_unit": image_obj.header.get_xyzt_units()[0],
                   "orientation": list(nib.aff2axcodes(image_obj.affine)),
                   "foreground_voxels": int(mask.sum()),
                   "ct_percentiles_sampled": percentiles(raw.ravel()[::max(1, raw.size // 1_000_000)]),
                   "pancreas_ct_percentiles": percentiles(raw[mask]),
                   "ct_payload_finite": True}
        canonical = nib.as_closest_canonical(nib.Nifti1Image(raw, image_obj.affine))
        canonical_mask = nib.as_closest_canonical(nib.Nifti1Image(mask.astype(np.uint8), image_obj.affine))
        canonical_mask = np.asarray(canonical_mask.dataobj, dtype=bool)
        overlay(np.asarray(canonical.dataobj), canonical_mask, inner_boundary(canonical_mask),
                output_dir / f"{record['case_id']}_raw.png", record["case_id"] + " raw CT", [-150, 250])
        del raw, mask, canonical, canonical_mask
        sample = dataset[index]
        for attempt in range(8):
            size = np.array(sample["label"].shape[1:])
            start = (size - sanity_patch) // 2
            crop = (slice(None), *[slice(int(a), int(a + n)) for a, n in zip(start, sanity_patch)])
            fixed = {key: sample[key][crop].unsqueeze(0).contiguous()
                     for key in ("image", "label", "boundary")}
            if fixed["label"].sum() > 0 and fixed["boundary"].sum() > 0:
                break
            sample = dataset[index]
        else:
            raise ValueError(f"{record['case_id']}: could not sample foreground and boundary for sanity patch")
        image, label, boundary = [sample[key][0].numpy() for key in ("image", "label", "boundary")]
        expected = inner_boundary(label)
        interior = (slice(1, -1),) * 3
        if not np.array_equal(boundary[interior], expected[interior]):
            raise AssertionError("Boundary differs from the augmented mask away from crop faces")
        if not np.isin(label, [0, 1]).all() or not np.all(boundary <= label):
            raise AssertionError("Augmentation corrupted binary labels or inner-boundary targets")
        overlay(image, label, boundary, output_dir / f"{record['case_id']}_augmented.png",
                record["case_id"] + " augmented preprocessed patch")
        summary.update(augmented_patch_shape=list(image.shape),
                       augmented_foreground_voxels=int(label.sum()),
                       augmented_boundary_voxels=int(boundary.sum()), boundary_interior_matches=True,
                       sanity_foreground_voxels=int(fixed["label"].sum()),
                       sanity_boundary_voxels=int(fixed["boundary"].sum()))
        summaries.append(summary)
        patches.append(fixed)
        print(f"Inspected training case {record['case_id']}", flush=True)
    return summaries, patches


def has_gradient(module):
    return any(p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)
               for p in module.parameters())


def train_fixed_patches(config, patches, steps, device):
    reports = {}
    amp = bool(config["training"]["amp"] and device.type == "cuda")
    patches = [{key: value.to(device) for key, value in patch.items()} for patch in patches]

    def precision():
        return torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()

    for variant in ("attention_unet", "boundary_attention_unet"):
        set_seed(config["seed"])
        settings = copy.deepcopy(config)
        settings["model"]["variant"] = variant
        model = build_model(settings).to(device)
        objective = JointLoss(settings["loss"]).to(device)
        gradients = {}
        if variant == "boundary_attention_unet":
            model.eval()
            # Gradient routing is independent of patch size; avoid a second
            # full-sized backward pass when probing configured GPU memory.
            # A corner of a large patch can be entirely padded, especially
            # for truncated scans. Probe a real annotated boundary instead.
            points = torch.nonzero(patches[0]["boundary"][0, 0], as_tuple=False)
            center = points[len(points) // 2].tolist()
            start = [min(max(int(c) - 16, 0), n - 32)
                     for c, n in zip(center, patches[0]["image"].shape[2:])]
            crop = (slice(None), slice(None), *[slice(a, a + 32) for a in start])
            first = {key: value[crop] for key, value in patches[0].items()}
            output = model(first["image"], compute_boundary=False)
            objective.segmentation(output["seg_logits"], first["label"]).backward()
            gradients["segmentation_reaches_boundary_encoder"] = bool(has_gradient(model.boundary_encoder))
            gradients["segmentation_reaches_fusion"] = bool(has_gradient(model.bottleneck_fusion))
            model.zero_grad(set_to_none=True)
            output = model(first["image"])
            F.binary_cross_entropy_with_logits(output["boundary_logits"], first["boundary"]).backward()
            gradients["auxiliary_reaches_boundary_encoder"] = bool(has_gradient(model.boundary_encoder))
            gradients["auxiliary_excludes_main_encoder"] = all(p.grad is None for p in model.encoder.parameters())
            if not all(gradients.values()):
                raise AssertionError(f"Incorrect gradient path: {gradients}")
            model.zero_grad(set_to_none=True)
            output = None
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["training"]["learning_rate"],
                                     weight_decay=settings["training"]["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", enabled=amp)

        def measure():
            with torch.no_grad(), precision():
                components = [objective(model(p["image"]), p["label"], p["boundary"])[1] for p in patches]
            result = {key: float(torch.stack([c[key] for c in components]).mean()) for key in components[0]}
            if not all(np.isfinite(value) for value in result.values()):
                raise FloatingPointError(f"{variant}: non-finite fixed-patch mean")
            return result

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        initial = measure()
        history = []
        update_times = []
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for step in range(steps):
            step_started = time.perf_counter()
            patch = patches[step % len(patches)]
            optimizer.zero_grad(set_to_none=True)
            with precision():
                loss, components = objective(model(patch["image"]), patch["label"], patch["boundary"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{variant}: non-finite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f"{variant}: non-finite gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["training"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            history.append({key: float(value) for key, value in components.items()})
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            update_times.append(time.perf_counter() - step_started)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        final = measure()
        reports[variant] = {"parameters": count_parameters(model), "initial_fixed_patch_mean": initial,
                            "final_fixed_patch_mean": final, "loss_decreased": final["loss"] < initial["loss"],
                            "steps": steps, "history": history, "gradient_checks": gradients,
                            "update_seconds": elapsed, "mean_update_seconds": elapsed / steps,
                            "update_seconds_per_step": update_times,
                            "mean_update_seconds_after_first": float(np.mean(update_times[1:])) if steps > 1 else None,
                            "process_peak_rss_mib_since_launch": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
                            "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else None,
                            "all_losses_and_gradients_finite": True}
        print(f"{variant}: {initial['loss']:.4f} -> {final['loss']:.4f}; {elapsed / steps:.3f}s/update", flush=True)
        del model, optimizer, loss, components
        output = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/boundary_attention.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path("data/totalsegmentator_filter/checks"))
    parser.add_argument("--cases", type=int, default=3)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--patch-size", type=int, nargs=3, default=[32, 32, 32])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable in this process")
    if args.steps < 1 or args.cases < 1 or any(n < 32 or n % 16 for n in args.patch_size):
        parser.error("Use positive cases/steps and patch dimensions >=32 divisible by 16")
    if any(n > configured for n, configured in zip(args.patch_size, config["data"]["patch_size"])):
        parser.error("Sanity patch must fit inside the configured training patch")
    torch.set_num_threads(config["training"]["cpu_threads"])
    set_seed(config["seed"])
    records = read_manifest(config["data"]["manifest"])
    splits = json.loads(Path(config["data"]["splits"]).read_text())
    validate_splits(records, splits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_training(records, splits, args.cases, Path(config["data"]["manifest"]).with_name("audit.csv"))
    cases, patches = inspect_cases(selected, config, args.output_dir, np.asarray(args.patch_size))
    report = {"purpose": "Training-only data and gradient sanity check; not a model comparison",
              "seed": config["seed"], "torch": torch.__version__, "device": str(device),
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "amp": bool(config["training"]["amp"] and device.type == "cuda"),
              "configured_patch_size": config["data"]["patch_size"], "sanity_patch_size": args.patch_size,
              "batch_size": 1, "model_settings": config["model"],
              "data_settings": config["data"], "cases": cases,
              "diagnostic_overrides": {"foreground_probability": 1.0, "affine_probability": 1.0},
              "notes": ["Raw CT payloads checked only for the listed training cases.",
                        "Raw overlay display window [-150,250] does not alter model preprocessing.",
                        "Fixed patches come from already augmented enlarged-patch boundary targets.",
                        "Initial/final losses are training-mode means over the same fixed patches.",
                        "CPU RSS is cumulative process peak, including volume loading and both variants.",
                        "GPU timing includes diagnostic finite-gradient checks; GPU memory includes all fixed input patches.",
                        "Gradient routing is checked on 32-cubed inputs; timing/memory use sanity_patch_size.",
                        "Update timings include diagnostic finite-gradient checks; the first update can include CUDA initialization costs.",
                        "This does not establish convergence or validation quality; memory fit applies only to the reported device/settings."]}
    report["models"] = train_fixed_patches(config, patches, args.steps, device)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Report: {args.output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
