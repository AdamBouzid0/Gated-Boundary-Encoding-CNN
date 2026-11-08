"""Bounded training-throughput measurements using the production input pipeline."""

from __future__ import annotations

import copy
import csv
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import validate_config
from .data import PancreasDataset, read_manifest, seed_worker, select_records, validate_splits
from .engine import autocast_context, choose_device, set_seed
from .losses import JointLoss
from .models import build_model, count_parameters


def _atomic_report(path: Path, report: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _patch_digest(batch: dict) -> str:
    digest = hashlib.sha256()
    for key in ("image", "label", "boundary"):
        tensor = batch[key].contiguous()
        digest.update(f"{key}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_variant(config, records, device, batches, result, reference_batches, on_batch):
    started = time.perf_counter()
    rows = result["batch_rows"]
    loader = iterator = None
    loop_started = None
    try:
        set_seed(config["seed"])
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        model = build_model(config).to(device)
        model.train()
        objective = JointLoss(config["loss"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"],
                                     weight_decay=config["training"]["weight_decay"])
        amp = bool(config["training"]["amp"] and device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        optimizer.zero_grad(set_to_none=True)
        workers = config["training"]["num_workers"]
        loader = DataLoader(
            PancreasDataset(records, config, training=True),
            batch_size=config["training"]["batch_size"], shuffle=False,
            num_workers=workers, worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(config["seed"]),
            pin_memory=device.type == "cuda", persistent_workers=workers > 0,
        )
        result["parameters"] = count_parameters(model)
        _synchronize(device)
        result["setup_seconds"] = time.perf_counter() - started
        accumulation = config["training"]["accumulation_steps"]
        loop_started = time.perf_counter()
        for index in range(batches):
            batch_started = time.perf_counter()
            if iterator is None:
                iterator = iter(loader)  # worker startup belongs to first-batch wait
            batch = next(iterator)
            wait_seconds = time.perf_counter() - batch_started
            fingerprint_started = time.perf_counter()
            patch_sha256 = _patch_digest(batch)
            fingerprint_seconds = time.perf_counter() - fingerprint_started
            identity = (tuple(batch["case_id"]), patch_sha256)
            if index in reference_batches and reference_batches[index] != identity:
                raise RuntimeError("Input case order or patches differ between model variants")
            reference_batches.setdefault(index, identity)
            compute_started = time.perf_counter()
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            boundary = batch["boundary"].to(device, non_blocking=True)
            window_start = (index // accumulation) * accumulation
            window_size = min(accumulation, batches - window_start)
            with autocast_context(device, amp):
                output = model(image)
                loss, components = objective(output, label, boundary)
            if not torch.isfinite(loss):
                result["finite_losses"] = False
                raise FloatingPointError(f"Non-finite loss at benchmark batch {index + 1}")
            scaler.scale(loss / window_size).backward()
            attempted = (index + 1) % accumulation == 0 or index + 1 == batches
            updated = False
            if attempted:
                result["optimizer_steps_attempted"] += 1
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # GradScaler lowers the scale when non-finite gradients skip
                # the update. Report actual updates separately from attempts.
                updated = not amp or scaler.get_scale() >= previous_scale
                result["optimizer_steps"] += int(updated)
                optimizer.zero_grad(set_to_none=True)
            values = {name: float(value) for name, value in components.items()}
            if not all(math.isfinite(value) for value in values.values()):
                result["finite_losses"] = False
                raise FloatingPointError(f"Non-finite loss component at benchmark batch {index + 1}")
            _synchronize(device)
            finished = time.perf_counter()
            row = {"batch": index + 1, "case_ids": list(batch["case_id"]), **values,
                   "loading_wait_seconds": wait_seconds,
                   "input_fingerprint_seconds": fingerprint_seconds,
                   "compute_seconds": finished - compute_started,
                   "batch_seconds": finished - batch_started,
                   "optimizer_step_attempted": attempted, "optimizer_step": updated,
                   "patch_sha256": patch_sha256}
            rows.append(row)
            result["batches_completed"] = len(rows)
            on_batch(row)
            print(f"{config['model']['variant']}: batch {index + 1}/{batches}, "
                  f"loss={values['loss']:.5f}, {row['batch_seconds']:.3f}s", flush=True)
        result["status"] = "completed"
    finally:
        finished = time.perf_counter()
        result["total_wall_seconds"] = finished - started
        result["training_loop_seconds"] = finished - loop_started if loop_started is not None else None
        result["loading_wait_seconds"] = sum(row["loading_wait_seconds"] for row in rows)
        result["compute_seconds"] = sum(row["compute_seconds"] for row in rows)
        result["input_fingerprint_seconds"] = sum(row["input_fingerprint_seconds"] for row in rows)
        result["sampled_case_ids"] = [case for row in rows for case in row["case_ids"]]
        result["mean_batch_seconds"] = (result["training_loop_seconds"] / len(rows)
                                         if rows and result["training_loop_seconds"] is not None else None)
        result["steady_mean_batch_seconds"] = (sum(row["batch_seconds"] for row in rows[1:]) / (len(rows) - 1)
                                                if len(rows) > 1 else None)
        result["optimizer_steps_skipped"] = result["optimizer_steps_attempted"] - result["optimizer_steps"]
        result["peak_cuda_allocated_gib"] = (torch.cuda.max_memory_allocated(device) / 2**30
                                              if device.type == "cuda" else None)
        result["peak_cuda_reserved_gib"] = (torch.cuda.max_memory_reserved(device) / 2**30
                                             if device.type == "cuda" else None)
        # The dataset is bounded to requested cases, so even worker prefetch
        # cannot open additional full training volumes. Release workers now.
        del iterator, loader


def benchmark_training(config: dict, variants=("attention_unet", "boundary_attention_unet"),
                       batches: int = 20, device_name: str = "auto",
                       output_dir: str | Path = "runs/runtime_smoke") -> dict:
    """Benchmark fresh models on identical, bounded training-only batches.

    Uses the production dataset/preprocessing, Adam, AMP, clipping and gradient
    accumulation. No checkpoints or held-out volume payloads are loaded/saved.
    Existing reports are never overwritten; pass a fresh output directory.
    """
    overall_started = time.perf_counter()
    config = copy.deepcopy(config)
    validate_config(config)
    variants = tuple(variants)
    if not variants or len(variants) != len(set(variants)) or any(
            name not in {"unet", "attention_unet", "boundary_attention_unet"} for name in variants):
        raise ValueError("Choose distinct model variants: unet, attention_unet, boundary_attention_unet")
    if not isinstance(batches, int) or isinstance(batches, bool) or batches < 1:
        raise ValueError("batches must be a positive integer")
    if config["data"]["patches_per_case"] != 1:
        raise ValueError("This bounded benchmark requires data.patches_per_case=1")
    if not config["data"]["pancreas_label_ids"]:
        raise ValueError("Set audited data.pancreas_label_ids before benchmarking")
    output_dir = Path(output_dir).expanduser().resolve()
    report_path, csv_path = output_dir / "report.json", output_dir / "batches.csv"
    if report_path.exists() or csv_path.exists():
        raise FileExistsError(f"Benchmark outputs already exist in {output_dir}; choose a fresh output_dir")
    records = read_manifest(config["data"]["manifest"])
    split_path = Path(config["data"]["splits"])
    if not split_path.is_file():
        raise FileNotFoundError("Freeze the configured split JSON before benchmarking")
    splits = json.loads(split_path.read_text())
    validate_splits(records, splits)
    training_records = select_records(records, splits, "train")
    batch_size = config["training"]["batch_size"]
    full_train_batches = math.ceil(len(training_records) / batch_size)
    if batches > full_train_batches:
        raise ValueError(f"Requested {batches} batches, but a full training epoch has only {full_train_batches}")
    ordering = torch.randperm(len(training_records), generator=torch.Generator().manual_seed(config["seed"])).tolist()
    selected = [training_records[index] for index in ordering[:min(batches * batch_size, len(training_records))]]
    torch.set_num_threads(config["training"]["cpu_threads"])
    device = choose_device(device_name)
    report = {
        "purpose": "Short training-throughput benchmark; no segmentation-quality or validation result",
        "status": "running", "device": str(device), "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "amp": bool(config["training"]["amp"] and device.type == "cuda"),
        "seed": config["seed"], "requested_batches": batches,
        "batch_size": batch_size, "accumulation_steps": config["training"]["accumulation_steps"],
        "full_train_cases": len(training_records), "full_train_batches": full_train_batches,
        "sampled_case_ids": [record["case_id"] for record in selected],
        "selection_policy": "Seeded permutation of frozen training cases; same ordered subset for every variant",
        "input_setup_seconds": time.perf_counter() - overall_started,
        "config": json.loads(json.dumps(config, default=str)), "variants": {},
        "projection_note": "Training-only extrapolation from a short sample; excludes full-volume validation, checkpointing and evaluation. Volume size, storage contention and cache state can change throughput substantially.",
        "cache_order_note": "Variants run sequentially in listed order. Later variants may benefit from filesystem/OS caches; caches are not flushed. CUDA peak counters are reset for each fresh model.",
        "timing_note": "Total variant wall time includes model/optimizer/loader setup. Batch times include visible loading/preprocessing wait, input hashing, device transfer and synchronized training compute. Worker preprocessing can overlap compute. Loop mean includes report-writing overhead; steady mean omits the first batch.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    csv_fields = ["variant", "batch", "case_ids", "loss", "seg_loss", "boundary_loss", "loading_wait_seconds",
                  "input_fingerprint_seconds", "compute_seconds", "batch_seconds", "optimizer_step_attempted",
                  "optimizer_step", "patch_sha256"]
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=csv_fields).writeheader()
    reference_batches = {}
    for variant in variants:
        result = {"status": "running", "batch_rows": [], "batches_completed": 0,
                  "optimizer_steps": 0, "optimizer_steps_attempted": 0, "finite_losses": True}
        report["variants"][variant] = result
        current = copy.deepcopy(config)
        current["model"]["variant"] = variant

        def on_batch(row):
            with csv_path.open("a", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=csv_fields).writerow(
                    {"variant": variant, **row, "case_ids": ";".join(row["case_ids"])})
            _atomic_report(report_path, report)

        try:
            _run_variant(current, selected, device, batches, result, reference_batches, on_batch)
        except Exception as error:
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
        if result.get("mean_batch_seconds") is not None and result["status"] == "completed":
            projected_epoch = result["mean_batch_seconds"] * full_train_batches
            result["projected_train_epoch_seconds"] = projected_epoch
            result["projected_configured_epochs_seconds"] = projected_epoch * config["training"]["epochs"]
            result["projected_50_epochs_seconds"] = projected_epoch * 50
        _atomic_report(report_path, report)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    completed = [item for item in report["variants"].values() if item["status"] == "completed"]
    report["status"] = "completed" if len(completed) == len(variants) else "partial" if completed else "failed"
    report["patches_identical_across_variants"] = (all(
        [(row["case_ids"], row["patch_sha256"]) for row in item["batch_rows"]] ==
        [(row["case_ids"], row["patch_sha256"]) for row in completed[0]["batch_rows"]]
        for item in completed[1:]) if len(completed) > 1 else None)
    report["total_wall_seconds"] = time.perf_counter() - overall_started
    _atomic_report(report_path, report)
    return report
