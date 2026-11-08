"""Training, frozen validation selection, and full-volume inference."""

from __future__ import annotations

import csv
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import validate_config
from .data import (PancreasDataset, make_or_load_splits, read_manifest,
                   seed_worker, select_records, validate_splits)
from .losses import JointLoss
from .metrics import aggregate_metrics, segmentation_metrics
from .models import build_model, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu or install a CUDA PyTorch build")
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("Supported device types are cpu and cuda")
    return device


def prepare_data(config: dict) -> tuple[list[dict], dict]:
    records = read_manifest(config["data"]["manifest"])
    expected = config["data"]["expected_cases"]
    if expected is not None and len(records) != expected:
        print(f"NOTE: manifest has {len(records)} cases; expected_cases is {expected}. "
              "Confirm the manifest includes the intended cohort.")
    for record in records:
        for key in ("image", "label"):
            if not Path(record[key]).is_file():
                raise FileNotFoundError(record[key])
    splits = make_or_load_splits(records, config["data"]["splits"],
                                 config["data"]["split_fractions"], config["seed"])
    print("Patient-grouped cases:", {name: len(splits[name]) for name in ("train", "val", "test")})
    return records, splits


def make_loader(records: list[dict], config: dict, training: bool,
                generator: torch.Generator | None = None) -> DataLoader:
    workers = config["training"]["num_workers"]
    return DataLoader(
        PancreasDataset(records, config, training),
        batch_size=config["training"]["batch_size"] if training else 1,
        shuffle=training, num_workers=workers, worker_init_fn=seed_worker,
        generator=generator, pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if enabled else nullcontext()


def train_epoch(model, loader, optimizer, objective, scaler, device,
                accumulation_steps: int, amp: bool, grad_clip: float) -> dict:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "seg_loss": 0.0, "boundary_loss": 0.0}
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        boundary = batch["boundary"].to(device, non_blocking=True)
        # A short last accumulation window still receives the correct scaling.
        window_start = (step // accumulation_steps) * accumulation_steps
        window_size = min(accumulation_steps, len(loader) - window_start)
        with autocast_context(device, amp):
            output = model(image)
            loss, components = objective(output, label, boundary)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at training step {step}")
        scaler.scale(loss / window_size).backward()
        if (step + 1) % accumulation_steps == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        for name in totals:
            totals[name] += float(components[name])
    return {name: value / len(loader) for name, value in totals.items()}


@torch.inference_mode()
def evaluate_loader(model, loader, config: dict, device: torch.device,
                    save_dir: str | Path | None = None) -> tuple[dict, list[dict]]:
    model.eval()
    rows = []
    amp = bool(config["training"]["amp"] and device.type == "cuda")
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    def predictor(patches):
        # The boundary ENCODER stays active. Only its auxiliary decoder/head
        # can be omitted, since they are not used by segmentation inference.
        with autocast_context(device, amp):
            return model(patches, compute_boundary=False)["seg_logits"].float()

    for batch in tqdm(loader, desc="volumes", leave=False):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        logits = sliding_window_inference(
            inputs=batch["image"], roi_size=tuple(config["data"]["patch_size"]),
            sw_batch_size=config["evaluation"]["sliding_window_batch_size"],
            predictor=predictor, overlap=config["evaluation"]["overlap"], mode="gaussian",
            sw_device=device, device=torch.device("cpu"), progress=False,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        prediction = logits.argmax(dim=1)[0].numpy().astype(np.uint8)
        target = batch["label"][0, 0].numpy().astype(bool)
        spacing = batch["spacing"][0].numpy()
        metrics = segmentation_metrics(prediction, target, spacing,
                                        config["evaluation"]["boundary_tolerance_mm"])
        case_id = batch["case_id"][0]
        rows.append({"case_id": case_id, "patient_id": batch["patient_id"][0],
                     **metrics, "inference_seconds": elapsed})
        if save_dir is not None:
            # Prevent case IDs in a manifest from becoming output paths.
            safe_id = case_id  # restricted to filename-safe characters by read_manifest
            affine = batch["affine"][0].numpy()
            prediction_file = nib.Nifti1Image(prediction, affine)
            prediction_file.header.set_xyzt_units("mm")
            nib.save(prediction_file, save_dir / f"{safe_id}.nii.gz")
    summary = aggregate_metrics(rows)
    summary["mean_inference_seconds"] = float(np.mean([row["inference_seconds"] for row in rows]))
    return summary, rows


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def atomic_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(config: dict, device_name: str = "auto", resume: str | None = None) -> Path:
    validate_config(config)
    set_seed(config["seed"])
    torch.set_num_threads(config["training"]["cpu_threads"])
    device = choose_device(device_name)
    if not config["data"]["pancreas_label_ids"]:
        raise ValueError("Inspect label IDs, then set data.pancreas_label_ids (see README)")
    records, splits = prepare_data(config)
    run_dir = Path(config["training"]["output_dir"]) / config["model"]["variant"] / f"seed_{config['seed']}"
    if resume and Path(resume).resolve().parent != run_dir.resolve():
        raise ValueError("Resume into the checkpoint's original run directory; check --output-dir and variant")
    if (run_dir / "last.pt").exists() and resume is None:
        raise FileExistsError(f"Existing run at {run_dir}; use --resume or a new --output-dir")
    run_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(config).to(device)
    objective = JointLoss(config["loss"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"],
                                  weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    amp = bool(config["training"]["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    generator = torch.Generator().manual_seed(config["seed"])
    start_epoch, best = 0, -1.0
    if resume:
        # Only load checkpoints you created or otherwise trust.
        state = torch.load(resume, map_location=device, weights_only=False)
        if state["splits"] != splits or state["config"]["seed"] != config["seed"]:
            raise ValueError("Resume seed or data splits do not match the checkpoint")
        for section in ("model", "data", "augmentation", "loss"):
            if state["config"][section] != config[section]:
                raise ValueError(f"Resume {section} configuration does not match the checkpoint")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best = state["epoch"] + 1, state["best_val_dice"]
        generator.set_state(state["loader_rng"].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda_rng"]])
    save_json(run_dir / "config.json", config)
    save_json(run_dir / "splits.json", splits)
    train_loader = make_loader(select_records(records, splits, "train"), config, True, generator)
    val_loader = make_loader(select_records(records, splits, "val"), config, False)
    save_json(run_dir / "environment.json", {
        "torch": torch.__version__, "device": str(device), "parameters": count_parameters(model),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    })
    print(f"{config['model']['variant']}: {count_parameters(model):,} trainable parameters on {device}")
    log_path = run_dir / "history.csv"
    fields = ["epoch", "loss", "seg_loss", "boundary_loss", "val_dice", "val_boundary_f1",
              "learning_rate", "epoch_seconds", "peak_cuda_gib"]
    for epoch in range(start_epoch, config["training"]["epochs"]):
        start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        components = train_epoch(model, train_loader, optimizer, objective, scaler, device,
                                 config["training"]["accumulation_steps"], amp,
                                 config["training"]["grad_clip"])
        validation = None
        if (epoch + 1) % config["training"]["validate_every"] == 0 or epoch + 1 == config["training"]["epochs"]:
            validation, _ = evaluate_loader(model, val_loader, config, device)
            scheduler.step(validation["dice"])
        is_best = validation is not None and validation["dice"] > best
        if is_best:
            best = validation["dice"]
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "best_val_dice": best, "config": config, "splits": splits,
            "loader_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }
        atomic_checkpoint(run_dir / "last.pt", state)
        if is_best:
            atomic_checkpoint(run_dir / "best.pt", state)
        row = {"epoch": epoch + 1, **components,
               "val_dice": validation["dice"] if validation else "",
               "val_boundary_f1": validation["boundary_f1"] if validation else "",
               "learning_rate": optimizer.param_groups[0]["lr"],
               "epoch_seconds": time.perf_counter() - start,
               "peak_cuda_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0}
        exists = log_path.exists()
        with log_path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        print(json.dumps(row))
    return run_dir


def evaluate_checkpoint(checkpoint: str | Path, split: str = "test", device_name: str = "auto",
                        save_predictions: bool = False) -> dict:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    torch.set_num_threads(config["training"]["cpu_threads"])
    device = choose_device(device_name)
    records = read_manifest(config["data"]["manifest"])
    # Use the frozen split saved WITH the checkpoint, not a newly generated one.
    splits = state["splits"]
    validate_splits(records, splits)
    model = build_model(config).to(device)
    model.load_state_dict(state["model"])
    loader = make_loader(select_records(records, splits, split), config, False)
    output_dir = Path(checkpoint).resolve().parent / f"evaluation_{split}"
    output_dir.mkdir(exist_ok=True)
    summary, rows = evaluate_loader(model, loader, config, device,
                                    output_dir / "predictions" if save_predictions else None)
    summary.update(variant=config["model"]["variant"], parameters=count_parameters(model),
                   checkpoint=str(Path(checkpoint).resolve()),
                   boundary_tolerance_mm=config["evaluation"]["boundary_tolerance_mm"])
    save_json(output_dir / "summary.json", summary)
    with (output_dir / "per_case.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    return summary

