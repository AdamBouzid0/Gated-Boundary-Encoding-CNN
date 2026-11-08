# 3D pancreas segmentation: Attention U-Net + boundary encoder

A PyTorch implementation for 3D pancreas segmentation, comparing Attention
U-Net with an extension that adds a supervised boundary encoder and
bottleneck fusion. The baseline ports the 2018 repository's **CT
single-attention / multiscale-output 3D U-Net**. The repository includes
dataset auditing, training and evaluation commands, an optional ungated
U-Net control, and a notebook workflow.

## Dataset and attribution

This project uses **TotalSegmentator v1**, created by **Jakob Wasserthal
et al.** at **University Hospital Basel**:
[original Zenodo release](https://zenodo.org/records/6802614),
[dataset paper](https://arxiv.org/abs/2208.05868). The release is licensed
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); retain the dataset authors, release and license
attribution. CT images and labels are not included in this repository.
Generated cohort manifests, split files and machine-specific reports are
also excluded from version control and must be created locally.

See [docs/TOTALSEGMENTATOR.md](docs/TOTALSEGMENTATOR.md) for layout,
curation, official partitions and an example local audit. Augmentation is
disabled in the current recipe; random training-patch sampling remains
enabled.

## 1. Install

Use Python 3.10 or newer and a virtual environment. Install the PyTorch build
appropriate for your machine first; see <https://pytorch.org/get-started/locally/>.
From this project directory:

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
# Install your CPU/CUDA PyTorch build here, then:
python -m pip install -e ".[dev]"
# For the notebook workflow:
python -m pip install -e ".[notebook]"
```

There is no need to install the old repository's `torchsample` or old
`torchvision` dependencies. This is a modern port with upstream attribution,
not a drop-in patch for the old training framework. See
[docs/UPSTREAM.md](docs/UPSTREAM.md) for exact source mapping and differences.

## 2. Run without any real data

```bash
python -m pytest -q
python scripts/smoke_test.py
```

The smoke test creates six **synthetic ellipsoid** NIfTI scans, trains the
baseline and boundary variant for one CPU epoch each, resumes the boundary
variant for another epoch, reconstructs full volumes, computes metrics and
saves NIfTI predictions. It is a plumbing check, not a pancreas benchmark or evidence
that the method improves segmentation.

## 3. Audit and connect the dataset

Each TotalSegmentator case contains `ct.nii.gz` and separate organ masks.
For this binary task, only `segmentations/pancreas.nii.gz` is a target;
other organ masks are not input channels. Training reads these original
files under `../Totalsegmentator_dataset/<case_id>/`, following
`data/totalsegmentator_filter/manifest.csv` and `splits_official.json`.
Training updates use the frozen training partition; validation uses its
separate partition. Boundary targets are generated from the pancreas
masks in memory; separate boundary files are not required.

A fresh clone does not contain generated audit reports or frozen splits.
Download and extract the dataset separately, then run the audit before
training or loading real data in the notebook. This example assumes the
dataset is in the sibling directory `../Totalsegmentator_dataset`;
change `--root` to its actual location. Source files are left unchanged:

```bash
python filter_totalsegmentator_pancreas.py \
  --root ../Totalsegmentator_dataset \
  --output-dir data/totalsegmentator_filter \
  --grid-tolerance-voxels 0.25
```

Existing reports are protected. For a deliberate re-audit, choose a fresh
output directory or use `--overwrite`. Training uses the generated
manifest in `totalsegmentator_filter/`.

The audit checks mask voxel values and CT/mask header geometry. It writes
the retained manifest, a valid-including-empty manifest, per-case audit
CSV, summary JSON and retained IDs. It distinguishes empty annotations
from missing or unreadable files. It does not fully decompress CT payloads.
The explicit quarter-voxel tolerance accepts small CT/mask header
differences measured in the inspected download; source files are
unchanged. The
[dataset notes](docs/TOTALSEGMENTATOR.md#small-ctmask-header-differences)
record the measurements and rationale.
Do not infer anatomical absence from an empty annotation, or full pancreas
coverage from a nonempty annotation. Border-touching masks need review.

The generated manifest uses these fields:

| Field | Meaning |
|---|---|
| `case_id` | Unique scan/example ID |
| `patient_id` | Same ID for ALL scans/patches from the same patient |
| `image` | NIfTI image path |
| `label` | Corresponding segmentation label path |
| `official_split` | Source `train`, `val` or `test` partition |

Paths are relative to the CSV's directory, or absolute. Files must be
aligned 3D scalar volumes. The supplied `meta.csv` lists official splits
for all 1,204 IDs: 1,082 train, 57 validation and 65 test, before filtering.
These assignments must be preserved when selecting retained cases.
There is no patient-ID column in this metadata: the manifest therefore
uses case IDs as grouping proxies, without claiming independent patients.
If a repeated-patient mapping becomes available, import it and recheck
leakage before training. For other datasets, the generic manifest helper
`scripts/prepare_manifest.py` remains available. The supplied
`data/manifest.example.csv` illustrates actual TotalSegmentator paths and
official labels using case IDs as patient proxies.

Inspect shapes, orientation, spacing, alignment and mask values:

```bash
python scripts/inspect_dataset.py \
  --manifest data/totalsegmentator_filter/manifest.csv --limit 5 \
  --grid-tolerance-voxels 0.25
```

After configuring the curated manifest and split paths, freeze the
official partitions before training:

```bash
python scripts/prepare_splits.py --config configs/boundary_attention.yaml
```

The trainer also honors `official_split` automatically. Partially specified
or unknown partition labels and conflicting patient assignments fail
validation.

The audit checks that individual pancreas masks contain only background
0 and foreground 1, matching the configured `pancreas_label_ids: [1]`.
This value describes the separate binary organ file; it is not a
global class ID in a merged multilabel segmentation. Inspect image/mask
overlays and boundary targets, including sampled training patches,
before a long run.

## 4. Configure the first experiment

Edit `configs/boundary_attention.yaml`:

- Use the curated manifest and a frozen split file for this exact cohort.
  `expected_cases: 795` records the example local audit, not a fixed
  release-wide pancreas count. Set it to your audited retained count or
  null if your copy differs. Preserve official partitions rather than
  applying the random-split fallback.
- `target_spacing_mm: null` preserves the downloaded approximately 1.5 mm
  isotropic grid, with consistent RAS orientation. The headers leave units
  unset, so `spacing_unit_if_unknown: mm` explicitly interprets them using
  the release's CT scale. Known metre/micron units are converted to mm;
  setting the unknown-unit option to null rejects undeclared units.
- Initial CT clipping is `[-200, 400]`, followed by per-volume z-score
  normalization. This choice follows training-case intensity inspection
  and is a starting setting, not a validated optimum.
- Choose patch size from organ coverage and measured memory use on your
  hardware. The default is 96x96x96. Dimensions must be multiples of 16
  and at least 32. If widths or patch size change, use the same core
  settings for both models.
- Spatial warps, flips and intensity noise are disabled. Patch locations
  remain random: the centre is sampled from pancreas foreground with 50%
  probability, otherwise from anywhere in the scan. This is sampling,
  not an image transformation.
  The one-voxel boundary halo gives a temporary 98x98x98 patch, from which
  the boundary is derived before the final 96x96x96 crop.

The three spatial tensor axes follow the canonical NIfTI array order. The
code uses PyTorch's conventional `(N,C,D,H,W)` names; these names alone do
not determine anatomical planes. Affines and spacing remain authoritative.

## 5. Train the baseline and the full model

Open [notebooks/train_and_evaluate.ipynb](notebooks/train_and_evaluate.ipynb)
in VS Code or Jupyter with the project environment as its kernel. Choose
`VARIANT` (`attention_unet` or `boundary_attention_unet`) in the controls
cell. The notebook shows the data, boundaries and an optional local runtime report;
training, validation and final test evaluation each have an explicit flag
that starts as `False`:

- `RUN_TRAINING=True` starts the selected model's training.
- `RUN_VALIDATION=True` evaluates the trained checkpoint and saves predictions
  for the notebook's overlays.
- `RUN_FINAL_TEST=True` evaluates the held-out test cases after settings are
  frozen.

The default `RUN_NAME` is `no_augmentation_v1`, so notebook outputs go under
`runs/no_augmentation_v1/<variant>/seed_42/`. For the second model, change
`VARIANT` and rerun from the controls cell with the same other settings.
To resume, set both `RESUME=True` and `RUN_TRAINING=True`.

Run terminal commands from the cloned project directory with its virtual
environment activated:

```bash
source .venv/bin/activate
```

Train Attention U-Net first:

```bash
python train_segmentation.py -c configs/boundary_attention.yaml \
  --variant attention_unet \
  --device cuda \
  --output-dir runs/no_augmentation_v1
```

Then train the boundary variant:

```bash
python train_segmentation.py -c configs/boundary_attention.yaml \
  --variant boundary_attention_unet \
  --device cuda \
  --output-dir runs/no_augmentation_v1
```

Run the models one at a time. These commands match the notebook's default
output directories and use the configured 50 epochs, disabled augmentation,
and audited pancreas label 1. Add `--epochs 10`, for example, to override
the epoch count. Use `--device cpu` if CUDA is unavailable.

Both runs reuse the same curated cases, frozen official partitions and
core settings. The manifest and split digest must agree; do not reuse a
split file from a different cohort. Patient grouping checks use the supplied
`patient_id`, which is currently a case-ID proxy for this release. Test cases
are reserved for final evaluation.

An optional `--variant unet` gives the ungated control. There is no automatic
sweep or mandatory intermediate boundary-head-only model.

Training includes Adam, soft Dice segmentation loss, a weighted BCE boundary
loss in the full model, CUDA mixed precision, gradient accumulation, gradient
clipping and validation-based learning-rate reduction.

With the commands above, each variant writes to
`runs/no_augmentation_v1/<variant>/seed_42/`:

```text
config.json          frozen run settings
splits.json          frozen case assignments and manifest digest
environment.json     PyTorch/device/parameter count
history.csv          losses, validation scores, epoch timing, peak CUDA memory
best.pt              checkpoint selected using validation Dice
last.pt              latest checkpoint for resuming
```

`last.pt` is saved after each completed epoch. `best.pt` is saved when
validation Dice improves. Validation reconstructs full volumes every five
epochs and at the final epoch, so the first `best.pt` normally appears after
epoch 5, or at the end of a shorter run. Test data are not evaluated inside
the training loop. Without `--output-dir`, the CLI instead uses the YAML
default `runs/<variant>/seed_42/`.

Existing checkpoints are not silently overwritten. To continue the baseline:

```bash
python train_segmentation.py -c configs/boundary_attention.yaml \
  --variant attention_unet \
  --device cuda \
  --output-dir runs/no_augmentation_v1 \
  --resume runs/no_augmentation_v1/attention_unet/seed_42/last.pt
```

To continue the boundary model:

```bash
python train_segmentation.py -c configs/boundary_attention.yaml \
  --variant boundary_attention_unet \
  --device cuda \
  --output-dir runs/no_augmentation_v1 \
  --resume runs/no_augmentation_v1/boundary_attention_unet/seed_42/last.pt
```

Resume into the same run directory with the same data, model and loss settings.
`--epochs` sets the total target epoch count, including epochs already
completed. Model, optimizer, scheduler, scaler and RNG states are restored.
Worker-side sampling/prefetch state is not fully checkpointed, so exact
bitwise continuation with multiprocessing is not promised. Load trusted
checkpoints. For a separate experiment, choose a fresh `--output-dir` or
notebook `RUN_NAME`.

### Measure runtime with 20 batches

The runtime check processes **20 actual training batches per model**, not
20 epochs. With batch size 1 and accumulation 2, this schedules 10 optimizer
updates per model. It includes CT loading and preprocessing through the
training DataLoader. After creating the manifest and splits, run an initial
measurement:

```bash
python scripts/benchmark_training.py --config configs/boundary_attention.yaml \
  --device cuda --batches 20 \
  --variants unet attention_unet boundary_attention_unet \
  --output-dir runs/runtime_smoke
```

This saves timing/loss reports, not model checkpoints. The separate synthetic
`scripts/smoke_test.py` does save its own small test checkpoints.
Existing benchmark outputs are protected; choose a fresh output directory
when repeating a measurement.

The reports are local outputs and are not committed. The notebook can
load `runs/runtime_smoke/report.json` or another supported local cache if
present; otherwise set its optional
`RUN_BENCHMARK` flag to create a measurement. Training-only projections
exclude full-volume validation and checkpoint writes and vary with case
sizes, startup overhead and disk caching. See
[verification/README.md](verification/README.md) for reproducible checks
and their limitations.

## 6. Validate, then evaluate the held-out test set

Use validation to inspect predictions and choose settings or checkpoints:

```bash
python evaluate.py \
  --checkpoint runs/no_augmentation_v1/attention_unet/seed_42/best.pt \
  --split val --device cuda --save-predictions

python evaluate.py \
  --checkpoint runs/no_augmentation_v1/boundary_attention_unet/seed_42/best.pt \
  --split val --device cuda --save-predictions
```

After the training recipe and checkpoint-selection rule are frozen, evaluate
each validation-selected checkpoint on the held-out official test partition:

```bash
python evaluate.py \
  --checkpoint runs/no_augmentation_v1/attention_unet/seed_42/best.pt \
  --split test --device cuda

python evaluate.py \
  --checkpoint runs/no_augmentation_v1/boundary_attention_unet/seed_42/best.pt \
  --split test --device cuda
```

Results are saved beside the checkpoint in `evaluation_val/` or
`evaluation_test/`. Add `--save-predictions` to export NIfTI predictions.

Evaluation produces per-case CSV scores and a JSON summary:

- Dice, IoU, precision and recall.
- Boundary F1 at a configurable physical tolerance (2 mm by default).
- Symmetric HD95 in mm.
- Mean inference time and parameter count. Training peak CUDA memory is in
  `history.csv`.

Boundary F1 here is a tolerance-based **boundary-voxel F1**, not an
area-weighted normalized surface Dice implementation. HD95 is undefined
when exactly one surface is missing; it is recorded as null and the number
of such cases is reported rather than silently replacing it with a score.
Both-empty masks receive perfect overlap; missed nonempty organs receive
zero overlap. Summaries are unweighted means across volumes.

Saved predictions are in the **canonical/preprocessed grid**, using its
correct affine, not automatically restored to the original acquisition grid.
Restore to the original grid before submitting to any benchmark requiring it.

## Where to start reading the code

1. `src/pancreas_seg/models.py`: original-style gate and added boundary stream.
2. `src/pancreas_seg/boundaries.py`: erosion-derived boundary targets.
3. `src/pancreas_seg/losses.py`: segmentation + lambda * boundary loss.
4. `src/pancreas_seg/data.py`: NIfTI loading, grouped splits, crops and augmentation.
5. `src/pancreas_seg/engine.py`: training and full-volume evaluation.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for tensor dimensions,
gradient paths and the scope of the comparison. No segmentation improvement
or literature novelty is established by the implementation alone.

See [verification/README.md](verification/README.md) for commands to verify
the implementation locally. Generated reports are excluded from version control.
