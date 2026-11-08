# Verification

Run these commands from the repository root after installing the project
and its development dependencies. Generated reports, checkpoints and
diagnostic images are local artifacts and are not committed. Keep results
under `runs/` or the ignored report files in `verification/`.

Run the automated tests:

```bash
python -m pytest -q
```

The tests use synthetic fixtures to check model shapes and gradient paths,
boundary targets, geometry handling, frozen splits and the bounded runtime
benchmark. They do not require the downloaded CT cohort.

Exercise training, checkpoint reload/resume, full-volume inference and
prediction export on small synthetic volumes:

```bash
python scripts/smoke_test.py --report runs/verification/smoke_report.json
```

This runs on CPU with reduced model widths. Its scores check the pipeline;
they do not measure real pancreas-segmentation quality.

After curating the real dataset and freezing its splits as described in
[the dataset guide](../docs/TOTALSEGMENTATOR.md), measure 20 training-loader
batches per model:

```bash
python scripts/benchmark_training.py --config configs/boundary_attention.yaml \
  --device cuda --batches 20 \
  --variants unet attention_unet boundary_attention_unet \
  --output-dir runs/runtime_smoke_check
```

Use a fresh output directory for each measurement; existing reports are
preserved. The command writes `report.json` and `batches.csv`. With batch
size 1 and accumulation 2, 20 batches normally produce 10 optimizer
updates. The report records actual updates, skipped AMP updates, input-patch
hashes, loading/compute times and GPU memory peaks. It loads training cases
only and saves no production checkpoints. Startup, execution order, caches
and case sizes affect a short measurement; training-time projections
exclude full-volume validation, checkpoint writes and final evaluation.

Syntax-check the notebook without executing its cells:

```bash
python - <<'PY'
import json
from pathlib import Path

notebook = json.loads(Path("notebooks/train_and_evaluate.ipynb").read_text())
for index, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] == "code":
        compile("".join(cell["source"]), f"notebook_cell_{index}", "exec")
print("Notebook code compiles.")
PY
```

The notebook's benchmark, training, validation and final-test switches
default to `False`. It can display a local saved benchmark report when one
exists; published source does not require a committed runtime report.

Optionally compare the Attention U-Net port against the pinned upstream
source described in [UPSTREAM.md](../docs/UPSTREAM.md):

```bash
python scripts/check_upstream_parity.py \
  --upstream-dir /path/to/Attention-Gated-Networks \
  --report runs/verification/upstream_parity.json
```

This checks logits with matched weights on a small synthetic input in
evaluation mode. It does not reproduce published training results or
establish an improvement over the baseline.
