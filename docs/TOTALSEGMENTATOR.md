# TotalSegmentator v1: data preparation

**TotalSegmentator v1** was created by **Jakob Wasserthal et al.** at
**University Hospital Basel**. It contains 1,204 CT examinations with
annotations for 104 anatomical structures. The original v1
release is [Zenodo record 6802614](https://zenodo.org/records/6802614),
with the [TotalSegmentator paper](https://arxiv.org/abs/2208.05868).
The dataset is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Retain the
dataset authors, release link and license attribution.

CT images, organ labels and generated cohort files are not committed to
this repository. Download and extract the data separately, then audit
your copy and create its frozen splits before training.

## Expected layout and first setup

The following commands assume a sibling dataset directory; replace
`--root` if your files are elsewhere:

```text
../Totalsegmentator_dataset/
  meta.csv
  s0000/
    ct.nii.gz
    segmentations/pancreas.nii.gz
```

Only CT is a model input. The separate pancreas mask is the binary target;
other organ masks are not input channels.

From the project directory:

```bash
python filter_totalsegmentator_pancreas.py \
  --root ../Totalsegmentator_dataset \
  --output-dir data/totalsegmentator_filter \
  --grid-tolerance-voxels 0.25
```

Read the generated `summary.json`. In `configs/boundary_attention.yaml`,
set `expected_cases` to the retained count from your audit, or null. The
supplied value 795 records the inspected local copy, not a guaranteed
pancreas-positive count for the full release. Keep `pancreas_label_ids: [1]`
only when the binary encoding check passes. Then freeze the official
assignments without loading CT payloads or starting training:

```bash
python scripts/prepare_splits.py --config configs/boundary_attention.yaml
```

The generated manifest and split JSON are local files excluded from
version control. The manifest references the source images and labels;
the audit never moves, modifies or deletes them. It needs NumPy and
NiBabel, plus the repository's NumPy-only geometry helper. Existing reports
are protected; use a fresh output directory or `--overwrite` for a
deliberate re-audit.

## Audit criteria and outputs

The audit distinguishes missing files, unreadable files, invalid geometry,
invalid mask encoding, valid empty masks and retained nonempty masks.
It checks scalar 3D volumes, array dimensions, affine compatibility,
declared units, finite mask values, foreground count and border contact.
Separate pancreas masks must use background 0 and foreground 1. This is
not a global pancreas class ID in a merged multilabel file.

Only CT headers and pancreas-mask voxel arrays are read by the audit.
It does not fully decompress every CT payload; inspect sampled CTs and
image/mask overlays separately.

| Generated file | Purpose |
|---|---|
| `manifest.csv` | Valid, nonempty cases for the primary experiment |
| `valid_manifest.csv` | All valid cases, including empty annotations |
| `audit.csv` | Per-case status, encoding, geometry and coverage flags |
| `summary.json` | Counts, metadata matching and limitations |
| `kept_case_ids.txt` | Retained case IDs |
| `splits_official.json` | Frozen assignments created by `prepare_splits.py` |

An empty mask means no annotated pancreas foreground; it does not prove
anatomical absence. A nonempty mask does not prove full pancreas coverage.
Border-touching cases remain flagged rather than silently excluded.
Primary evaluation is conditional on valid annotated pancreas presence.
Keep the valid-empty cohort for any later absence/false-positive analysis.

## Example audit of the inspected download

One inspected copy contained 1,204 case directories but only 1,203 CT
files: `s0864/ct.nii.gz` was missing. Its audit produced the following
counts under the explicit geometry policy below. These are observations
of that copy, not fixed cohort sizes promised by the release.

| Outcome | Cases |
|---|---:|
| Discovered case directories | 1,204 |
| Valid nonempty masks retained | 795 |
| Valid empty masks | 408 |
| Missing CT (`s0864` in this copy) | 1 |
| Missing masks, unreadable files, invalid encoding or invalid geometry | 0 |
| Valid cases including empty masks | 1,203 |
| Retained masks touching a volume border | 237 |

The 237 border-touching cases are a subset of the 795 retained cases.
All examined mask values were 0/1. Retained official assignments were
709 training, 36 validation and 50 test cases. Regenerate the reports for
your files; generated reports and machine-specific diagnostics are not
included in a fresh clone.

### Small CT/mask header differences

The initial strict audit of that copy flagged 93 affine differences with
matching array shapes, origins and RAS orientation: small CT sform shears
versus orthogonalized mask grids. The maximum voxel-centre corner offset
in a CT-grid component was 0.136772 voxels; the maximum physical corner
distance was 0.212886 mm.

The recipe explicitly uses `grid_tolerance_voxels: 0.25`. The check maps
mask voxel-centre corners into the CT grid and bounds their maximum
absolute component offset. This accepts the measured small differences
while rejecting larger shifts and shape mismatches. The generic default
is strict with tolerance zero; the configured tolerance must stay below
half a voxel.

Every accepted nonidentical mask header uses the CT affine in memory
before canonical orientation, including differences within the strict
affine tolerance. Mask voxel values are preserved without interpolation
or source-file edits. The example audit recorded 217 header adjustments
across valid cases: 124 within the strict affine tolerance and 93 requiring
the explicit quarter-voxel policy. Of the latter, 45 were nonempty and 48
were empty masks. Header adjustment counts are recorded in audit outputs.

## Metadata and frozen partitions

The inspected semicolon-delimited `meta.csv` has columns `image_id`,
`age`, `gender`, `institute`, `study_type` and `split`. Its assignments
before filtering were 1,082 `train`, 57 `val` and 65 `test`. Preserve the
official assignments of retained cases. This release already provides a
validation partition, so no new random subdivision is needed.

There is no `patient_id` column in that metadata. The audit uses `case_id`
as the grouping proxy and records `case_id_proxy`. This does not establish
independence between patients. If a repeated-patient mapping is available,
import it and check for overlap across partitions before training.

The manifest stores assignments as `official_split`. Both variants must
use the same curated manifest and frozen split file. The loader imports
complete train/validation/test assignments exactly and rejects partially
specified, unknown or conflicting labels. Split digests and grouping
checks reject incompatible frozen splits.

For a different release with only official train/test assignments,
validation is derived from official training groups without moving test
cases. A manifest with no official split column, or an entirely blank
column, uses the reproducible grouped random-split fallback.

## Preprocessing and boundary targets

The starting CT window is `[-200, 400]`, followed by per-volume z-score
normalization. This was selected from training-case inspection and is not
a validated optimum. In the inspected copy, retained headers were RAS
with approximately 1.5 mm isotropic spacing and undeclared spatial units.
`target_spacing_mm: null` preserves that downloaded grid;
`spacing_unit_if_unknown: mm` explicitly interprets undeclared units as
millimetres. Check these assumptions on your files.

Augmentation is disabled: no affine/elastic warps, flips or intensity
noise. Patch locations remain random. With 50% probability the centre is
sampled from pancreas foreground; otherwise it is sampled anywhere in the
scan. This is patch sampling rather than an image transformation.

Boundary targets are generated in memory from the binary pancreas mask,
not saved as separate files. The one-voxel halo gives a temporary 98-cubed
patch; a six-connected erosion produces the inner boundary before CT,
mask and boundary are cropped to 96 cubed. This avoids inventing surfaces
where the final crop cuts through the organ. With batch size 1, the CT
input and each target have shape `(1, 1, 96, 96, 96)`. Both encoders
receive only the CT image.

Check representative overlays and annotation coverage before training.
For example, the inspected copy's `s1209` had tiny disconnected foreground
components distant from the main pancreas annotation. No component
removal is applied automatically.

Use [the notebook](../notebooks/train_and_evaluate.ipynb) for data preview,
model selection, training and validation plots. Official validation data
select settings/checkpoints; official test data are reserved for final
evaluation. The [verification instructions](../verification/README.md)
describe synthetic checks and local runtime measurement. Generated
diagnostic JSON/XML files, images and benchmark reports remain local and
are excluded from version control.
