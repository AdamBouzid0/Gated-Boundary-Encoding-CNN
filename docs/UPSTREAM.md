# Relationship to the 2018 repository

Paper: O. Oktay et al., *Attention U-Net: Learning Where to Look for the
Pancreas*, 2018, <https://arxiv.org/abs/1804.03999>.

Repository: <https://github.com/ozan-oktay/Attention-Gated-Networks>.

Inspected source snapshot: `eee4881fdc31920efd873773e0b744df8dacbfb6`.
This identifies the inspected checkout; it is not a claim that this commit
was the precise snapshot used for the published experiments.

## Source mapping

| Upstream source | This project |
|---|---|
| [`train_segmentation.py`](https://github.com/ozan-oktay/Attention-Gated-Networks/blob/eee4881fdc31920efd873773e0b744df8dacbfb6/train_segmentation.py) | `train_segmentation.py`, `src/pancreas_seg/engine.py` |
| [`unet_CT_single_att_dsv_3D.py`](https://github.com/ozan-oktay/Attention-Gated-Networks/blob/eee4881fdc31920efd873773e0b744df8dacbfb6/models/networks/unet_CT_single_att_dsv_3D.py) | Encoder, decoder, one sub-gate per gated skip, and multiscale output fusion in `models.py` |
| [`grid_attention_layer.py`](https://github.com/ozan-oktay/Attention-Gated-Networks/blob/eee4881fdc31920efd873773e0b744df8dacbfb6/models/layers/grid_attention_layer.py) | `GridAttention3d` |
| [`models/networks/utils.py`](https://github.com/ozan-oktay/Attention-Gated-Networks/blob/eee4881fdc31920efd873773e0b744df8dacbfb6/models/networks/utils.py) | `ConvBlock3d`, `UpBlock3d`, gating projection and DSV heads |

The architecture is a modern port, not an installation of the old training
framework. The reference uses legacy PyTorch APIs and dependencies such as
`torchsample`. The port uses current tensor/autocast APIs and MONAI for
optional paired augmentation and sliding-window inference. Augmentation
is disabled in the default configuration. Upstream MIT attribution is retained in
`third_party/LICENSE_Attention-Gated-Networks`.

## Architectural details deliberately retained

- The **CT** variant has five feature resolutions, four 2x2x2 pools, and
  3x3x3 convolution blocks. The separate generic `unet_grid_attention_3D`
  file uses anisotropic operations; it is not the template used here.
- Default widths are `[16, 32, 64, 128, 256]`, equivalent to upstream
  `feature_scale=4`.
- Three skips are gated. The finest skip is not gated.
- The first gate uses a projected bottleneck; subsequent gates use the
  preceding decoder features. This is the CT model's sequential gating,
  not the generic model's shared bottleneck signal at every gate.
- `theta(x)` uses a kernel and stride equal to `attention_subsample` in the
  **reference code**. With the default 2x2x2 factor it is therefore a
  learned spatial reduction. The paper's 1x1x1 schematic is simplified.
- `phi(g)` is interpolated to the `theta(x)` grid before addition. ReLU,
  a 1x1x1 scalar score, sigmoid, coefficient interpolation, multiplication
  with the original x, output projection, and gate fusion follow.
- Four decoder logit heads are upsampled, concatenated finest-first, and
  fused by a final 1x1x1 layer. This follows the reference DSV output
  implementation; it does not add four separate auxiliary losses.
- `attention_heads: 2` enables two parallel sub-gates per skip, following
  the repository's multi-attention variant.

## Deliberate changes

- Binary pancreas/background output with two logits; pancreas source label
  IDs are configured rather than assuming they are always 1.
- Explicit-size `interpolate(..., align_corners=False)` handles spatial
  sizes more robustly. The decoder uses interpolation rather than a
  transposed convolution, matching the CT reference's decoder.
- An extra narrow encoder receives the image. Its bottleneck features are
  concatenated with the main bottleneck and projected to the original
  main width. A separate narrow decoder predicts boundaries using only
  the boundary stream, before main-stream fusion.
- Frozen splits with patient-group validation when identifiers are
  supplied, validation-only model selection, and an explicit final-test
  command. TotalSegmentator's official partitions are preserved; case IDs
  are grouping proxies where patient identities are unavailable. The old
  script evaluates its test loader every epoch; this training loop uses
  validation for model selection.
- Foreground-aware training crops, image/mask geometry validation,
  optional paired spatial augmentation, and halo-protected boundary
  targets. Augmentation is disabled in the default recipe.
- Batch normalization is the default for reference fidelity; optional
  group normalization is an explicit change useful to investigate with
  very small batches.
- Initialization and the training recipe are not claimed to reproduce
  published results. The losses are an explicit soft Dice objective and a weighted
  BCE boundary objective. Changing learning settings requires validation.

## Check numerical architecture parity

With an existing local checkout at the recorded revision:

```bash
python scripts/check_upstream_parity.py --upstream-dir /path/to/Attention-Gated-Networks
```

This maps identical weights into the upstream CT single-attention model
and this port, then compares logits on a small even-sized 3D input in
evaluation mode. It checks the port's computation, not training convergence
or segmentation quality. It executes relevant definitions from the local
checkout, so use source you trust.

