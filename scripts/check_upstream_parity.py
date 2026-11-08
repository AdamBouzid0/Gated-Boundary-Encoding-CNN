"""Optional numerical check against the exact 2018 repository CT model.

Provide a local checkout; this script does not download anything. It executes
only the relevant class definitions from that trusted upstream Python source,
avoiding imports of the repository's obsolete training dependencies.
"""

import argparse
import ast
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from pancreas_seg.models import PancreasUNet3d


def load_classes(path: Path, names: set[str], namespace: dict) -> None:
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if {node.name for node in selected} != names:
        raise ValueError(f"Expected class definitions not found in {path}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    namespace = {"torch": torch, "nn": nn, "F": F, "init_weights": lambda *a, **k: None}
    load_classes(args.upstream_dir / "models/networks/utils.py",
                 {"UnetConv3", "UnetUp3_CT", "UnetGridGatingSignal3", "UnetDsv3"}, namespace)
    load_classes(args.upstream_dir / "models/layers/grid_attention_layer.py",
                 {"_GridAttentionBlockND", "GridAttentionBlock3D"}, namespace)
    load_classes(args.upstream_dir / "models/networks/unet_CT_single_att_dsv_3D.py",
                 {"unet_CT_single_att_dsv_3D", "MultiAttentionBlock"}, namespace)
    torch.manual_seed(17)
    reference = namespace["unet_CT_single_att_dsv_3D"](feature_scale=32, n_classes=2, in_channels=1)
    port = PancreasUNet3d(variant="attention_unet", base_channels=2, norm="batch")
    for i, name in enumerate(("conv1", "conv2", "conv3", "conv4", "center")):
        getattr(reference, name).load_state_dict(port.encoder.blocks[i].state_dict())
    reference.gating.conv1.load_state_dict(port.decoder.gating.state_dict())
    for i, level in enumerate((4, 3, 2)):
        reference_block = getattr(reference, f"attentionblock{level}")
        reference_block.gate_block_1.load_state_dict(port.decoder.attention_blocks[i].gates[0].state_dict())
        reference_block.combine_gates.load_state_dict(port.decoder.attention_blocks[i].combine.state_dict())
    for i, level in enumerate((4, 3, 2, 1)):
        getattr(reference, f"up_concat{level}").conv.load_state_dict(port.decoder.up_blocks[i].conv.state_dict())
        dsv = getattr(reference, f"dsv{level}")
        head = dsv if level == 1 else dsv.dsv[0]
        head.load_state_dict(port.segmentation_head.heads[i].state_dict())
    reference.final.load_state_dict(port.segmentation_head.fusion.state_dict())
    reference.eval()
    port.eval()
    image = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        expected = reference(image)
        actual = port(image)["seg_logits"]
    error = float((expected - actual).abs().max())
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    report = {"status": "passed", "reference": "unet_CT_single_att_dsv_3D",
              "input_shape": list(image.shape), "max_absolute_logit_error": error,
              "note": "Same weights, eval mode, even-sized synthetic input; not a trained-result reproduction"}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

