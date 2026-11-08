"""Differentiable segmentation and auxiliary boundary objectives."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SegmentationLoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 0.0,
                 include_background: bool = True):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.include_background = include_background

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        target = target[:, 0].long()
        probabilities = torch.softmax(logits.float(), dim=1)
        one_hot = F.one_hot(target, num_classes=2).movedim(-1, 1).float()
        if not self.include_background:
            probabilities, one_hot = probabilities[:, 1:], one_hot[:, 1:]
        axes = tuple(range(2, probabilities.ndim))
        intersection = (probabilities * one_hot).sum(axes)
        denominator = probabilities.sum(axes) + one_hot.sum(axes)
        dice_loss = 1 - ((2 * intersection + 1e-5) / (denominator + 1e-5)).mean()
        return self.dice_weight * dice_loss + self.ce_weight * F.cross_entropy(logits.float(), target)


class JointLoss(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.segmentation = SegmentationLoss(
            config["dice_weight"], config["ce_weight"], config["include_background"])
        self.boundary_weight = config["boundary_weight"]
        self.register_buffer("edge_pos_weight", torch.tensor(float(config["boundary_pos_weight"])))

    def forward(self, output: dict, label: Tensor, boundary: Tensor) -> tuple[Tensor, dict]:
        seg = self.segmentation(output["seg_logits"], label)
        edge_logits = output["boundary_logits"]
        edge = seg.new_zeros(()) if edge_logits is None else F.binary_cross_entropy_with_logits(
            edge_logits.float(), boundary.float(), pos_weight=self.edge_pos_weight)
        total = seg + self.boundary_weight * edge
        return total, {"loss": total.detach(), "seg_loss": seg.detach(), "boundary_loss": edge.detach()}

