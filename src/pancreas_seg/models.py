"""Modern port of Oktay et al.'s CT single-attention / DSV architecture.

Reference: Attention-Gated-Networks, commit
eee4881fdc31920efd873773e0b744df8dacbfb6. See docs/UPSTREAM.md and
third_party/LICENSE_Attention-Gated-Networks. Tensor layout is N,C,D,H,W;
the physical meaning of spatial axes is carried by the NIfTI affine.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def normalization(channels: int, kind: str) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm3d(channels)
    if kind == "group":
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unknown normalization: {kind!r}")


class ConvBlock3d(nn.Module):
    """Two 3x3x3 convolutions, normalization and ReLU, as in the CT model."""

    def __init__(self, in_channels: int, out_channels: int, norm: str = "batch"):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            normalization(out_channels, norm),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            normalization(out_channels, norm),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv2(self.conv1(x))


class Encoder3d(nn.Module):
    """Five resolutions, four 2x2x2 pooling operations."""

    def __init__(self, in_channels: int, widths: Sequence[int], norm: str):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvBlock3d(in_channels if i == 0 else widths[i - 1], width, norm)
            for i, width in enumerate(widths)
        ])
        self.pool = nn.MaxPool3d(2)

    def forward(self, x: Tensor) -> list[Tensor]:
        features = []
        for i, block in enumerate(self.blocks):
            if i:
                x = self.pool(x)
            x = block(x)
            features.append(x)
        return features


class GridAttention3d(nn.Module):
    """Port of GridAttentionBlock3D's sigmoid/additive attention path.

    IMPORTANT: the reference CODE uses a strided subsampling kernel for theta,
    not just the 1x1x1 projection drawn in the paper's schematic. phi(g) is
    interpolated onto theta(x)'s grid before addition. The original x bypass
    is multiplied by interpolated alpha, then passes through W.
    """

    def __init__(self, in_channels: int, gate_channels: int, inter_channels: int,
                 subsample: Sequence[int] = (2, 2, 2), norm: str = "batch"):
        super().__init__()
        subsample = tuple(subsample)
        self.theta = nn.Conv3d(in_channels, inter_channels, subsample,
                               stride=subsample, bias=False)
        self.phi = nn.Conv3d(gate_channels, inter_channels, 1, bias=True)
        self.psi = nn.Conv3d(inter_channels, 1, 1, bias=True)
        self.W = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 1),
            normalization(in_channels, norm),
        )

    def forward(self, x: Tensor, g: Tensor) -> tuple[Tensor, Tensor]:
        projected_x = self.theta(x)
        projected_g = F.interpolate(self.phi(g), size=projected_x.shape[2:],
                                     mode="trilinear", align_corners=False)
        combined = F.relu(projected_x + projected_g, inplace=True)
        coarse_alpha = torch.sigmoid(self.psi(combined))
        alpha = F.interpolate(coarse_alpha, size=x.shape[2:],
                              mode="trilinear", align_corners=False)
        return self.W(x * alpha), alpha  # broadcast the one-channel alpha


class AttentionBlock3d(nn.Module):
    """One or two sub-gates followed by the reference 1x1x1 fusion."""

    def __init__(self, in_channels: int, gate_channels: int, heads: int,
                 subsample: Sequence[int], norm: str):
        super().__init__()
        self.gates = nn.ModuleList([
            GridAttention3d(in_channels, gate_channels, in_channels, subsample, norm)
            for _ in range(heads)
        ])
        self.combine = nn.Sequential(
            nn.Conv3d(in_channels * heads, in_channels, 1),
            normalization(in_channels, norm), nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor, g: Tensor) -> tuple[Tensor, Tensor]:
        pairs = [gate(x, g) for gate in self.gates]
        features = torch.cat([pair[0] for pair in pairs], dim=1)
        maps = torch.cat([pair[1] for pair in pairs], dim=1)
        return self.combine(features), maps


class UpBlock3d(nn.Module):
    """Interpolate decoder features, concatenate the skip, then convolve."""

    def __init__(self, decoder_channels: int, skip_channels: int, norm: str):
        super().__init__()
        self.conv = ConvBlock3d(decoder_channels + skip_channels, skip_channels, norm)

    def forward(self, skip: Tensor, decoder: Tensor) -> Tensor:
        decoder = F.interpolate(decoder, size=skip.shape[2:],
                                mode="trilinear", align_corners=False)
        return self.conv(torch.cat([skip, decoder], dim=1))


class Decoder3d(nn.Module):
    def __init__(self, widths: Sequence[int], norm: str, attention: bool,
                 heads: int = 1, subsample: Sequence[int] = (2, 2, 2)):
        super().__init__()
        self.attention = attention
        self.skip_levels = (3, 2, 1, 0)
        self.up_blocks = nn.ModuleList([
            UpBlock3d(widths[level + 1], widths[level], norm)
            for level in self.skip_levels
        ])
        if attention:
            self.gating = nn.Sequential(
                nn.Conv3d(widths[-1], widths[-1], 1),
                normalization(widths[-1], norm), nn.ReLU(inplace=True),
            )
            # The finest skip is ungated, matching the 2018 CT implementation.
            self.attention_blocks = nn.ModuleList([
                AttentionBlock3d(widths[level], widths[level + 1], heads, subsample, norm)
                for level in self.skip_levels[:-1]
            ])

    def forward(self, features: list[Tensor], center: Tensor
                ) -> tuple[list[Tensor], list[Tensor]]:
        current = center
        decoded, maps = [], []
        for i, (level, up) in enumerate(zip(self.skip_levels, self.up_blocks)):
            skip = features[level]
            if self.attention and level > 0:
                g = self.gating(center) if i == 0 else current
                skip, alpha = self.attention_blocks[i](skip, g)
                maps.append(alpha)
            current = up(skip, current)
            decoded.append(current)
        return decoded, maps  # coarse decoder stage -> finest decoder stage


class MultiscaleOutput3d(nn.Module):
    """Port of the CT repository's multiscale DSV output fusion.

    Each decoder resolution has a class-logit head. Upsampled logits are
    concatenated finest-first and fused by a final 1x1x1 convolution.
    """

    def __init__(self, widths: Sequence[int], out_channels: int, multiscale: bool):
        super().__init__()
        self.multiscale = multiscale
        channels = list(reversed(widths[:-1]))
        if multiscale:
            self.heads = nn.ModuleList([nn.Conv3d(c, out_channels, 1) for c in channels])
            self.fusion = nn.Conv3d(4 * out_channels, out_channels, 1)
        else:
            self.head = nn.Conv3d(widths[0], out_channels, 1)

    def forward(self, decoded: list[Tensor]) -> Tensor:
        if not self.multiscale:
            return self.head(decoded[-1])
        size = decoded[-1].shape[2:]
        logits = [F.interpolate(head(feature), size=size, mode="trilinear",
                                align_corners=False)
                  for head, feature in zip(self.heads, decoded)]
        return self.fusion(torch.cat(list(reversed(logits)), dim=1))


class PancreasUNet3d(nn.Module):
    """Shared baseline and extension; the boundary branch only sees the image.

    variant='unet': ungated control.
    variant='attention_unet': 2018-style CT attention / DSV model.
    variant='boundary_attention_unet': add a narrow encoder, bottleneck fusion,
    and a boundary decoder supervised BEFORE fusion with the main stream.
    """

    def __init__(self, variant: str = "boundary_attention_unet", in_channels: int = 1,
                 base_channels: int = 16, boundary_base_channels: int = 8,
                 norm: str = "batch", attention_heads: int = 1,
                 attention_subsample: Sequence[int] = (2, 2, 2),
                 multiscale_output: bool = True):
        super().__init__()
        if variant not in {"unet", "attention_unet", "boundary_attention_unet"}:
            raise ValueError(f"Unknown model variant: {variant}")
        if attention_heads not in (1, 2):
            raise ValueError("attention_heads must be 1 or 2")
        self.variant = variant
        self.has_boundary = variant == "boundary_attention_unet"
        widths = [base_channels * 2**i for i in range(5)]
        self.encoder = Encoder3d(in_channels, widths, norm)
        self.decoder = Decoder3d(widths, norm, variant != "unet", attention_heads,
                                 attention_subsample)
        self.segmentation_head = MultiscaleOutput3d(widths, 2, multiscale_output)
        if self.has_boundary:
            boundary_widths = [boundary_base_channels * 2**i for i in range(5)]
            self.boundary_encoder = Encoder3d(in_channels, boundary_widths, norm)
            self.bottleneck_fusion = nn.Sequential(
                nn.Conv3d(widths[-1] + boundary_widths[-1], widths[-1], 1),
                normalization(widths[-1], norm), nn.ReLU(inplace=True),
            )
            # This head has NO access to main-stream features: its loss directly
            # supervises boundary_encoder, including its deepest layers.
            self.boundary_decoder = Decoder3d(boundary_widths, norm, attention=False)
            self.boundary_head = nn.Conv3d(boundary_widths[0], 1, 1)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm3d, nn.GroupNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, image: Tensor, compute_boundary: bool = True,
                return_attention: bool = False) -> dict:
        if image.ndim != 5 or min(image.shape[2:]) < 32:
            raise ValueError("Expected (N,C,D,H,W) with each spatial dimension >=32")
        main_features = self.encoder(image)
        center = main_features[-1]
        boundary_logits = None
        if self.has_boundary:
            boundary_features = self.boundary_encoder(image)
            center = self.bottleneck_fusion(torch.cat([center, boundary_features[-1]], dim=1))
            if compute_boundary:
                boundary_decoded, _ = self.boundary_decoder(boundary_features, boundary_features[-1])
                boundary_logits = self.boundary_head(boundary_decoded[-1])
        decoded, attention = self.decoder(main_features, center)
        output = {"seg_logits": self.segmentation_head(decoded),
                  "boundary_logits": boundary_logits}
        if return_attention:
            output["attention_maps"] = attention
        return output


def build_model(config: dict) -> PancreasUNet3d:
    return PancreasUNet3d(**config["model"])


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

