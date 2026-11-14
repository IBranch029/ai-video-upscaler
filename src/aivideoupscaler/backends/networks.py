"""Network architectures for the official Real-ESRGAN checkpoints.

Two architectures are needed:

``RRDBNet``
    Used by ``RealESRGAN_x4plus``, ``RealESRGAN_x2plus`` and
    ``RealESRGAN_x4plus_anime_6B``.  The generator is a stack of
    residual-in-residual dense blocks with five convolutions each.

``SRVGGNetCompact``
    Used by ``realesr-animevideov3``: a compact VGG-style network with PReLU
    activations and a pixel-shuffle upsampler.

The layer names and parameter shapes match the reference implementation
(``basicsr.archs``) exactly, so the released ``.pth`` files load with
``strict=True``.  :func:`build_network` inspects a checkpoint and picks the
right architecture automatically, so no model-specific configuration is needed.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


def pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Space-to-depth transform used by the x1/x2 ESRGAN variants."""
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    return (
        x.view(b, c, hh // scale, scale, hw // scale, scale)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(b, out_channel, hh // scale, hw // scale)
    )


class ResidualDenseBlock(nn.Module):
    """Five-convolution residual dense block (ESRGAN / Real-ESRGAN)."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the dense block and add the scaled residual.

        Every convolution receives the concatenation of the block input and
        all previously computed features, which is what makes the block dense.
        """
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-residual dense block."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the three dense blocks and add the scaled residual."""
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """The RRDB generator used by the ``RealESRGAN_x*plus`` family."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upscale an NCHW float tensor."""
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x

        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class SRVGGNetCompact(nn.Module):
    """The compact VGG-style generator used by ``realesr-animevideov3``."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 16,
        upscale: int = 4,
        act_type: str = "prelu",
    ) -> None:
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(
            nn.PReLU(num_parameters=num_feat) if act_type == "prelu" else nn.ReLU(inplace=True)
        )
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(
                nn.PReLU(num_parameters=num_feat) if act_type == "prelu" else nn.ReLU(inplace=True)
            )
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upscale an NCHW float tensor and add the nearest-neighbour base."""
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


def unwrap_state_dict(state: Any) -> dict[str, Any]:
    """Extract the raw parameter dict from a checkpoint.

    The official releases wrap the parameters in ``params``, ``params_ema`` or
    ``state_dict`` containers, and some are prefixed with ``module.``.
    """
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(state)!r}")

    for key in ("params_ema", "params", "state_dict", "model"):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break

    cleaned: dict[str, Any] = {}
    for key, value in state.items():
        name = key
        if name.startswith("module."):
            name = name[len("module.") :]
        cleaned[name] = value
    return cleaned


def describe_state_dict(state: dict[str, Any]) -> dict[str, int]:
    """Derive the architecture hyper-parameters from a checkpoint's keys."""
    keys = list(state.keys())

    if any(key.startswith("conv_first.") for key in keys):
        # RRDBNet family: conv_first + body.N.rdb{1,2,3}.conv{1..5}
        num_block = len({key.split(".")[1] for key in keys if key.startswith("body.")})
        num_feat = int(state["conv_first.weight"].shape[0])
        num_grow_ch = int(state["body.0.rdb1.conv1.weight"].shape[0])
        # The x2 variant consumes 4x the input channels via pixel-unshuffle.
        in_ch = int(state["conv_first.weight"].shape[1])
        scale = {3: 4, 12: 2, 48: 1}.get(in_ch, 4)
        return {
            "kind": 0,
            "num_block": max(1, num_block),
            "num_feat": num_feat,
            "num_grow_ch": num_grow_ch,
            "scale": scale,
        }

    if "body.0.weight" in keys:
        # SRVGGNetCompact family: body.N.weight / body.N.bias
        conv_indices = sorted(
            {
                int(key.split(".")[1])
                for key in keys
                if key.startswith("body.") and key.endswith(".weight") and key.count(".") == 2
            }
        )
        num_feat = int(state["body.0.weight"].shape[0])
        last = state[f"body.{conv_indices[-1]}.weight"]
        # The last convolution produces num_out_ch * upscale ** 2 channels.
        scale = round((int(last.shape[0]) / 3) ** 0.5)
        num_conv = max(0, (len(conv_indices) - 2) // 2)
        return {
            "kind": 1,
            "num_conv": num_conv,
            "num_feat": num_feat,
            "scale": scale if scale > 0 else 4,
        }

    raise ValueError("Unrecognised Real-ESRGAN checkpoint layout.")


def build_network(state: dict[str, Any]) -> tuple[nn.Module, int]:
    """Build the network matching ``state`` and return it with its scale."""
    info = describe_state_dict(state)

    if info["kind"] == 0:
        net = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            scale=int(info["scale"]),
            num_feat=int(info["num_feat"]),
            num_block=int(info["num_block"]),
            num_grow_ch=int(info["num_grow_ch"]),
        )
    else:
        net = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=int(info["num_feat"]),
            num_conv=int(info["num_conv"]),
            upscale=int(info["scale"]),
        )

    net.load_state_dict(state, strict=True)
    return net, int(info["scale"])
