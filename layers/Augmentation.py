

from __future__ import annotations

import torch
import torch.nn as nn


class Jitter(nn.Module):

    def __init__(self, scale: float = 0.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x += torch.randn_like(x) * self.scale

        return x


class Scale(nn.Module):

    def __init__(self, scale: float = 0.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_size, channels, _ = x.shape

            x *= (
                1
                + torch.randn(
                    batch_size,
                    channels,
                    1,
                    device=x.device,
                )
                * self.scale
            )

        return x


class Flip(nn.Module):

    def __init__(self, prob: float = 0.0):
        super().__init__()
        self.prob = prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and torch.rand(1) < self.prob:
            return torch.flip(
                x,
                [-1],
            )

        return x


class TemporalMask(nn.Module):

    def __init__(self, ratio: float = 0.0):
        super().__init__()
        self.ratio = ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            _, _, time_steps = x.shape

            num_mask = int(
                time_steps * self.ratio
            )

            mask_indices = torch.randperm(
                time_steps
            )[:num_mask]

            x[:, :, mask_indices] = 0

        return x


class ChannelMask(nn.Module):

    def __init__(self, ratio: float = 0.0):
        super().__init__()
        self.ratio = ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            _, channels, _ = x.shape

            num_mask = int(
                channels * self.ratio
            )

            mask_indices = torch.randperm(
                channels
            )[:num_mask]

            x[:, mask_indices, :] = 0

        return x


class FrequencyMask(nn.Module):

    def __init__(self, ratio: float = 0.0):
        super().__init__()
        self.ratio = ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            _, _, time_steps = x.shape

            x_fft = torch.fft.rfft(
                x,
                dim=-1,
            )

            mask = (
                torch.rand(
                    x_fft.shape,
                    device=x.device,
                )
                > self.ratio
            )

            x_fft = x_fft * mask

            x = torch.fft.irfft(
                x_fft,
                n=time_steps,
                dim=-1,
            )

        return x


def _get_value(
    augmentation: str,
    prefix: str,
    default: float,
) -> float:

    suffix = augmentation[len(prefix):]

    if suffix == "":
        return default

    return float(suffix)


def get_augmentation(augmentation: str) -> nn.Module:

    if augmentation.startswith("jitter"):
        return Jitter(
            _get_value(
                augmentation,
                "jitter",
                0.0,
            )
        )

    if augmentation.startswith("scale"):
        return Scale(
            _get_value(
                augmentation,
                "scale",
                0.0,
            )
        )

    if augmentation.startswith("drop"):
        return nn.Dropout(
            _get_value(
                augmentation,
                "drop",
                0.1,
            )
        )

    if augmentation.startswith("flip"):
        return Flip(
            _get_value(
                augmentation,
                "flip",
                0.0,
            )
        )

    if augmentation.startswith("frequency"):
        return FrequencyMask(
            _get_value(
                augmentation,
                "frequency",
                0.0,
            )
        )

    if augmentation.startswith("mask"):
        return TemporalMask(
            _get_value(
                augmentation,
                "mask",
                0.0,
            )
        )

    if augmentation.startswith("channel"):
        return ChannelMask(
            _get_value(
                augmentation,
                "channel",
                0.0,
            )
        )

    if augmentation == "none":
        return nn.Identity()

    raise ValueError(
        f"Unknown augmentation {augmentation}"
    )


__all__ = [
    "Jitter",
    "Scale",
    "Flip",
    "TemporalMask",
    "ChannelMask",
    "FrequencyMask",
    "get_augmentation",
]
