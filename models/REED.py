
from __future__ import annotations

import math
import random
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn

from layers.Augmentation import get_augmentation
from layers.Transformer_EncDec import EncoderLayer


def _parse_augmentation_policy(policy: str) -> Sequence[str]:
    items = [
        item.strip()
        for item in str(policy).split(",")
        if item.strip()
    ]

    if not items:
        raise ValueError(
            "augmentation_policy must contain at least one augmentation."
        )

    return items


class _EncoderConfigBridge:

    _FIELD_MAP = {
        "n_heads": "attention_heads",
        "t_layer": "temporal_depth",
        "v_layer": "channel_depth",
        "seq_len": "sequence_length",
        "enc_in": "input_channels",
        "augmentations": "augmentation_policy",
        "num_class": "num_classes",
    }

    def __init__(self, config) -> None:
        object.__setattr__(self, "_config", config)

    def __getattr__(self, name):
        config = object.__getattribute__(self, "_config")

        if name in self._FIELD_MAP:
            return getattr(config, self._FIELD_MAP[name])

        return getattr(config, name)


class CurvatureResidualStack(nn.Module):

    def __init__(
        self,
        layers: Iterable[nn.Module],
        *,
        enabled: bool,
        max_scale: float,
        norm: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(list(layers))
        self.norm = norm
        self.enabled = bool(enabled and len(self.layers) >= 2)
        self.max_scale = float(max_scale)

        if self.max_scale <= 0.0:
            raise ValueError(
                f"max_scale must be > 0, got {self.max_scale}."
            )

        if self.enabled:
            self.gate_logits = nn.Parameter(
                torch.zeros(
                    len(self.layers) - 1,
                    dtype=torch.float32,
                )
            )
        else:
            self.register_parameter(
                "gate_logits",
                None,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_raw_update = None

        for layer_index, layer in enumerate(self.layers):
            layer_input = x
            base_output = layer(layer_input)
            raw_update = base_output - layer_input

            if self.enabled and layer_index >= 1:
                curvature = (
                    raw_update
                    - previous_raw_update
                )

                gate = (
                    self.max_scale
                    * torch.tanh(
                        self.gate_logits[layer_index - 1]
                    )
                ).to(
                    dtype=base_output.dtype,
                    device=base_output.device,
                )

                x = (
                    base_output
                    + gate * curvature
                )
            else:
                x = base_output

            # Use the raw layer displacement by definition.
            previous_raw_update = raw_update

        if self.norm is not None:
            x = self.norm(x)

        return x


class SinusoidalAxisEncoding(nn.Module):

    def __init__(
        self,
        width: int,
        max_length: int = 5000,
    ) -> None:
        super().__init__()

        width = int(width)

        if width <= 0:
            raise ValueError(
                f"width must be positive, got {width}."
            )

        encoding = torch.zeros(
            max_length,
            width,
        ).float()

        position = torch.arange(
            0,
            max_length,
        ).float().unsqueeze(1)

        frequency = (
            torch.arange(
                0,
                width,
                2,
            ).float()
            * (
                -(
                    math.log(10000.0)
                    / width
                )
            )
        ).exp()

        encoding[:, 0::2] = torch.sin(
            position * frequency
        )

        cosine_width = encoding[
            :,
            1::2,
        ].shape[1]

        encoding[:, 1::2] = torch.cos(
            position
            * frequency[:cosine_width]
        )

        self.register_buffer(
            "encoding",
            encoding.unsqueeze(0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_count = int(
            x.size(1)
        )

        if token_count > self.encoding.size(1):
            raise ValueError(
                f"Token count {token_count} exceeds "
                f"max_length {self.encoding.size(1)}."
            )

        return self.encoding[
            :,
            :token_count,
        ].to(
            dtype=x.dtype,
            device=x.device,
        )


class CrossChannelPatchEmbedding(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()

        patch_len = int(
            config.patch_len
        )

        if patch_len <= 1:
            raise ValueError(
                "CrossChannelPatchEmbedding requires patch_len > 1."
            )

        self.padding = nn.ReplicationPad1d(
            (
                0,
                patch_len,
            )
        )

        self.projection = nn.Conv2d(
            in_channels=1,
            out_channels=int(config.d_model),
            kernel_size=(
                int(config.input_channels),
                patch_len,
            ),
            stride=(
                1,
                patch_len,
            ),
            padding=0,
            padding_mode="circular",
            bias=False,
        )

        nn.init.kaiming_normal_(
            self.projection.weight,
            mode="fan_in",
            nonlinearity="leaky_relu",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.padding(
            x
        ).unsqueeze(1)

        return self.projection(
            x
        ).squeeze(2).transpose(
            1,
            2,
        )


class ChannelAxisEmbedding(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()

        augmentation_specs = _parse_augmentation_policy(
            config.augmentation_policy
        )

        self.augmentations = nn.ModuleList(
            [
                get_augmentation(spec)
                for spec in augmentation_specs
            ]
        )

        # Keep construction order identical to the validated implementation.
        self.projection = nn.Linear(
            int(config.sequence_length),
            int(config.d_model),
        )

        self.position_encoding = SinusoidalAxisEncoding(
            width=int(config.sequence_length)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(
            1,
            2,
        )

        augmentation_index = random.randint(
            0,
            len(self.augmentations) - 1,
        )

        augmented = self.augmentations[
            augmentation_index
        ](x)

        augmented = (
            augmented
            + self.position_encoding(augmented)
        )

        return self.projection(
            augmented
        )


class TemporalAxisEmbedding(nn.Module):

    def __init__(
        self,
        config,
        *,
        use_delta: bool,
    ) -> None:
        super().__init__()

        self.patch_len = int(
            config.patch_len
        )

        self.use_delta = bool(
            use_delta
        )

        if self.use_delta and self.patch_len != 1:
            raise ValueError(
                "Temporal delta embedding requires patch_len == 1."
            )

        augmentation_specs = _parse_augmentation_policy(
            config.augmentation_policy
        )

        self.augmentations = nn.ModuleList(
            [
                get_augmentation(spec)
                for spec in augmentation_specs
            ]
        )

        if self.patch_len > 1:
            self.base_projection = CrossChannelPatchEmbedding(
                config
            )
        else:
            self.base_projection = nn.Linear(
                int(config.input_channels),
                int(config.d_model),
            )

        self.position_encoding = SinusoidalAxisEncoding(
            width=int(config.sequence_length)
        )

        if self.use_delta:
            cpu_rng_state = (
                torch.random.get_rng_state()
            )

            input_channels = int(
                config.input_channels
            )

            self.delta_projection = nn.Linear(
                input_channels,
                int(config.d_model),
                bias=False,
            )

            torch.random.set_rng_state(
                cpu_rng_state
            )

            nn.init.zeros_(
                self.delta_projection.weight
            )

        else:
            self.delta_projection = None

    @staticmethod
    def first_difference(
        x: torch.Tensor,
    ) -> torch.Tensor:
        delta = torch.zeros_like(
            x
        )

        if x.size(-1) > 1:
            delta[..., 1:] = (
                x[..., 1:]
                - x[..., :-1]
            )

        return delta

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x.transpose(
            1,
            2,
        )

        augmentation_index = random.randint(
            0,
            len(self.augmentations) - 1,
        )

        augmented = self.augmentations[
            augmentation_index
        ](x)

        state = (
            augmented
            + self.position_encoding(augmented)
        )

        if self.patch_len > 1:
            return self.base_projection(
                state
            )

        base = self.base_projection(
            state.transpose(
                1,
                2,
            )
        )

        if not self.use_delta:
            return base

        delta_raw = self.first_difference(
            augmented
        )

        delta_tokens = delta_raw.transpose(
            1,
            2,
        )

        delta = self.delta_projection(
            delta_tokens
        )

        return base + delta


class DecisionSubspacePenalty(nn.Module):

    def __init__(
        self,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.eps = float(
            eps
        )

        if self.eps <= 0.0:
            raise ValueError(
                f"eps must be > 0, got {self.eps}."
            )

    def forward(
        self,
        features: torch.Tensor,
        classifier_weight: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(
                "features must have shape [batch, d_model]."
            )

        work = (
            features.float()
            if features.dtype in (
                torch.float16,
                torch.bfloat16,
            )
            else features
        )

        weight = classifier_weight.detach().to(
            device=work.device,
            dtype=work.dtype,
        )

        class_count = int(
            weight.size(0)
        )

        gram = torch.matmul(
            weight,
            weight.transpose(
                0,
                1,
            ),
        )

        identity = torch.eye(
            class_count,
            device=work.device,
            dtype=work.dtype,
        )

        gram = (
            gram
            + self.eps * identity
        )

        rhs = torch.matmul(
            work,
            weight.transpose(
                0,
                1,
            ),
        ).transpose(
            0,
            1,
        )

        coefficients = torch.linalg.solve(
            gram,
            rhs,
        ).transpose(
            0,
            1,
        )

        parallel = torch.matmul(
            coefficients,
            weight,
        )

        outside = (
            work
            - parallel
        )

        numerator = outside.pow(2).sum(
            dim=1
        )

        denominator = work.pow(2).sum(
            dim=1
        ).clamp_min(
            self.eps
        )

        return (
            numerator
            / denominator
        ).mean()


class Model(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()

        self.channel_depth = int(
            config.channel_depth
        )

        self.temporal_depth = int(
            config.temporal_depth
        )

        self.d_model = int(
            config.d_model
        )

        self.num_classes = int(
            config.num_classes
        )

        if (
            self.channel_depth <= 0
            and self.temporal_depth <= 0
        ):
            raise ValueError(
                "At least one branch depth must be positive."
            )

        self.use_curvature_residual = bool(
            config.enable_curvature_residual
        )

        self.curvature_scale = float(
            config.curvature_scale
        )

        self.use_temporal_delta = bool(
            config.enable_temporal_delta
            and self.temporal_depth > 0
        )

        self.use_subspace_penalty = bool(
            config.enable_subspace_penalty
        )

        layer_config = _EncoderConfigBridge(
            config
        )

        # Preserve module-construction order for reproducibility.
        if self.channel_depth > 0:
            self.channel_branch = nn.Sequential(
                ChannelAxisEmbedding(
                    config
                ),
                CurvatureResidualStack(
                    [
                        EncoderLayer(
                            layer_config
                        )
                        for _ in range(
                            self.channel_depth
                        )
                    ],
                    enabled=self.use_curvature_residual,
                    max_scale=self.curvature_scale,
                ),
            )
        else:
            self.channel_branch = None

        if self.temporal_depth > 0:
            self.temporal_branch = nn.Sequential(
                TemporalAxisEmbedding(
                    config,
                    use_delta=self.use_temporal_delta,
                ),
                CurvatureResidualStack(
                    [
                        EncoderLayer(
                            layer_config
                        )
                        for _ in range(
                            self.temporal_depth
                        )
                    ],
                    enabled=self.use_curvature_residual,
                    max_scale=self.curvature_scale,
                ),
            )
        else:
            self.temporal_branch = None

        self.classifier = nn.Linear(
            self.d_model,
            self.num_classes,
        )

        self.subspace_penalty = DecisionSubspacePenalty(
            eps=float(config.subspace_eps)
        )

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "Expected input with shape [batch, time, channels], "
                f"got {tuple(x.shape)}."
            )

        fused = None

        if self.channel_branch is not None:
            channel_feature = self.channel_branch(
                x
            ).mean(
                dim=1
            )

            fused = channel_feature

        if self.temporal_branch is not None:
            temporal_feature = self.temporal_branch(
                x
            ).mean(
                dim=1
            )

            fused = (
                temporal_feature
                if fused is None
                else fused + temporal_feature
            )

        if fused is None:
            raise RuntimeError(
                "No active model branch."
            )

        return fused

    def classify(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(
            features
        )

    def compute_subspace_penalty(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_subspace_penalty:
            raise RuntimeError(
                "Decision-subspace penalty is disabled."
            )

        return self.subspace_penalty(
            features,
            self.classifier.weight,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        features = self.encode(
            x
        )

        return self.classify(
            features
        )


__all__ = [
    "Model",
    "CurvatureResidualStack",
    "SinusoidalAxisEncoding",
    "CrossChannelPatchEmbedding",
    "ChannelAxisEmbedding",
    "TemporalAxisEmbedding",
    "DecisionSubspacePenalty",
]