

from __future__ import annotations

import argparse
import os
import random
from typing import Iterable

import numpy as np
import psutil
import torch

from exp.exp_classification import ClassificationExperiment


METRIC_NAMES = (
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "auroc_ovr",
    "auprc_macro",
)


def _float_tag(value: float) -> str:

    return (
        f"{float(value):g}"
        .replace("-", "m")
        .replace(".", "p")
        .replace("+", "")
    )


def _install_framework_fields(args) -> None:

    args.data = args.dataset
    args.root_path = args.data_dir
    args.data_path = args.data_file

    args.seq_len = args.sequence_length
    args.enc_in = args.input_channels

    args.n_heads = args.attention_heads
    args.t_layer = args.temporal_depth
    args.v_layer = args.channel_depth
    args.augmentations = args.augmentation_policy

    args.num_workers = args.workers
    args.itr = args.runs
    args.train_epochs = args.epochs
    args.learning_rate = args.lr
    args.lradj = args.lr_schedule
    args.use_amp = args.amp
    args.loss = "CE"

    args.val_only = args.validation_only
    args.use_multi_gpu = args.multi_gpu
    args.gpu_idx = [args.gpu]


def _configure_device(args) -> None:

    args.use_gpu = bool(
        torch.cuda.is_available()
        and not args.cpu
    )

    if not args.use_gpu:
        args.device_ids = []
        return

    if args.use_multi_gpu:
        device_ids = [
            int(item)
            for item in str(args.devices)
            .replace(" ", "")
            .split(",")
            if item
        ]

        if not device_ids:
            raise ValueError(
                "--devices must contain at least one CUDA index "
                "when --multi-gpu is enabled."
            )

        args.device_ids = device_ids
        args.gpu = device_ids[0]

    else:
        args.device_ids = [
            int(args.gpu)
        ]


def _set_cpu_affinity(
    gpu_indices: Iterable[int],
    cpus_per_gpu: int,
) -> None:

    cpu_indices = []

    for gpu_index in gpu_indices:
        start = (
            int(gpu_index)
            * int(cpus_per_gpu)
        )

        stop = (
            start
            + int(cpus_per_gpu)
        )

        cpu_indices.extend(
            range(
                start,
                stop,
            )
        )

    try:
        psutil.Process().cpu_affinity(
            cpu_indices
        )

        print(
            f"CPU affinity set to {len(cpu_indices)} logical CPUs."
        )

    except Exception as exc:
        print(
            "CPU affinity was not changed: "
            f"{exc}"
        )


def _seed_everything(seed: int) -> None:

    random.seed(
        seed
    )

    os.environ[
        "PYTHONHASHSEED"
    ] = str(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed(
            seed
        )

        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _temporal_delta_active(args) -> bool:
    return bool(
        args.enable_temporal_delta
        and int(args.temporal_depth) > 0
        and int(args.patch_len) == 1
    )


def _curvature_active(args) -> bool:
    return bool(
        args.enable_curvature_residual
        and (
            int(args.channel_depth) >= 2
            or int(args.temporal_depth) >= 2
        )
    )


def _architecture_name(args) -> str:
    components = [
        "REED"
    ]

    if _temporal_delta_active(
        args
    ):
        components.append(
            "TemporalDelta"
        )

    if _curvature_active(
        args
    ):
        components.append(
            "CurvatureResidual"
        )

    if args.enable_subspace_penalty:
        components.append(
            "DecisionSubspace"
        )

    return " + ".join(
        components
    )


def _print_run_header(args) -> None:
    print()
    print(
        "=" * 88
    )

    print(
        f"Architecture: {_architecture_name(args)}"
    )

    print(
        f"Dataset: {args.dataset}"
    )

    print(
        f"d_model: {args.d_model}"
    )

    print(
        f"patch_len: {args.patch_len}"
    )

    print(
        f"Temporal depth: {args.temporal_depth}"
    )

    print(
        f"Channel depth: {args.channel_depth}"
    )

    print(
        "Temporal delta: "
        f"{'enabled' if _temporal_delta_active(args) else 'disabled'}"
    )

    print(
        "Curvature residual: "
        f"{'enabled' if _curvature_active(args) else 'disabled'}"
    )

    if args.enable_curvature_residual:
        print(
            f"Curvature scale: {args.curvature_scale}"
        )

    print(
        "Decision-subspace penalty: "
        f"{'enabled' if args.enable_subspace_penalty else 'disabled'}"
    )

    if args.enable_subspace_penalty:
        print(
            f"Subspace weight: {args.subspace_weight}"
        )

        print(
            f"Subspace warmup: {args.subspace_warmup}"
        )

        print(
            f"Subspace epsilon: {args.subspace_eps}"
        )

        print(
            "Subspace classifier geometry: detached"
        )

        print(
            "Subspace inference overhead: zero"
        )

    protocol = (
        "validation-only; test is never accessed"
        if args.validation_only
        else "validation-selected checkpoint; test evaluated once"
    )

    print(
        f"Evaluation protocol: {protocol}"
    )

    if args.use_gpu:
        try:
            device_name = torch.cuda.get_device_name(
                args.gpu
            )
        except Exception:
            device_name = "unknown"

        print(
            f"Device: cuda:{args.gpu} ({device_name})"
        )
    else:
        print(
            "Device: CPU"
        )

    print(
        "=" * 88
    )
    print()


def _validate_args(args) -> None:
    if (
        int(args.temporal_depth) <= 0
        and int(args.channel_depth) <= 0
    ):
        raise ValueError(
            "At least one branch depth must be positive."
        )

    if int(args.d_model) <= 0:
        raise ValueError(
            "--d_model must be > 0."
        )

    if int(args.patch_len) <= 0:
        raise ValueError(
            "--patch_len must be > 0."
        )

    if int(args.attention_heads) <= 0:
        raise ValueError(
            "--attention-heads must be > 0."
        )

    if (
        args.enable_temporal_delta
        and int(args.temporal_depth) <= 0
    ):
        raise ValueError(
            "--enable-temporal-delta requires "
            "--temporal-depth > 0."
        )

    if (
        args.enable_temporal_delta
        and int(args.patch_len) != 1
    ):
        raise ValueError(
            "--enable-temporal-delta requires "
            "--patch_len 1."
        )

    if (
        args.enable_curvature_residual
        and float(args.curvature_scale) <= 0.0
    ):
        raise ValueError(
            "--curvature-scale must be > 0."
        )

    if float(args.subspace_weight) < 0.0:
        raise ValueError(
            "--subspace-weight must be >= 0."
        )

    if int(args.subspace_warmup) < 0:
        raise ValueError(
            "--subspace-warmup must be >= 0."
        )

    if float(args.subspace_eps) <= 0.0:
        raise ValueError(
            "--subspace-eps must be > 0."
        )

    if int(args.runs) <= 0:
        raise ValueError(
            "--runs must be > 0."
        )

    if int(args.epochs) <= 0:
        raise ValueError(
            "--epochs must be > 0."
        )

    if int(args.batch_size) <= 0:
        raise ValueError(
            "--batch-size must be > 0."
        )

    if int(args.patience) < 0:
        raise ValueError(
            "--patience must be >= 0."
        )

    if float(args.lr) <= 0.0:
        raise ValueError(
            "--lr must be > 0."
        )


def _build_run_name(
    args,
    seed: int,
) -> str:
    name = (
        f"{args.model}_{args.dataset}"
        f"_seed{seed}"
        f"_dm{args.d_model}"
        f"_dp{_float_tag(args.dropout)}"
        f"_td{args.temporal_depth}"
        f"_cd{args.channel_depth}"
        f"_bs{args.batch_size}"
        f"_lr{_float_tag(args.lr)}"
        f"_pl{args.patch_len}"
    )

    if _temporal_delta_active(
        args
    ):
        name += "_delta"

    if _curvature_active(
        args
    ):
        name += (
            "_curve"
            f"{_float_tag(args.curvature_scale)}"
        )

    if args.enable_subspace_penalty:
        name += (
            "_subspace"
            f"_w{_float_tag(args.subspace_weight)}"
            f"_wu{args.subspace_warmup}"
        )

    if args.validation_only:
        name += "_valonly"

    return name


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "REED classification with temporal-difference embedding, "
            "curvature residual correction, and decision-subspace regularization."
        )
    )


    parser.add_argument(
        "--model",
        type=str,
        default="REED",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--data-file",
        type=str,
        default="",
    )


    parser.add_argument(
        "--sequence-length",
        type=int,
        default=96,
        help=(
            "Placeholder value. TRAIN data overwrites it "
            "before model construction."
        ),
    )

    parser.add_argument(
        "--input-channels",
        type=int,
        default=7,
        help=(
            "Placeholder value. TRAIN data overwrites it "
            "before model construction."
        ),
    )

    parser.add_argument(
        "--d_model",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--patch_len",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--attention-heads",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--temporal-depth",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--channel-depth",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--enable-temporal-delta",
        action="store_true",
        help=(
            "Enable the zero-start first-difference "
            "temporal projection."
        ),
    )

    parser.add_argument(
        "--enable-curvature-residual",
        action="store_true",
        help=(
            "Enable bounded second-order residual "
            "correction across depth."
        ),
    )

    parser.add_argument(
        "--curvature-scale",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--enable-subspace-penalty",
        action="store_true",
        help=(
            "Enable the training-only decision-subspace "
            "feature penalty."
        ),
    )

    parser.add_argument(
        "--subspace-weight",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--subspace-warmup",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--subspace-eps",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--augmentation-policy",
        type=str,
        default=(
            "flip0.,"
            "frequency0.,"
            "jitter0.,"
            "mask0.,"
            "channel0.,"
            "drop0."
        ),
    )


    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "First seed. Repeated runs use "
            "seed, seed+1, ..."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="constant",
    )

    parser.add_argument(
        "--amp",
        action="store_true",
    )


    parser.add_argument(
        "--cpu",
        action="store_true",
        help=(
            "Force CPU execution even when CUDA is available."
        ),
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--multi-gpu",
        action="store_true",
    )

    parser.add_argument(
        "--devices",
        type=str,
        default="0,1,2,3",
    )

    parser.add_argument(
        "--cpu-affinity",
        action="store_true",
        help=(
            "Optionally pin the process to a CPU block. "
            "Disabled by default for portability."
        ),
    )

    parser.add_argument(
        "--cpus-per-gpu",
        type=int,
        default=12,
    )


    parser.add_argument(
        "--validation-only",
        action="store_true",
        help=(
            "Train with validation checkpoint selection "
            "without constructing or accessing TEST."
        ),
    )

    parser.add_argument(
        "--delete-checkpoint-after-test",
        action="store_true",
        help=(
            "Delete checkpoint.pth after the final test evaluation."
        ),
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _validate_args(
        args
    )

    _install_framework_fields(
        args
    )

    _configure_device(
        args
    )

    if args.cpu_affinity:
        gpu_indices = (
            args.device_ids
            if args.device_ids
            else [args.gpu]
        )

        _set_cpu_affinity(
            gpu_indices=gpu_indices,
            cpus_per_gpu=args.cpus_per_gpu,
        )

    _print_run_header(
        args
    )

    metrics_per_run = []

    base_seed = int(
        args.seed
    )

    for run_index in range(
        int(args.runs)
    ):
        seed = (
            base_seed
            + run_index
        )

        _seed_everything(
            seed
        )

        args.current_seed = seed
        args.seed = seed

        run_name = _build_run_name(
            args,
            seed,
        )

        experiment = ClassificationExperiment(
            args
        )

        print(
            f">>>>>>> start training: {run_name}"
        )

        experiment.train(
            run_name
        )

        if args.validation_only:
            print(
                f">>>>>>> validation-only final: {run_name}"
            )

            metrics = (
                experiment.evaluate_validation_only()
            )

            print(
                "VAL_ONLY_SAFETY_PASS: TEST NEVER ACCESSED"
            )

        else:
            print(
                f">>>>>>> testing: {run_name}"
            )

            metrics = experiment.test(
                run_name
            )

        metrics_per_run.append(
            metrics
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    means = {
        metric: float(
            np.mean(
                [
                    run_metrics[metric]
                    for run_metrics in metrics_per_run
                ]
            )
        )
        for metric in METRIC_NAMES
    }

    stds = {
        metric: float(
            np.std(
                [
                    run_metrics[metric]
                    for run_metrics in metrics_per_run
                ]
            )
        )
        for metric in METRIC_NAMES
    }

    print()
    print(
        "=" * 88
    )

    title = (
        "VALIDATION-ONLY RESULTS"
        if args.validation_only
        else "FINAL RESULTS"
    )

    print(
        f"{title}: {_architecture_name(args)}"
    )

    print(
        "=" * 88
    )

    print(
        f"Mean accuracy: {means['accuracy']:.4f}, "
        f"precision: {means['precision_macro']:.4f}, "
        f"recall: {means['recall_macro']:.4f}, "
        f"f1: {means['f1_macro']:.4f}, "
        f"AUROC: {means['auroc_ovr']:.4f}, "
        f"AUPRC: {means['auprc_macro']:.4f}"
    )

    print(
        f"Std accuracy: {stds['accuracy']:.4f}, "
        f"precision: {stds['precision_macro']:.4f}, "
        f"recall: {stds['recall_macro']:.4f}, "
        f"f1: {stds['f1_macro']:.4f}, "
        f"AUROC: {stds['auroc_ovr']:.4f}, "
        f"AUPRC: {stds['auprc_macro']:.4f}"
    )

    print(
        "=" * 88
    )


if __name__ == "__main__":
    main()
