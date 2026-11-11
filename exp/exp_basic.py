

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from models import REED


MODEL_REGISTRY = {
    "REED": REED,
}


class BaseExperiment(ABC):


    def __init__(self, args):
        self.args = args
        self.model_registry = MODEL_REGISTRY

        self.device = self._acquire_device()
        self.model = self._build_model().to(
            self.device
        )

    @abstractmethod
    def _build_model(self):

        raise NotImplementedError

    def _acquire_device(self) -> torch.device:


        use_gpu = bool(
            getattr(self.args, "use_gpu", True)
            and torch.cuda.is_available()
        )

        self.args.use_gpu = use_gpu

        if not use_gpu:
            print("Use CPU")
            return torch.device("cpu")

        use_multi_gpu = bool(
            getattr(self.args, "use_multi_gpu", False)
        )

        if use_multi_gpu:
            devices = str(
                getattr(self.args, "devices", "0")
            ).replace(" ", "")

            device_ids = [
                int(device_id)
                for device_id in devices.split(",")
                if device_id
            ]

            if not device_ids:
                raise ValueError(
                    "At least one GPU id is required when "
                    "use_multi_gpu=True."
                )

            self.args.device_ids = device_ids
            self.args.gpu = device_ids[0]

        gpu_id = int(
            getattr(self.args, "gpu", 0)
        )

        if gpu_id < 0:
            raise ValueError(
                f"gpu must be >= 0, got {gpu_id}."
            )

        device = torch.device(
            f"cuda:{gpu_id}"
        )

        print(
            f"Use GPU: cuda:{gpu_id}"
        )

        return device


__all__ = [
    "BaseExperiment",
    "MODEL_REGISTRY",
]
