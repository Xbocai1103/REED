

from __future__ import annotations

import os
import random
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import BaseExperiment
from utils.tools import EarlyStopping, adjust_learning_rate


MetricDict = Dict[str, float]


class ClassificationExperiment(BaseExperiment):


    def _build_model(self):
        # Infer sequence length, input width, and class count from TRAIN only.
        train_data, _ = self._get_data(flag="TRAIN")

        self.args.sequence_length = int(train_data.max_seq_len)
        self.args.input_channels = int(train_data.X.shape[2])
        self.args.num_classes = int(len(np.unique(train_data.y)))

        # Upstream data/layer utilities still read these framework fields.
        self.args.seq_len = self.args.sequence_length
        self.args.enc_in = self.args.input_channels
        self.args.num_class = self.args.num_classes
        self.args.pred_len = 0

        model = self.model_registry[
            self.args.model
        ].Model(
            self.args
        ).float()

        if (
            self.args.use_multi_gpu
            and self.args.use_gpu
        ):
            model = nn.DataParallel(
                model,
                device_ids=self.args.device_ids,
            )

        return model

    def _get_data(self, flag):
        random.seed(
            int(self.args.current_seed)
        )
        self.args.seed = int(
            self.args.current_seed
        )
        return data_provider(
            self.args,
            flag,
        )

    def _select_optimizer(self):
        return optim.Adam(
            self.model.parameters(),
            lr=float(self.args.learning_rate),
        )

    def _select_criterion(self):
        return nn.CrossEntropyLoss()

    def _core_model(self):
        if isinstance(
            self.model,
            nn.DataParallel,
        ):
            return self.model.module

        return self.model

    def _subspace_weight(
        self,
        epoch_index: int,
    ) -> float:
        if not bool(
            self.args.enable_subspace_penalty
        ):
            return 0.0

        target = float(
            self.args.subspace_weight
        )

        warmup = int(
            self.args.subspace_warmup
        )

        if warmup <= 0:
            return target

        progress = min(
            1.0,
            float(epoch_index + 1)
            / float(warmup),
        )

        return (
            target
            * progress
        )

    @staticmethod
    def _format_metrics(
        prefix: str,
        loss: float,
        metrics: MetricDict,
    ) -> str:
        return (
            f"{prefix} --- "
            f"Loss: {loss:.5f}, "
            f"Accuracy: {metrics['accuracy']:.5f}, "
            f"Precision: {metrics['precision_macro']:.5f}, "
            f"Recall: {metrics['recall_macro']:.5f}, "
            f"F1: {metrics['f1_macro']:.5f}, "
            f"AUROC: {metrics['auroc_ovr']:.5f}, "
            f"AUPRC: {metrics['auprc_macro']:.5f}"
        )

    def evaluate(
        self,
        data_loader,
        criterion,
    ) -> Tuple[float, MetricDict]:
        losses = []
        logits_all = []
        labels_all = []

        self.model.eval()

        with torch.no_grad():
            for batch_x, labels, padding_mask in data_loader:
                del padding_mask

                batch_x = batch_x.float().to(
                    self.device
                )

                labels = labels.long().to(
                    self.device
                )

                logits = self.model(
                    batch_x
                )

                loss = criterion(
                    logits,
                    labels,
                )

                losses.append(
                    float(
                        loss.detach().cpu().item()
                    )
                )

                logits_all.append(
                    logits.detach().cpu()
                )

                labels_all.append(
                    labels.detach().cpu()
                )

        if not logits_all:
            raise RuntimeError(
                "Evaluation loader produced no batches."
            )

        logits = torch.cat(
            logits_all,
            dim=0,
        )

        labels = torch.cat(
            labels_all,
            dim=0,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predictions = torch.argmax(
            probabilities,
            dim=1,
        ).numpy()

        labels_np = labels.numpy()
        probabilities_np = probabilities.numpy()

        labels_onehot = torch.nn.functional.one_hot(
            labels,
            num_classes=int(self.args.num_classes),
        ).float().numpy()

        metrics = {
            "accuracy": accuracy_score(
                labels_np,
                predictions,
            ),
            "precision_macro": precision_score(
                labels_np,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "recall_macro": recall_score(
                labels_np,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "f1_macro": f1_score(
                labels_np,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "auroc_ovr": roc_auc_score(
                labels_onehot,
                probabilities_np,
                multi_class="ovr",
            ),
            "auprc_macro": average_precision_score(
                labels_onehot,
                probabilities_np,
                average="macro",
            ),
        }

        self.model.train()

        return (
            float(np.mean(losses)),
            metrics,
        )

    # Keep the base-project validation hook name expected by existing scripts.
    def vali(
        self,
        data_set,
        data_loader,
        criterion,
    ):
        del data_set
        return self.evaluate(
            data_loader,
            criterion,
        )

    def train(
        self,
        run_name: str,
    ):
        train_data, train_loader = self._get_data(
            flag="TRAIN"
        )

        validation_data, validation_loader = self._get_data(
            flag="VAL"
        )

        print(
            f"TRAIN X/y: {train_data.X.shape} / {train_data.y.shape}"
        )

        print(
            f"VAL   X/y: {validation_data.X.shape} / "
            f"{validation_data.y.shape}"
        )

        checkpoint_dir = os.path.join(
            "./checkpoints",
            self.args.model,
            run_name,
        )

        os.makedirs(
            checkpoint_dir,
            exist_ok=True,
        )

        train_steps = len(
            train_loader
        )

        early_stopping = EarlyStopping(
            patience=int(self.args.patience),
            verbose=True,
            delta=1e-5,
        )

        optimizer = self._select_optimizer()
        criterion = self._select_criterion()

        use_subspace_penalty = bool(
            self.args.enable_subspace_penalty
        )

        if (
            use_subspace_penalty
            and isinstance(
                self.model,
                nn.DataParallel,
            )
        ):
            raise RuntimeError(
                "Decision-subspace regularization is validated only for "
                "single-GPU training."
            )

        core_model = self._core_model()

        for epoch in range(
            int(self.args.train_epochs)
        ):
            total_losses = []
            classification_losses = []
            penalty_losses = []

            self.model.train()
            start_time = time.time()

            penalty_weight = self._subspace_weight(
                epoch
            )

            for batch_x, labels, padding_mask in train_loader:
                del padding_mask

                optimizer.zero_grad()

                batch_x = batch_x.float().to(
                    self.device
                )

                labels = labels.long().to(
                    self.device
                )

                if use_subspace_penalty:
                    features = core_model.encode(
                        batch_x
                    )

                    logits = core_model.classify(
                        features
                    )

                    penalty = core_model.compute_subspace_penalty(
                        features
                    )
                else:
                    logits = self.model(
                        batch_x
                    )
                    penalty = None

                classification_loss = criterion(
                    logits,
                    labels,
                )

                if penalty is None:
                    loss = classification_loss
                else:
                    loss = (
                        classification_loss
                        + penalty_weight * penalty
                    )

                total_losses.append(
                    float(
                        loss.detach().cpu().item()
                    )
                )

                classification_losses.append(
                    float(
                        classification_loss.detach().cpu().item()
                    )
                )

                if penalty is not None:
                    penalty_losses.append(
                        float(
                            penalty.detach().cpu().item()
                        )
                    )

                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=4.0,
                )

                optimizer.step()

            epoch_seconds = (
                time.time()
                - start_time
            )

            train_loss = float(
                np.mean(total_losses)
            )

            classification_loss_mean = float(
                np.mean(classification_losses)
            )

            penalty_loss_mean = (
                float(
                    np.mean(penalty_losses)
                )
                if penalty_losses
                else 0.0
            )

            validation_loss, validation_metrics = self.evaluate(
                validation_loader,
                criterion,
            )

            print(
                f"Epoch: {epoch + 1} "
                f"cost time: {epoch_seconds:.2f}s"
            )

            if use_subspace_penalty:
                print(
                    f"Epoch: {epoch + 1}, "
                    f"Steps: {train_steps}, | "
                    f"Train Loss: {train_loss:.5f}, "
                    f"CE: {classification_loss_mean:.5f}, "
                    f"Subspace Penalty: {penalty_loss_mean:.5f}, "
                    f"Penalty Weight: {penalty_weight:.5f}"
                )
            else:
                print(
                    f"Epoch: {epoch + 1}, "
                    f"Steps: {train_steps}, | "
                    f"Train Loss: {train_loss:.5f}"
                )

            print(
                self._format_metrics(
                    "Validation results",
                    validation_loss,
                    validation_metrics,
                )
            )

            # Existing EarlyStopping minimizes its score, therefore negative
            # validation Macro-F1 performs maximization.
            early_stopping(
                -validation_metrics["f1_macro"],
                self.model,
                checkpoint_dir,
            )

            if early_stopping.early_stop:
                print(
                    "Early stopping"
                )
                break

            adjust_learning_rate(
                optimizer,
                epoch + 1,
                self.args,
            )

        best_model_path = os.path.join(
            checkpoint_dir,
            "checkpoint.pth",
        )

        if not os.path.exists(
            best_model_path
        ):
            raise FileNotFoundError(
                "Validation-best checkpoint was not created: "
                f"{best_model_path}"
            )

        state_dict = torch.load(
            best_model_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            state_dict
        )

        print(
            "Loaded validation-best checkpoint: "
            f"{best_model_path}"
        )

        return self.model

    def evaluate_validation_only(
        self,
    ) -> MetricDict:
        validation_data, validation_loader = self._get_data(
            flag="VAL"
        )

        criterion = self._select_criterion()

        validation_loss, metrics = self.vali(
            validation_data,
            validation_loader,
            criterion,
        )

        print(
            self._format_metrics(
                "Validation-best results",
                validation_loss,
                metrics,
            )
        )

        return metrics

    def test(
        self,
        run_name: str,
        load_checkpoint: bool = False,
    ) -> MetricDict:
        validation_data, validation_loader = self._get_data(
            flag="VAL"
        )

        test_data, test_loader = self._get_data(
            flag="TEST"
        )

        checkpoint_dir = os.path.join(
            "./checkpoints",
            self.args.model,
            run_name,
        )

        model_path = os.path.join(
            checkpoint_dir,
            "checkpoint.pth",
        )

        if load_checkpoint:
            if not os.path.exists(
                model_path
            ):
                raise FileNotFoundError(
                    f"No model found at {model_path}"
                )

            state_dict = torch.load(
                model_path,
                map_location=self.device,
            )

            self.model.load_state_dict(
                state_dict
            )

        criterion = self._select_criterion()

        validation_loss, validation_metrics = self.vali(
            validation_data,
            validation_loader,
            criterion,
        )

        test_loss, test_metrics = self.vali(
            test_data,
            test_loader,
            criterion,
        )

        print(
            "Final evaluation using the validation-selected checkpoint"
        )

        print(
            self._format_metrics(
                "Validation results",
                validation_loss,
                validation_metrics,
            )
        )

        print(
            self._format_metrics(
                "Test results",
                test_loss,
                test_metrics,
            )
        )

        if bool(
            self.args.delete_checkpoint_after_test
        ):
            self._delete_checkpoint(
                model_path
            )

        return test_metrics

    @staticmethod
    def _delete_checkpoint(
        checkpoint_path: str,
    ) -> None:
        if os.path.exists(
            checkpoint_path
        ):
            os.remove(
                checkpoint_path
            )

            print(
                "Validation-best checkpoint deleted."
            )


__all__ = [
    "ClassificationExperiment",
]
