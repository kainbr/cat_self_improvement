import logging
from logging import Logger
from typing import Callable

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn import metrics
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from pkg.data import Data
from pkg.data.utils import PADDING_VALUE
from pkg.logic.utils import State, log_results
from pkg.model.stedu.stedu_assessment import Model as AssessmentModel
from pkg.utils.logging import Timer

logger: Logger = logging.getLogger(__name__)

GRADNORM_CLIPPING_VALUE_DEFAULT = 1
EARLY_STOPPING_TOLERANCE = 10


class AssessmentLogic:
    def __init__(
        self,
        data: Data,
        model: AssessmentModel,
        optimizer: DictConfig,
        device: str,
        eval_modes: list[str],
        debug: bool,
        seed: int,
        writer: SummaryWriter,
        gradnorm_clipping_value: float | None = GRADNORM_CLIPPING_VALUE_DEFAULT,
    ):

        self.device = device
        self.eval_modes = eval_modes
        self.debug = debug
        self.seed = seed
        self.writer = writer
        self.gradnorm_clipping_value = gradnorm_clipping_value
        self.optimizer_config = optimizer

        self.state = State()
        # IDEA: Keep parts of state on reset

        # data
        self.data = data

        # model
        self.model = model
        self.trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(f"AssessmentModel:\n{self.model}")

        # keeping track of best models
        self.best_model_weights = self.model.state_dict()

    def train(
        self,
        max_epochs: int,
        sample_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        logger.info("Train assessment model")

        # TODO: (optional) implement reset for model parameters for training from scratch

        # init new training
        self.optimizer: optim.Optimizer = hydra.utils.instantiate(
            self.optimizer_config, params=self.model.parameters()
        )
        self.state.num_epochs_not_improved = 0

        self.eval(generate_positions_fn=sample_fn)

        for epoch in range(1, max_epochs + 1):
            self.train_epoch(generate_positions_fn=sample_fn)
            self.eval(generate_positions_fn=sample_fn)

            if self.state.num_epochs_not_improved > EARLY_STOPPING_TOLERANCE:
                logger.info("Break assessment training due to early stopping")
                logger.info("Reloading best assessment model weights")
                self.model.load_state_dict(self.best_model_weights)
                return

        # no early stopping
        logger.info("Stop assessment training due to max epochs")

    def train_epoch(
        self,
        generate_positions_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):

        self.model.train()

        timer = Timer()
        result = {}

        self.state.epoch += 1

        for batch in self.data.train_loader:
            self.state.iteration += 1
            users = batch[0].to(self.device)
            sequences = batch[1].to(self.device)

            # apply generation function on user sequences
            given_positions = generate_positions_fn(users, sequences)
            loss, info_dict = self.model.forward(
                x=sequences, given_positions=given_positions
            )

            # Opt
            self.optimizer.zero_grad()
            loss.backward()
            if self.gradnorm_clipping_value is not None:
                grad_norm_before_clipping = nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradnorm_clipping_value,
                )
            self.optimizer.step()

            result = {
                "train_epoch": self.state.epoch,
                "train_iteration": self.state.iteration,
                "train_step_time": timer.get_duration_in_seconds(),
                "train_step_loss": loss.item(),
            }

            if self.gradnorm_clipping_value is not None:
                result.update(grad_norm=grad_norm_before_clipping.item())

            log_results(
                writer=self.writer, state=self.state, result=result, model="assessment"
            )

    @torch.no_grad()
    def eval(
        self,
        generate_positions_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> float | None:

        self.model.eval()

        eval_result = dict()
        for mode in self.eval_modes:
            logger.info(f"{mode=}")

            timer = Timer()

            loss_times_num_responses, acc_times_num_responses, num_responses = 0, 0, 0
            targets, logits = [], []

            for batch in getattr(self.data, f"{mode}_loader"):
                users = batch[0].to(self.device)
                sequences = batch[1].to(self.device)

                # apply generation function on user sequences
                given_positions = generate_positions_fn(users, sequences)
                loss, info_dict = self.model.forward(
                    x=sequences, given_positions=given_positions
                )

                loss = loss.detach().cpu().numpy()
                acc = info_dict["acc_response"].detach().cpu().numpy()
                num_response = info_dict["num_response"].detach().cpu().numpy()

                loss_times_num_responses += loss * num_response
                acc_times_num_responses += acc * num_response
                num_responses += num_response

                # for AUC
                targets_batch = info_dict["targets_response_per_seq"]
                logits_batch = info_dict["logits_response_per_seq"]
                torch.equal(logits_batch.isnan(), targets_batch == PADDING_VALUE)
                target_mask = ~logits_batch.isnan()
                targets_batch = torch.masked_select(targets_batch, target_mask)
                logits_batch = torch.masked_select(logits_batch, target_mask)
                targets.append(targets_batch.detach().cpu().numpy())
                logits.append(logits_batch.detach().cpu().numpy())

                del loss, target_mask, targets_batch, logits_batch
                del info_dict
                torch.cuda.empty_cache()

            loss_mean = float(loss_times_num_responses / num_responses)
            acc_mean = float(acc_times_num_responses / num_responses)

            # AUC
            targets_all = np.concatenate(targets, axis=0)
            logits_all = np.concatenate(logits, axis=0)
            probs_all = torch.sigmoid(torch.tensor(logits_all)).numpy()
            auc_total = metrics.roc_auc_score(y_true=targets_all, y_score=probs_all)

            result = {
                f"{mode}_time": timer.get_duration_in_seconds(),
                f"{mode}_epoch_mean_loss": loss_mean,
                f"{mode}_epoch_acc": acc_mean,
                f"{mode}_epoch_auc": auc_total,
            }
            eval_result.update(result)

            # Update metrics (+ update best_model_weights)
            if mode == "val":

                val_metric = float(acc_mean)

                # ATTENTION: assuming higher is better
                if val_metric > self.state.best_val_metric:
                    self.state.best_val_metric = val_metric
                    self.state.best_iteration = self.state.iteration

                    logger.info(
                        f"Improved assessment model: {val_metric=} (reset early stopping counter)"
                    )
                    self.state.num_epochs_not_improved = 0

                    logger.info("Saving best assessment model weights")
                    self.best_model_weights = self.model.state_dict()
                    self.state.num_updates += 1

                else:
                    self.state.num_epochs_not_improved += 1
                    logger.info(
                        f"{self.state.num_epochs_not_improved=}: {val_metric=} <= {self.state.best_val_metric=}"
                    )

                eval_result.update(
                    {
                        f"best_iteration": self.state.best_iteration,
                        f"best_val_metric": self.state.best_val_metric,
                        f"num_updates": self.state.num_updates,
                    }
                )

        log_results(
            result=eval_result, state=self.state, writer=self.writer, model="assessment"
        )
