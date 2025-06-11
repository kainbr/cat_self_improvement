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
from pkg.model.stedu.policy_random import Model as RandomPolicy
from pkg.model.stedu.policy_uncertainty import Model as UncertaintyPolicy
from pkg.model.stedu.stedu_assessment import Model as AssessmentModel
from pkg.model.stedu.stedu_policy import Model as PolicyModel
from pkg.utils.logging import Timer

logger: Logger = logging.getLogger(__name__)

GRADNORM_CLIPPING_VALUE_DEFAULT = 1
EARLY_STOPPING_TOLERANCE = 10


class PolicyLogic:
    def __init__(
        self,
        data: Data,
        model: PolicyModel | RandomPolicy | UncertaintyPolicy,
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

        self.state = State()

        # data
        self.data = data

        # model
        self.model = model

        # optimizer
        if isinstance(self.model, PolicyModel):
            self.optimizer: optim.Optimizer = hydra.utils.instantiate(
                optimizer, params=self.model.parameters()
            )

    def train_epoch(
        self,
        sample_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        logger.info(f"Train current policy model on cache for one epoch")

        self.model.train()

        timer = Timer()
        result = {}
        self.state.epoch += 1

        for batch in self.data.train_loader:
            self.state.iteration += 1
            users = batch[0].to(self.device)
            sequences = batch[1].to(self.device)

            # apply generation function on user sequences
            given_positions = sample_fn(users, sequences)
            loss, info_dict = self.model.forward(
                x=sequences, given_positions=given_positions
            )

            self.optimizer.zero_grad()
            loss.backward()

            if self.gradnorm_clipping_value is not None:
                grad_norm_before_clipping = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradnorm_clipping_value
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
                result=result,
                state=self.state,
                writer=self.writer,
                model="policy_training",
            )

    @torch.no_grad()
    def eval_imitation(
        self,
        sample_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        """Eval policy prediction (only for inspection).
        Performance of policy model imitating generated data."""

        logger.info(f"Eval current policy model on imitating the cached sequences")

        self.model.eval()

        eval_result = dict()
        for mode in self.eval_modes:
            logger.info(f"{mode=}")
            timer = Timer()
            losses, y_trues, y_preds = [], [], []

            for batch in getattr(self.data, f"{mode}_loader"):
                users = batch[0].to(self.device)
                sequences = batch[1].to(self.device)

                # apply generation function on user sequences
                given_positions = sample_fn(users, sequences)
                loss, info_dict = self.model.forward(
                    x=sequences, given_positions=given_positions
                )
                losses.append(loss.detach().cpu().numpy())

                y_true_batch = info_dict["targets"].detach().cpu().numpy()
                y_pred_batch = info_dict["predictions"].detach().cpu().numpy()

                y_trues.append(y_true_batch.flatten())
                y_preds.append(y_pred_batch.flatten())

                del loss
                del info_dict
                torch.cuda.empty_cache()

            y_true = np.concatenate(y_trues, axis=0)
            y_pred = np.concatenate(y_preds, axis=0)

            acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)

            result = {
                f"{mode}_time": timer.get_duration_in_seconds(),
                f"{mode}_epoch_mean_loss": float(np.mean(losses)),
                f"{mode}_epoch_acc": acc,
            }
            eval_result.update(result)

        log_results(
            result=eval_result,
            state=self.state,
            writer=self.writer,
            model="policy_imitation",
        )

    @torch.no_grad()
    def eval_score(
        self,
        assessment_model: AssessmentModel,
        sample_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        """Eval policy with respect to score of assessment model (metric of interest).
        Performance of assessment model on remaining question bank."""

        logger.info(f"Eval current policy model with (best) assessment model")

        assessment_model.eval()

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
                given_positions = sample_fn(users, sequences)
                loss, info_dict = assessment_model.forward(
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

            # # double check acc
            # threshold = 0.5
            # binary_pred = (probs_all >= threshold).astype(int)
            # acc_total = metrics.accuracy_score(y_true=targets_all, y_pred=binary_pred)
            # assert np.allclose(np.array(acc_mean), np.array(acc_total), atol=0.001)

            # sanity check targets_all
            assert len(targets_all) == num_responses

            result = {
                f"{mode}_time": timer.get_duration_in_seconds(),
                f"{mode}_epoch_mean_loss": loss_mean,
                f"{mode}_epoch_acc": acc_mean,
                f"{mode}_epoch_auc": auc_total,
                f"{mode}_num_predictions": num_responses.item(),
            }
            eval_result.update(result)

            if mode == "val":

                val_metric = float(acc_mean)

                # ATTENTION: assuming higher is better
                if val_metric > self.state.best_val_metric:
                    self.state.best_val_metric = val_metric
                    self.state.best_iteration = self.state.iteration

                    logger.info(
                        f"Improved policy model: {val_metric=} (reset early stopping counter)"
                    )
                    self.state.num_epochs_not_improved = 0

                    # logger.info("Saving best policy model weights")
                    # self.best_model_weights = self.model.state_dict()
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
            result=eval_result, state=self.state, writer=self.writer, model="policy"
        )

        return val_metric
