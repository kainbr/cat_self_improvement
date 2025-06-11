import json
import logging
from logging import Logger
from typing import Callable

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from pkg.data import Data
from pkg.data.utils import PADDING_VALUE
from pkg.logic.assessment import AssessmentLogic
from pkg.logic.policy import PolicyLogic
from pkg.logic.utils import SampleState
from pkg.model.stedu.gather_utils import gather_randomly_and_split_sequences
from pkg.model.stedu.policy_random import Model as RandomPolicy
from pkg.model.stedu.policy_uncertainty import Model as UncertaintyPolicy
from pkg.model.stedu.stedu_assessment import Model as AssessmentModel
from pkg.model.stedu.stedu_policy import Model as PolicyModel
from pkg.model.utils import ResponseSequenceCache
from pkg.utils.logging import RESULTS_FILE_NAME, Timer, save_snapshot_of_source_code
from pkg.utils.reproduce import save_config, seed_everything

logger: Logger = logging.getLogger(__name__)
EARLY_STOPPING_TOLERANCE = 50
EARLY_STOPPING_INNER_LOOP_TOLERANCE = 10
GRADNORM_CLIPPING_VALUE_DEFAULT = 1
SAVE_CHECKPOINT = False


@hydra.main(config_path="pkg/config", config_name="cat", version_base="1.3")
def main(config: DictConfig) -> float:

    # save code + config
    save_snapshot_of_source_code(file_name="cat.py")
    save_config(config=config)

    # dispatch to trainer
    logic: Logic = hydra.utils.instantiate(config=config)

    if logic.time_inference:
        logic.run_time_inference()
        return

    best_metric = logic.run()

    logger.info(f"Fin: {best_metric=}")


class Logic:
    def __init__(
        self,
        data: DictConfig,
        assessment_model: DictConfig,
        policy_model: DictConfig,
        optimizer: DictConfig,
        max_epochs: int,
        device: str,
        eval_modes: list[str],
        debug: bool,
        seed: int,
        max_assessment_epochs: int,
        num_samples: int,
        finetune_assessment_model: bool,
        response_seq_score_type: str,  # "acc", "loss"
        temperature: float,
        num_given_positions: int,
        gradnorm_clipping_value: float | None = GRADNORM_CLIPPING_VALUE_DEFAULT,
        time_inference: bool = False,
        force_random_num_given_positions: bool = False,
    ) -> None:

        self.sample_state = SampleState()
        self.writer = SummaryWriter(log_dir=".tb")

        self.device = torch.device(device)
        self.max_epochs = max_epochs
        self.eval_modes = eval_modes
        self.debug = debug
        self.seed = seed

        # algorithm logic
        self.num_given_positions = num_given_positions
        self.num_samples = num_samples
        self.temperature = temperature
        self.finetune_assessment = finetune_assessment_model
        self.response_seq_score_type = response_seq_score_type
        self.time_inference = time_inference

        # optimizer
        self.gradnorm_clipping_value = gradnorm_clipping_value
        self.assessment_optimizer_config = optimizer
        self.policy_optimizer_config = optimizer
        # TODO: Test sharing models/components again?

        # assessment training
        self.max_assessment_epochs = max_assessment_epochs

        # `force_random_...` trains assessment model on random lengths
        # (we do this, to align training of our policy with uncertainty policy training, for fair comparison)
        self.force_random_num_given_positions = force_random_num_given_positions

        # data
        seed_everything(seed=self.seed)
        self.data: Data = hydra.utils.instantiate(data)

        # models
        seed_everything(seed=self.seed)
        self.assessment_model: AssessmentModel = hydra.utils.instantiate(
            assessment_model,
            vectorizer=self.data.dataset.vectorizer,
            max_interaction_len=self.data.dataset.max_interaction_len,
            max_position_idx=self.data.dataset.max_position_idx,
        ).to(self.device)
        self.assessment = AssessmentLogic(
            data=self.data,
            model=self.assessment_model,
            optimizer=self.assessment_optimizer_config,
            device=self.device,
            eval_modes=self.eval_modes,
            debug=self.debug,
            seed=self.seed,
            writer=self.writer,
            gradnorm_clipping_value=self.gradnorm_clipping_value,
        )

        seed_everything(seed=self.seed)
        if policy_model._target_ == "pkg.model.stedu.policy_random.Model":
            self.policy_model: RandomPolicy = hydra.utils.instantiate(policy_model).to(
                self.device
            )
        elif policy_model._target_ == "pkg.model.stedu.policy_uncertainty.Model":
            self.policy_model: UncertaintyPolicy = hydra.utils.instantiate(
                policy_model, assessment_model=self.assessment_model
            ).to(self.device)
            self.policy_model_best: UncertaintyPolicy = hydra.utils.instantiate(
                policy_model, assessment_model=self.assessment_model
            ).to(self.device)
            self.assessment_model_best: AssessmentModel = hydra.utils.instantiate(
                assessment_model,
                vectorizer=self.data.dataset.vectorizer,
                max_interaction_len=self.data.dataset.max_interaction_len,
                max_position_idx=self.data.dataset.max_position_idx,
            ).to(self.device)
        else:
            self.policy_model: PolicyModel = hydra.utils.instantiate(
                policy_model,
                vectorizer=self.data.dataset.vectorizer,
                max_interaction_len=self.data.dataset.max_interaction_len,
                max_position_idx=self.data.dataset.max_position_idx,
            ).to(self.device)
            self.policy_model_best: PolicyModel = hydra.utils.instantiate(
                policy_model,
                vectorizer=self.data.dataset.vectorizer,
                max_interaction_len=self.data.dataset.max_interaction_len,
                max_position_idx=self.data.dataset.max_position_idx,
            ).to(self.device)

        self.policy = PolicyLogic(
            data=self.data,
            model=self.policy_model,
            optimizer=self.policy_optimizer_config,
            device=self.device,
            eval_modes=self.eval_modes,
            debug=self.debug,
            seed=self.seed,
            writer=self.writer,
        )

        # training seed
        seed_everything(seed=self.seed)

    def run(self) -> None:
        # (Require) initial training of assessment model
        self.assessment.train(
            max_epochs=self.max_assessment_epochs,
            sample_fn=self.get_policy_sampler(
                schema="random",
                policy_model=self.policy_model,
                random_num_given_positions=isinstance(
                    self.policy_model, UncertaintyPolicy
                )
                or self.force_random_num_given_positions,
            ),
        )
        assert self.assessment.model is self.assessment_model

        # (l.1)
        score_best_policy = self.policy.eval_score(
            assessment_model=self.assessment_model,
            sample_fn=self.get_policy_sampler(
                schema="argmax", policy_model=self.policy_model
            ),
        )

        # we can skip the main loop for random policy evaluation
        if isinstance(self.policy_model, RandomPolicy):
            return score_best_policy

        # we can skip the main loop for uncertainty policy evaluation
        if isinstance(self.policy_model, UncertaintyPolicy):
            if not self.finetune_assessment:
                return score_best_policy
            else:
                assert isinstance(self.policy_model_best, UncertaintyPolicy)

                for _ in range(1, self.max_epochs + 1):
                    # fix assessment model for sampling from "best policy"
                    self.assessment_model_best.load_state_dict(
                        self.assessment_model.state_dict()
                    )
                    self.policy_model_best.assessment_model = self.assessment_model_best

                    self.policy.state.iteration += 1
                    self.assessment.train(
                        max_epochs=self.max_assessment_epochs,
                        sample_fn=self.get_policy_sampler(
                            schema="prob",
                            policy_model=self.policy_model_best,
                            random_num_given_positions=True,  # because: uncertainty
                        ),
                    )

                    score_policy = self.policy.eval_score(
                        assessment_model=self.assessment_model,
                        sample_fn=self.get_policy_sampler(
                            schema="argmax", policy_model=self.policy_model
                        ),
                    )

                    if score_best_policy < score_policy:
                        score_best_policy = score_policy
                    else:
                        break

                return score_best_policy

        self.policy_model_best.load_state_dict(self.policy_model.state_dict())

        # (l.2)
        for _ in range(1, self.max_epochs + 1):
            # (ll.3-6)
            response_sequence_cache = self.sample_subset_candidates(self.num_samples)
            response_sequence_cache.argmin()

            # (ll.7-10)
            self.policy.train_epoch(
                sample_fn=self.get_policy_sampler(
                    schema="from_cache",
                    cache=response_sequence_cache,
                    policy_model=self.policy_model,
                ),
            )
            self.policy.eval_imitation(
                sample_fn=self.get_policy_sampler(
                    schema="from_cache",
                    cache=response_sequence_cache,
                    policy_model=self.policy_model,
                ),
            )

            # (l.11)
            score_policy = self.policy.eval_score(
                assessment_model=self.assessment_model,
                sample_fn=self.get_policy_sampler(
                    schema="argmax", policy_model=self.policy_model
                ),
            )
            if score_best_policy < score_policy:
                # (l.12)
                score_best_policy = score_policy
                self.policy_model_best.load_state_dict(self.policy_model.state_dict())
                # (l.13)
                if self.finetune_assessment:
                    self.assessment.train(
                        max_epochs=self.max_assessment_epochs,
                        sample_fn=self.get_policy_sampler(
                            schema="prob",
                            policy_model=self.policy_model_best,
                            random_num_given_positions=self.force_random_num_given_positions,
                        ),
                    )

            if self.policy.state.num_epochs_not_improved > EARLY_STOPPING_TOLERANCE:
                logger.info("Break main loop due to early stopping")
                break

        return self.policy.state.best_val_metric

    @torch.no_grad()
    def run_time_inference(self) -> None:
        logger.info(
            f"Only run timing for inference on random policy_model (loop over all test students)"
        )

        self.eval_modes = ["test"]

        self.policy_model.eval()

        timer = Timer()

        sample_fn = self.get_policy_sampler(
            schema="argmax", policy_model=self.policy_model
        )

        for batch_idx, batch in enumerate(self.data.test_loader):
            users = batch[0].to(self.device)
            sequences = batch[1].to(self.device)
            _ = sample_fn(users, sequences)

        logger.info(f"{timer.get_duration_in_seconds()=}")
        return timer.get_duration_in_seconds()

    @torch.no_grad()
    def sample_subset_candidates(self, m: int) -> ResponseSequenceCache:
        logger.info(f"sample {m=} subset candidates")

        self.sample_state.iteration += 1
        self.assessment_model.eval()

        response_sequence_cache = ResponseSequenceCache()

        for mode in self.eval_modes:
            logger.info(f"{mode=}")

            for _ in range(m):
                for batch in getattr(self.data, f"{mode}_loader"):
                    users = batch[0].to(self.device)
                    sequences = batch[1].to(self.device)

                    # Sample
                    sample_fn = self.get_policy_sampler(
                        schema="prob", policy_model=self.policy_model_best
                    )
                    given_positions = sample_fn(users, sequences)

                    # Score sampled response sequence with current assessment model
                    _, info_dict = self.assessment_model.forward(
                        x=sequences, given_positions=given_positions
                    )

                    if self.response_seq_score_type == "acc":
                        scores = info_dict["acc_response_per_seq"]
                    elif self.response_seq_score_type == "loss":
                        scores = -info_dict["loss_per_seq"]  # negative loss
                    else:
                        raise NotImplementedError

                    # Add to intermediate cache
                    response_sequence_cache.add_response_seqs(
                        users=users, scores=scores, response_seqs=given_positions
                    )

                    del info_dict, _
                    torch.cuda.empty_cache()

        return response_sequence_cache

    def get_policy_sampler(
        self,
        schema: str,
        policy_model: PolicyModel,
        cache: ResponseSequenceCache | None = None,
        random_num_given_positions: bool = False,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:

        if schema in ["random", "argmax", "prob"]:
            generate_positions_fn = lambda users, sequences: sample_given_positions(
                x=sequences,
                scheme=schema,
                policy_model=policy_model,
                num_given_positions=self.num_given_positions,
                random_num_given_positions=random_num_given_positions,
                temperature=self.temperature,
            )
        elif schema == "from_cache":
            assert cache is not None
            assert not random_num_given_positions
            generate_positions_fn = (
                lambda users, sequences: cache.get_random_response_seqs(users=users)[1]
            )
        else:
            raise ValueError

        return generate_positions_fn


@torch.no_grad()
def sample_given_positions(
    x: torch.Tensor,
    scheme: str,
    num_given_positions: int,
    policy_model: PolicyModel,
    temperature: float | None = None,
    random_num_given_positions: bool = False,
) -> torch.Tensor | None:

    assert scheme in ["random", "argmax", "prob"]

    if random_num_given_positions:
        # random num_given_positions per batch for now (TODO: per instance)
        # used for example in uncertainty policy
        num_given_positions = torch.randint(num_given_positions + 1, size=(1,)).item()

    if num_given_positions == 0:
        return None

    B, S, m, f = x.shape
    assert x.dim() == 4
    assert f == 3

    policy_model.eval()

    if (scheme == "random") or isinstance(policy_model, RandomPolicy):
        x_given, _ = gather_randomly_and_split_sequences(
            x=x, num_given_positions=num_given_positions
        )
        assert x_given.shape == (B, num_given_positions, m, f)
        x_given_positions = x_given[..., 0, 2]  # Assumes positions == questions
    else:
        x_given = None

        for i in range(num_given_positions):
            _, info_dict = policy_model.forward(
                x=x,
                given_positions=x_given,
                sampling_scheme=scheme,
                temperature=temperature,
            )
            next_x_given: torch.Tensor = info_dict["next"]
            assert next_x_given.shape == (B, 1)

            if x_given is None:
                x_given = next_x_given
            else:
                x_given = torch.cat([x_given, next_x_given], dim=1)
                assert x_given.shape == (B, i + 1)

        assert x_given.shape == (B, num_given_positions)
        x_given_positions = x_given

    assert (x_given_positions != PADDING_VALUE).all()

    return x_given_positions.detach()


if __name__ == "__main__":
    main()
