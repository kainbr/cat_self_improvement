import logging

import numpy as np
import torch

from pkg.data import BaseDataset
from pkg.data.neurips.utils import Interaction

logger = logging.getLogger(name=__name__)


class Dataset(BaseDataset):
    def __init__(
        self,
        seed: int,
        num_train_students: int,
        num_val_students: int,
        num_test_students: int,
        num_questions: int,
        num_splits: int,
        ratio_random_questions: float,
        guessing_prob: float,
        slipping_prob: float,
        max_sequence_len: None = None,  # unused: only for interface compatibility
    ) -> None:

        super().__init__(
            dataset="synthetic_1",
            user_features=[],
            interaction_features=[],
            position_type="question",
            max_num_sequences=num_train_students + num_val_students + num_test_students,
            max_sequence_len=num_questions,
            data_path="",
            cache_path=None,  # No caching
        )

        # Assert that data is not cached
        assert self.sequences is None

        assert isinstance(ratio_random_questions, float)
        assert (ratio_random_questions >= 0.0) and (ratio_random_questions <= 1.0)
        assert (slipping_prob >= 0.0) and (slipping_prob <= 1.0)
        assert (guessing_prob >= 0.0) and (guessing_prob <= 1.0)
        assert (num_questions * ratio_random_questions).is_integer()

        num_random_questions = int(num_questions * ratio_random_questions)
        num_questions -= num_random_questions
        assert num_random_questions >= 0
        assert num_questions % num_splits == 0
        num_repetitions = num_questions // num_splits

        num_students = num_train_students + num_val_students + num_test_students

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        response_probs = torch.randint(0, 2, size=(num_students, num_splits))
        response_probs = response_probs.type(torch.float32)
        response_probs[response_probs == 0.0] += guessing_prob
        response_probs[response_probs == 1.0] -= slipping_prob
        response_probs = response_probs.repeat_interleave(num_repetitions, dim=1)

        random_probs = torch.full(
            size=(num_students, num_random_questions), fill_value=0.5
        )
        response_probs = torch.cat([random_probs, response_probs], dim=1)

        # Sample responses
        response_matrix = response_probs > torch.rand_like(response_probs)

        # Transform to interaction sequence
        interaction_sequences: dict[int, list[Interaction]] = {
            i: [
                Interaction(
                    question=j,
                    response=r.item(),
                    position=self.vectorizer.encode_value(group="question", value=j),
                )
                for j, r in enumerate(response_matrix[i])
            ]
            for i in range(num_students)
        }

        self.max_interaction_len = max(
            [len(i) for s in interaction_sequences.values() for i in s]
        )

        logger.info(f"Encoding user sequences")
        self.sequences = {
            i: np.concatenate(
                [
                    np.expand_dims(
                        s.encode(
                            vectorizer=self.vectorizer,
                            max_len=self.max_interaction_len,
                        ),
                        axis=0,
                    )
                    for s in sequences
                ],
                axis=0,
            )
            for i, sequences in interaction_sequences.items()
        }

        # statistics
        # TODO: Refactor that max_position_idx is removed if it is not needed for positional embedding anymore
        # ATTENTION! +1 because of 0 position. look out! dangerous when refactoring
        # ATTENTION! Assumes that we have encoded positions
        self.max_position_idx = max(s[:, -1].max() for s in self.sequences.values()) + 1

        # logging statistics
        logger.info(f"{len(self.sequences)=}")
        logger.info(f"{self.max_position_idx=}")
        logger.info(f"{self.vectorizer.data=}")
