import hashlib
import itertools
import json
import logging
import pickle
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Subset

from pkg.data.utils import PADDING_VALUE, Vectorizer

logger = logging.getLogger(name=__name__)


class Data:
    def __init__(
        self,
        dataset: OmegaConf,
        batch_size: int,
        batch_size_eval: int,
        num_workers: int,
        seed: int,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval
        self.num_workers = num_workers
        self.seed = seed

        self.setup()

    def setup(self):

        self.dataset: BaseDataset = hydra.utils.instantiate(config=self.dataset)

        all_indices = self.save_all_splits()
        self.extract_data_for_educat(indices=all_indices)

        logger.info("Done ...")

    def save_all_splits(self) -> np.ndarray:
        # loop to save all splits
        logger.info(f"Saving all splits ...")
        for seed in [0, 1, 2, 3, 4]:
            self.seed = seed

            # NUM_SPLITS-fold splits with fixed seed
            NUM_SPLITS = 5
            np.random.seed(0)
            indices = np.arange(len(self.dataset))
            np.random.shuffle(indices)
            idx_splits = np.array_split(indices, indices_or_sections=NUM_SPLITS)

            # assign splits to test, valid and training set according to seed
            if self.seed not in range(NUM_SPLITS):
                raise ValueError
            test_idx = self.seed
            val_idx = (self.seed + 1) % NUM_SPLITS
            train_idx = set(range(NUM_SPLITS)) - {test_idx, val_idx}

            test_indices = list(idx_splits[test_idx])
            val_indices = list(idx_splits[val_idx])
            train_splits = [
                split for (i, split) in enumerate(idx_splits) if i in train_idx
            ]
            train_indices = [m for m in itertools.chain(*train_splits)]

            # sanity check
            assert sorted(indices) == sorted(test_indices + val_indices + train_indices)

            # make subsets based on splits
            self.train_set = Subset(dataset=self.dataset, indices=train_indices)
            self.val_set = Subset(dataset=self.dataset, indices=val_indices)
            self.test_set = Subset(dataset=self.dataset, indices=test_indices)

            train_val_test_split_indices = dict(
                train_indices=[int(idx) for idx in train_indices],
                val_indices=[int(idx) for idx in val_indices],
                test_indices=[int(idx) for idx in test_indices],
            )
            with open(f"train_val_test_split_indices_seed_{self.seed}.json", "w") as f:
                json.dump(train_val_test_split_indices, fp=f)

        return indices

    def extract_data_for_educat(self, indices: list[int]) -> None:
        logger.info(f"Extract sequences for educat ...")

        # known quantities
        q_group = 1
        c_group = 2
        r_group = 0

        # init targets
        concept_map = {}
        student_question_response_triplets = []

        assert len(self.dataset) == len(indices)

        # loop over dataset
        for idx in range(len(self.dataset)):

            # get student and seq
            student_id, seq = self.dataset[idx]
            student_id = student_id.item()
            assert student_id == idx  # this is how the dataset is currently constructed

            S, m, meta = seq.shape

            assert seq.ndim == 3
            assert S <= self.dataset.max_sequence_len
            assert m == self.dataset.max_interaction_len
            assert meta == len(["value", "group", "pos"])
            assert (
                seq[:, 0, :1].repeat(self.dataset.max_interaction_len, axis=-1)
                == seq[:, :, 2]
            ).all  # all positions are questions
            assert (
                (seq[:1, :, 1].repeat(seq.shape[0], axis=0) == seq[:, :, 1])
            ).all  # all group idx are same

            # check that group match export heuristic
            assert seq[0, 0, 1] == q_group
            assert seq[0, 1, 1] == c_group
            assert seq[0, -1, 1] == r_group

            # loop over sequence
            for t in range(seq.shape[0]):

                # add to concept map
                question_id = seq[t, 0, 0]
                concept_ids = seq[t, :, 0][seq[t, :, 1] == c_group].tolist()
                response = seq[t, -1, 0]
                assert response in [0, 1]

                # add to concept map
                if question_id in concept_map:
                    assert concept_map[question_id] == concept_ids
                else:
                    concept_map[question_id] = concept_ids

                # add to triplets
                student_question_response_triplets.append(
                    (student_id, question_id, response)
                )
            # end for
        # end for

        # construct renumber questions map
        question_renumber_map = {
            old_idx: new_idx
            for new_idx, old_idx in enumerate(sorted(concept_map.keys()))
        }

        # construct renumber kcs map
        knowledge_renumber_map = {}
        new_idx = 0
        for old_concepts_idxs in concept_map.values():
            for old_concept_idx in old_concepts_idxs:
                if old_concept_idx not in knowledge_renumber_map:
                    knowledge_renumber_map.update({old_concept_idx: new_idx})
                    new_idx += 1

        # renumber in concept_map
        new_concept_map = {}
        for q, cs in concept_map.items():
            new_concept_map.update(
                {question_renumber_map[q]: [knowledge_renumber_map[c] for c in cs]}
            )
        concept_map = new_concept_map
        assert max([len(cs) for q, cs in concept_map.items()]) == m - 2

        # renumber in triplets
        new_student_question_response_triplets = []
        for triplet in student_question_response_triplets:
            new_student_question_response_triplets.append(
                (triplet[0], question_renumber_map[triplet[1]], triplet[2])
            )
        student_question_response_triplets = new_student_question_response_triplets

        # metadata
        num_students = len(self.dataset)
        num_questions = len(question_renumber_map)
        num_concepts = len(knowledge_renumber_map)
        num_records = len(student_question_response_triplets)

        metadata = dict(
            num_students=num_students,
            num_questions=num_questions,
            num_concepts=num_concepts,
            num_records=num_records,
            # train and test splits are seed specific
        )

        # triplets to dataframe
        df = pd.DataFrame(
            student_question_response_triplets,
            columns=["student_id", "question_id", "correct"],
        )
        assert df.shape == (num_records, 3)

        # SAVING

        logger.info(f"Saving {metadata=}")
        with open("metadata.json", "w") as f:
            json.dump(metadata, f)

        logger.info(f"Saving {concept_map=}")
        with open("concept_map.json", "w") as f:
            json.dump(concept_map, f)

        logger.info(f"Saving triplets ({df.shape=}) as csv")
        df.to_csv("triplets.csv", index=False)

        # Save triplets PER SEED

        logger.info(f"saving split triplets per seed")
        for seed in [0, 1, 2, 3, 4]:
            # load split indices for self.data_seed
            with open(f"train_val_test_split_indices_seed_{seed}.json", "r") as f:
                split_indices = json.load(f)

            df_train = df[df["student_id"].isin(split_indices["train_indices"])]
            df_val = df[df["student_id"].isin(split_indices["val_indices"])]
            df_test = df[df["student_id"].isin(split_indices["test_indices"])]

            assert len(df) == len(df_train) + len(df_val) + len(df_test)

            df_train.to_csv(f"train_triplets_seed_{seed}.csv", index=False)
            df_val.to_csv(f"val_triplets_seed_{seed}.csv", index=False)
            df_test.to_csv(f"test_triplets_seed_{seed}.csv", index=False)


def collate_fn(
    batch: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[torch.Tensor, torch.Tensor]:

    assert isinstance(batch, list)
    assert isinstance(batch[0], tuple)
    assert len(batch[0]) == 2

    users, sequences = zip(*batch)

    users_stacked = np.stack(users, axis=0)
    sequences_stacked = stack_instances_of_different_length(batch=sequences)

    B, S, m, f = sequences_stacked.shape
    assert users_stacked.shape == (B, 1)  # for now, could be extended

    return torch.tensor(users_stacked), torch.tensor(sequences_stacked)


def stack_instances_of_different_length(batch: list[np.ndarray]) -> np.ndarray:
    assert batch[0].ndim == 3
    B, S, m, f = len(batch), max([len(b) for b in batch]), *batch[0].shape[-2:]

    batch_stacked = np.full(shape=(B, S, m, f), fill_value=PADDING_VALUE)
    for i in range(len(batch)):
        batch_stacked[i, : len(batch[i])] = batch[i]
    return batch_stacked


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: str,
        user_features: list[str],
        interaction_features: list[str],
        position_type: str,
        max_num_sequences: int | None,
        max_sequence_len: int | None,
        data_path: str,
        cache_path: str | None,
    ) -> None:

        self.user_features = user_features
        self.interaction_features = interaction_features
        self.position_type = position_type
        self.max_num_sequences = max_num_sequences
        self.max_sequence_len = max_sequence_len
        self.data_path = Path(data_path)
        self.cache_path = None if cache_path is None else Path(cache_path)

        # Compute cache paths
        if self.cache_path is not None:
            hash_config = {
                "dataset": dataset,
                "features": list(interaction_features),
                "position_type": position_type,
                "max_num_sequences": max_num_sequences,
            }
            self.cache_hash = hashlib.md5(json.dumps(hash_config).encode()).hexdigest()
            self.vectorizer_cache = self.cache_path / f"v_{self.cache_hash}.pkl"
            self.dataset_cache = self.cache_path / f"d_{self.cache_hash}.pkl"
        else:
            self.vectorizer_cache = None
            self.dataset_cache = None

        # Load vectorizer from cache
        if self.vectorizer_cache is not None and self.vectorizer_cache.exists():
            self.vectorizer = Vectorizer(cache_path=self.vectorizer_cache)
        else:
            self.vectorizer = Vectorizer()
            self.vectorizer.encode_value(group="response", value=0)
            self.vectorizer.encode_value(group="response", value=1)

        # Load dataset from cache
        if self.dataset_cache is not None and self.dataset_cache.exists():
            logger.info(f"Loading dataset from {self.dataset_cache}")
            with open(self.dataset_cache, "rb") as f:
                data = pickle.load(f)
                self.max_interaction_len = data["max_interaction_len"]
                self.sequences = data["sequences"]
        else:
            self.max_interaction_len = None
            self.sequences = None

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index) -> tuple[np.ndarray, np.ndarray]:
        # TODO: include user information in user_id
        # ... = self.sequences would be tuple
        user_id = np.array([index])  # for now, could be changed to proper user_id
        sequence = self.sequences[index]

        if self.max_sequence_len is not None:
            sequence = sequence[: self.max_sequence_len]

        return user_id, sequence

    def save_to_cache(self) -> None:
        if self.dataset_cache is not None:
            logger.info(f"Saving dataset to {self.dataset_cache}")
            with open(self.dataset_cache, "wb") as f:
                data = {
                    "max_interaction_len": self.max_interaction_len,
                    "sequences": self.sequences,
                }
                pickle.dump(data, f)

        if self.vectorizer_cache is not None:
            logger.info(f"Saving vectorizer to {self.vectorizer_cache}")
            with open(self.vectorizer_cache, "wb") as f:
                pickle.dump(self.vectorizer.data, f)
