import hashlib
import itertools
import json
import logging
import pickle
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

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
        train_splits = [split for (i, split) in enumerate(idx_splits) if i in train_idx]
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

    @property
    def train_loader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=self.num_workers,
        )

    @property
    def train_eval_loader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size_eval,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
            num_workers=self.num_workers,
        )

    @property
    def val_loader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size_eval,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
            num_workers=self.num_workers,
        )

    @property
    def test_loader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size_eval,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
            num_workers=self.num_workers,
        )


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


@hydra.main(config_path="../config/data", config_name="base", version_base="1.3")
def main(config: DictConfig):

    data: Data = hydra.utils.instantiate(config=config)

    for batch in data.train_loader:
        batch

    users, sequences = batch

    print("debugging ...")


if __name__ == "__main__":
    main()
