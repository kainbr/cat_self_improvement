import logging

import hydra
import numpy as np
import torch
from data import BaseDataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from pkg.data.utils import PADDING_VALUE

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

        num_train_students = self.dataset.num_train_students
        num_val_students = self.dataset.num_val_students
        num_test_students = self.dataset.num_test_students

        self.dataset: BaseDataset = hydra.utils.instantiate(config=self.dataset)

        train_indices = list(np.arange(num_train_students))
        val_indices = list(max(train_indices) + 1 + np.arange(num_val_students))
        test_indices = list(max(val_indices) + 1 + np.arange(num_test_students))

        # make subsets based on splits
        self.train_set = Subset(dataset=self.dataset, indices=train_indices)
        self.val_set = Subset(dataset=self.dataset, indices=val_indices)
        self.test_set = Subset(dataset=self.dataset, indices=test_indices)

        train_val_test_split_indices = dict(
            train_indices=[int(idx) for idx in train_indices],
            val_indices=[int(idx) for idx in val_indices],
            test_indices=[int(idx) for idx in test_indices],
        )

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
