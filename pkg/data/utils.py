import logging
import pickle
from pathlib import Path

import numpy as np

PADDING_VALUE = -1

logger = logging.getLogger(name=__name__)


class Vectorizer:
    GROUPS_KEYWORD = "_groups_"
    VALUES_KEYWORD = "_values_"

    def __init__(self, cache_path: Path | None = None):
        self._frozen = False

        if (cache_path is not None) and (cache_path.exists()):
            logger.info(f"Load Vectorizer data from {cache_path}")
            with open(cache_path, "rb") as data_file:
                self._load(pickle.load(data_file))
        else:
            if cache_path is not None:
                logger.warning(f"Init empty Vectorizer since {cache_path.exists()=}")

            # Dict with key for each group and values are a 2-tuple of encoder and decoder dicts
            # Key "_groups_" stores groups to id mappings
            self._load({self.GROUPS_KEYWORD: ({}, {}), self.VALUES_KEYWORD: ({}, {})})

    def _load(self, data: dict[str, tuple[dict, dict]]) -> None:
        self.data = data

        # Set refs for easier access
        self._groups_enc = self.data[self.GROUPS_KEYWORD][0]
        self._groups_dec = self.data[self.GROUPS_KEYWORD][1]
        self._values_enc = self.data[self.VALUES_KEYWORD][0]
        self._values_dec = self.data[self.VALUES_KEYWORD][1]

    def freeze(self) -> None:
        self._frozen = True

    def decode_group(self, value: int) -> int | str:
        return self._groups_dec[value]

    def decode_value(self, value: int) -> tuple[str, int | str]:
        return self._values_dec[value]

    def encode_group(self, group: int | str) -> int:
        if group not in self._groups_enc:
            if self._frozen:
                raise Exception("Vectorizer is frozen. Cannot add new groups.")

            idx = len(self._groups_enc.values())
            self._groups_enc[group], self._groups_dec[idx] = idx, group

        return self._groups_enc[group]

    def encode_value(self, group: int | str, value: int | str) -> int:
        # Check if group is encoded or encode otherwise
        if group not in self._groups_enc:
            self.encode_group(group=group)

        if (group, value) not in self._values_enc:
            if isinstance(value, np.int64):
                value = value.item()

            if self._frozen:
                raise Exception("Vectorizer is frozen. Cannot add new values.")

            idx = len(self._values_enc.values())
            self._values_enc[(group, value)] = idx
            self._values_dec[idx] = (group, value)

        return self._values_enc[(group, value)]

    def get_indices_of_group(self, group: int | str) -> list[int]:
        if group not in self._groups_enc:
            raise KeyError

        return [v for k, v in self._values_enc.items() if k[0] == group]

    def get_size(self, group: str | None = None) -> int:
        if group is None:
            return len(self._values_enc)
        else:
            return len(self.get_indices_of_group(group))

    def get_groups(self) -> list[str]:
        return sorted(self._groups_enc)

    def get_num_groups(self) -> int:
        return len(self._groups_enc)

    def save(self, cache_path: Path) -> None:
        logger.info(f"Save Vectorizer data to {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(self.data, f)
