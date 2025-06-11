import logging
from collections import namedtuple
from logging import Logger

import numpy as np
import torch

CacheEntry = namedtuple("CacheEntry", ["score", "seq"])

logger: Logger = logging.getLogger(__name__)


class ResponseSequenceCache:

    def __init__(self, max_sequences_per_entries: int | None = None):
        self.max_sequences_per_entries = max_sequences_per_entries
        self.cache: dict[int, list[CacheEntry]] = {}

    def add_response_seqs(
        self,
        users: torch.Tensor,
        scores: torch.Tensor,
        response_seqs: torch.Tensor,
    ) -> None:
        for user, score, response_seq in zip(users, scores, response_seqs):
            user = user.item()
            score = score.item()
            response_seq = response_seq.cpu().detach()

            if user not in self.cache:
                self.cache[user] = [CacheEntry(score, response_seq)]
            else:

                if self._is_seq_in_user(user=user, seq=response_seq):
                    # logger.info(f"Duplicated sample for {user=}")
                    continue

                cache_is_full = (self.max_sequences_per_entries is not None) and (
                    len(self.cache[user]) >= self.max_sequences_per_entries
                )

                if not cache_is_full:
                    self.cache[user].append(CacheEntry(score, response_seq))
                else:
                    # only add, if score is better, assumes higher score is better
                    scores = [entry.score for entry in self.cache[user]]
                    min_score, min_score_idx = np.min(scores), np.argmin(scores)

                    if min_score < score:
                        self.cache[user].pop(min_score_idx)
                        self.cache[user].append(CacheEntry(score, response_seq))

    def _is_seq_in_user(self, user: int, seq: torch.Tensor):
        entries = self.cache[user]

        for entry in entries:
            if torch.equal(seq, entry.seq):
                return True

        return False

    def argmin(self) -> None:
        for user, entries in self.cache.items():
            self.cache[user] = [entries[np.argmax([entry.score for entry in entries])]]

    def get_random_response_seqs(
        self, users: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = []
        response_seqs = []
        for user in users:
            user = user.item()
            entries = self.cache[user]

            if len(entries) == 1:
                sampled_idx = 0
            else:
                sampled_idx: int = torch.randint(len(entries), size=(1,)).item()

            scores.append(entries[sampled_idx].score)
            response_seqs.append(entries[sampled_idx].seq)

        stacked_scores = torch.tensor(scores).unsqueeze(dim=-1).to(users.device)
        stacked_response_seqs = torch.stack(response_seqs, dim=0).to(users.device)

        return stacked_scores, stacked_response_seqs

    def log(self) -> None:
        logger.info(f"number of cache entries {len(self.cache)}")
        for user in self.cache.keys():
            if user <= 10:
                _new = self.cache[user][0]
                logger.info(f"{user=} {_new.score} with {_new.seq}")
        # Log the top-k items that were selected
        counts = np.unique(
            np.array([v[0].seq.tolist() for v in self.cache.values()]).flatten(),
            return_counts=True,
        )
        a = counts[1].argsort(axis=0)[::-1]
        b = {k: v for k, v in zip(counts[0][a][:10], counts[1][a][:10])}
        logger.info(f"{b}")
