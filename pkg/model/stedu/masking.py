import torch
from torch import nn

from pkg.data.utils import PADDING_VALUE, Vectorizer


class Masking(nn.Module):
    def __init__(self, vectorizer: Vectorizer):
        super().__init__()
        self.vectorizer = vectorizer
        self.PADDING_VALUE = PADDING_VALUE

    def forward(self, groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class ResponsesMasking(Masking):

    def forward(self, groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        response_mask = groups == self.vectorizer.encode_group("response")
        target_mask = response_mask
        query_mask = response_mask

        return query_mask, target_mask


class ResponsesAndQuestionMasking(Masking):

    def forward(
        self, groups: torch.Tensor, ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        response_mask = groups == self.vectorizer.encode_group("response")
        target_mask = response_mask

        assert 0.0 < ratio <= 1.0
        random_noise = torch.rand_like(groups.float())
        random_mask = random_noise < ratio  # noisy
        question_mask = groups == self.vectorizer.encode_group("question")
        question_mask = torch.logical_and(question_mask, random_mask)

        query_mask = torch.logical_or(response_mask, question_mask)

        return query_mask, target_mask


class ResponsesAndAnswerMasking(Masking):

    def forward(self, groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        response_mask = groups == self.vectorizer.encode_group("response")
        target_mask = response_mask

        if "answer" in self.vectorizer.get_groups():
            answer_mask = groups == self.vectorizer.encode_group("answer")
            query_mask = torch.logical_or(response_mask, answer_mask)
        else:
            query_mask = response_mask

        return query_mask, target_mask
