import torch
from torch import nn

from pkg.data.utils import Vectorizer


class Embedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        vectorizer: Vectorizer,
        initialize_question_to_zero: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.vectorizer = vectorizer
        self.initialize_question_to_zero = initialize_question_to_zero

        self.value_emb = nn.Embedding(
            num_embeddings=self.vectorizer.get_size() + 1,
            embedding_dim=self.d_model,
            padding_idx=0,
        )

        self.query_emb = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.d_model,
        )

        if self.initialize_question_to_zero:
            self._reset_question_params_to_zero()

    def _reset_question_params_to_zero(self) -> None:
        question_indices = (
            torch.tensor(self.vectorizer.get_indices_of_group("question")) + 1
        )  # TODO: refactor, dangerous +1, see forward below
        for idx in question_indices:
            torch.nn.init.zeros_(self.value_emb.weight[idx])

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Adding `+1` to move padding value `-1` to `0`.
        # Index `0` is reserved for as padding idx in the embedding.
        values_emb = self.value_emb(values + 1)
        query_emb = self.query_emb.weight[0]

        assert values_emb.shape == (*values.shape, self.d_model)
        assert query_emb.shape == (self.d_model,)

        return values_emb, query_emb
