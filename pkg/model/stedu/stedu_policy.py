import logging
import math
from logging import Logger

import hydra
import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from pkg.data.utils import PADDING_VALUE, Vectorizer
from pkg.model.stedu.classifier import Classifier
from pkg.model.stedu.embedding import Embedding
from pkg.model.stedu.gather_utils import (
    gather_and_split_sequences,
    get_key_padding_mask,
)
from pkg.model.stedu.masking import Masking
from pkg.model.stedu.transformer import BaseTransformer

logger: Logger = logging.getLogger(__name__)


class Model(nn.Module):
    def __init__(
        self,
        d_model: int,
        vectorizer: Vectorizer,
        max_interaction_len: int,
        max_position_idx: int,
        embedding: DictConfig,
        transformer: DictConfig,
        encoder: DictConfig,
        decoder: DictConfig,
        attention: DictConfig,
        classifier: DictConfig,
        masking: DictConfig,
    ):
        super().__init__()

        self.name = "policy_model"

        self.PADDING_VALUE = PADDING_VALUE
        self.d_model = d_model
        self.vectorizer = vectorizer
        self.max_interaction_len = max_interaction_len
        self.max_position_idx = max_position_idx

        self.bos_token_emb = nn.Embedding(
            num_embeddings=self.max_interaction_len, embedding_dim=self.d_model
        )

        self.embedding: Embedding = hydra.utils.instantiate(
            embedding, d_model=self.d_model, vectorizer=self.vectorizer
        )

        self.masking: Masking = hydra.utils.instantiate(
            masking, vectorizer=self.vectorizer
        )

        self.transformer: BaseTransformer = hydra.utils.instantiate(
            transformer,
            d_model=d_model,
            encoder=encoder,
            decoder=decoder,
            attn=attention,
            max_position_idx=max_position_idx,
        )

        self.classifier: Classifier = hydra.utils.instantiate(
            classifier, d_in=self.d_model, d_out=1
        )

        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        x: Tensor,
        given_positions: Tensor | None,
        sampling_scheme: str = "argmax",
        temperature: float = 1.0,
    ) -> tuple[Tensor, dict]:

        B, S, m, f = x.shape
        assert x.dim() == 4
        assert m == self.max_interaction_len
        assert f == 3

        if given_positions is None:
            # start with x_given that is empty along sequence dim
            given_positions = x[:, :0, 0, 2]
            assert given_positions.shape == (B, 0)

        num_given_positions = given_positions.shape[1]
        x_given, x_partial = gather_and_split_sequences(
            x=x, given_positions=given_positions
        )
        # Combine
        x_partial = torch.cat([x_given, x_partial], dim=1)

        # TODO: refactor BOS handling (BOS needed especially for num_given_positions == 0)
        # TODO: group set to 0, currently not used, fix for robustness
        bos_placeholder = torch.zeros_like(x_partial[:, :1])
        x_given = torch.cat(
            [bos_placeholder, x_given], dim=1
        )  # sets position (and group) to `0`, since not overwritten

        values_given, values_partial = x_given[..., 0], x_partial[..., 0]
        groups_given, groups_partial = x_given[..., 1], x_partial[..., 1]
        positions_given, positions_partial = x_given[..., 2], x_partial[..., 2]

        # Compute key padding masks
        key_padding_mask_given = get_key_padding_mask(x_given)
        key_padding_mask_partial = get_key_padding_mask(x_partial)

        # Embed tokens
        values_given_emb, query_emb = self.embedding.forward(values=values_given)
        values_partial_emb, _ = self.embedding.forward(values=values_partial)

        # TODO: refactor BOS handling (BOS needed especially for num_given_positions == 0)
        assert values_given_emb.shape == (B, num_given_positions + 1, m, self.d_model)
        bos_token_ = (
            torch.arange(m, dtype=torch.int64, device=values_given_emb.device)
            .view([1, 1, m])
            .repeat([B, 1, 1])
        )
        bos_token = self.bos_token_emb(bos_token_)
        values_given_emb[:, :1, :] = bos_token

        # Apply masking strategy
        query_mask, targets_mask = self.masking.forward(groups=groups_partial)
        # masks responses in partial, targets_mask, e.g. positions of responses are not needed for policy ...

        # Swap query embedding at given indices
        values_partial_emb[query_mask] = query_emb

        out_scores = self.transformer.forward(
            x_given=values_given_emb,
            key_padding_mask_given=key_padding_mask_given,
            positions_given=positions_given,
            x_partial=values_partial_emb,
            key_padding_mask_partial=key_padding_mask_partial,
            positions_partial=positions_partial,
        )
        # transformer outputs attention scores (mean per head) from given to partial (question to question only)

        # split into part where we have labels (given_positions) for training and next_prediction logits
        out_scores_for_loss = out_scores[:, :-1, :]
        out_scores_for_prediction = out_scores[:, -1, :]
        assert out_scores_for_loss.shape == (B, num_given_positions, S)

        # get argmax of logit for next position prediction (for which we do not have labels)
        # will be used in sampling ...
        if sampling_scheme == "argmax":
            indices = out_scores_for_prediction.argmax(dim=-1).unsqueeze(-1)
        elif sampling_scheme == "prob":
            indices = torch.multinomial(
                torch.nn.functional.softmax(
                    out_scores_for_prediction / temperature, dim=-1
                ),
                1,
            )
        else:
            raise NotImplementedError

        # assuming all positions in `m` dimension are same
        assert torch.equal(
            positions_partial[..., 0].unsqueeze(-1).repeat(1, 1, m), positions_partial
        )
        positions = positions_partial[..., 0]
        assert positions.dim() == 2
        assert positions.shape == out_scores_for_prediction.shape
        next_given_position = torch.take_along_dim(
            input=positions,
            indices=indices,
            dim=-1,
        )

        if num_given_positions == 0:
            # for num_given_positions == 0, we can calculate no loss and simply sample next position
            loss = torch.zeros(1, device=next_given_position.device)
            return (
                loss,
                {
                    "targets": None,
                    "predictions": None,
                    "next": next_given_position,
                },
            )

        # else (for num_given_positions != 0): calc loss, get targets and predictions below and return

        # TODO: refactor classifier below, make more readable

        # targets as per task construction (this works, because we have constructed partial to be targets in proper order + other concatenated)
        # ATTENTION: can't use PE arange or alike, otherwise cheating
        targets = torch.zeros_like(out_scores_for_loss)
        targets[..., :num_given_positions] = torch.diag(
            torch.ones(num_given_positions, dtype=targets.dtype, device=targets.device)
        ).unsqueeze(0)

        # handle -inf in attention scores, overwrite with large negative values (see .log() below)
        # (otherwise loss function will result in NaNs)
        out_scores_for_loss[out_scores_for_loss == -torch.inf] = math.log(1e-13)

        # loss
        # should work out of the box, since CE is: - \sum_{y \in \mathcal{Y}} p(y) * log(q(y))
        # p(y) is one-hot encoded (q(y) are pred probs) and thus negative classes are not considered ...
        loss = self.loss_fn(
            out_scores_for_loss.transpose(1, 2), targets.transpose(1, 2)
        )
        loss = loss.mean()

        if self.training:
            return loss, {}
        else:
            return (
                loss,
                {
                    "targets": targets.argmax(dim=-1),
                    "predictions": out_scores_for_loss.argmax(dim=-1),
                    "next": next_given_position,
                },
            )
