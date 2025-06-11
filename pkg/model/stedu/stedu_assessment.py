import logging
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

        self.name = "assessment_model"

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
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        x: Tensor,
        given_positions: Tensor | None = None,
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

        # Split input sequences in encoder and decoder part
        x_given, x_partial = gather_and_split_sequences(
            x=x, given_positions=given_positions
        )
        num_given_positions = given_positions.shape[1]

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

        # Swap query embedding at given indices
        values_partial_emb[query_mask] = query_emb

        out = self.transformer.forward(
            x_given=values_given_emb,
            key_padding_mask_given=key_padding_mask_given,
            positions_given=positions_given,
            x_partial=values_partial_emb,
            key_padding_mask_partial=key_padding_mask_partial,
            positions_partial=positions_partial,
        )

        # classifier
        logits = self.classifier.forward(out).squeeze()
        targets = values_partial
        assert ~torch.isnan(logits).any()  # there should be no nans up to this point

        # renaming for refactor
        mask = targets_mask

        # response
        targets_response = targets.clone().float()  # for BCE
        logits_response = logits.clone()
        mask_response = mask.clone()  # and group == "response"
        logits_response[~mask_response] = torch.nan

        loss_response = self.loss_fn.forward(logits_response, targets_response)
        assert torch.isnan(loss_response[~mask_response]).all()
        assert ~torch.isnan(loss_response[mask_response]).any()
        mean_loss_response = loss_response.nanmean()
        loss_per_seq = loss_response.nanmean(dim=-1).nanmean(dim=-1)

        corr_response = torch.eq(
            logits_response > 0, targets_response
        )  # eq with nan should be False
        num_corr_response = corr_response.sum()
        num_response = mask_response.sum()
        acc_response = num_corr_response / num_response

        num_corr_response_per_seq = corr_response.sum(dim=-1).sum(dim=-1)
        num_response_per_seq = mask_response.sum(dim=-1).sum(dim=-1)
        acc_response_per_seq = num_corr_response_per_seq / num_response_per_seq

        targets_response_per_seq = targets_response[..., -1].long()
        logits_response_per_seq = logits_response[..., -1]

        # get most_uncertain position (for sampling ...)

        # assuming positions are all the same per interaction
        positions = positions_partial[..., 0]
        assert logits_response_per_seq.shape == positions.shape
        logits_response_per_seq_ = logits_response_per_seq.nan_to_num(nan=-torch.inf)

        # either sample or take argmax (= most uncertain, notice the "-")
        uncertainty_score = -logits_response_per_seq_.abs()
        if sampling_scheme == "prob":
            most_uncertain_idx = torch.multinomial(
                torch.nn.functional.softmax(uncertainty_score / temperature, dim=-1),
                1,
            )
        elif sampling_scheme == "argmax":
            most_uncertain_idx = uncertainty_score.argmax(dim=-1).unsqueeze(dim=-1)
        else:
            raise NotImplementedError

        # assuming question == positions
        most_uncertain_position = torch.take_along_dim(
            input=positions, indices=most_uncertain_idx, dim=-1
        )

        # loss + return
        return mean_loss_response, {
            "targets_response_per_seq": targets_response_per_seq,
            "logits_response_per_seq": logits_response_per_seq,
            "loss_per_seq": loss_per_seq.unsqueeze(dim=-1),
            "acc_response_per_seq": acc_response_per_seq.unsqueeze(dim=-1),
            "acc_response": acc_response,
            "num_response": num_response,
            "most_uncertain_position": most_uncertain_position,
        }
