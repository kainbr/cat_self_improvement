import logging
from logging import Logger

import hydra
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.nn import functional as F

from pkg.data.utils import PADDING_VALUE
from pkg.model.attention import BaseMultiheadAttention
from pkg.model.stedu.ape import AbsolutePositionEmbedding
from pkg.utils.logging import logger_warn_once

logger: Logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        attn: DictConfig,
    ):
        super().__init__()

        self.attn = attn
        self.d_model = d_model
        self.nhead = nhead

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    attn=attn,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        src_positions: Tensor | None = None,
    ) -> Tensor:

        B, S, f = src.shape
        assert f == self.d_model
        assert (src_mask is None) or (src_mask.shape == (S, S))
        assert (src_key_padding_mask is None) or (src_key_padding_mask.shape == (B, S))
        assert (src_positions is None) or (src_positions.shape == (B, S))

        nan_mask = ~src_key_padding_mask.all(dim=1)

        # convert padding_mask to float
        if src_key_padding_mask is not None:
            src_key_padding_mask = torch.zeros_like(
                src_key_padding_mask, dtype=src.dtype
            ).masked_fill_(src_key_padding_mask, float("-inf"))

        if (~nan_mask).any():
            logger_warn_once(logger=logger, msg=f"Applying `nan_mask` in {self}")
            out = src

            src = src[nan_mask]
            src_key_padding_mask = src_key_padding_mask[nan_mask]
            src_positions = src_positions[nan_mask]

        for layer in self.layers:
            src, _ = layer.forward(
                src=src,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                src_positions=src_positions,
            )

        if (~nan_mask).any():
            out[nan_mask] = src
        else:
            out = src

        return out


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        attn: DictConfig,
    ):
        super().__init__()

        self.attn = attn
        self.d_model = d_model
        self.nhead = nhead

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    attn=attn,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        tgt: Tensor,
        mem: Tensor,
        tgt_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        tgt_positions: Tensor | None = None,
        mem_mask: Tensor | None = None,
        mem_key_padding_mask: Tensor | None = None,
        mem_positions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:

        B, S_tgt, f = tgt.shape
        assert f == self.d_model
        assert (tgt_mask is None) or tgt_mask.shape == (S_tgt, S_tgt)
        assert (tgt_key_padding_mask is None) or tgt_key_padding_mask.shape == (
            B,
            S_tgt,
        )
        assert (tgt_positions is None) or tgt_positions.shape == (B, S_tgt)

        B, S_mem, f = mem.shape
        assert f == self.d_model
        assert (mem_mask is None) or mem_mask.shape == (S_tgt, S_mem)
        assert (mem_key_padding_mask is None) or mem_key_padding_mask.shape == (
            B,
            S_mem,
        )
        assert (mem_positions is None) or mem_positions.shape == (B, S_mem)

        nan_mask = ~tgt_key_padding_mask.all(dim=1)

        # convert padding_masks to float
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = torch.zeros_like(
                tgt_key_padding_mask, dtype=tgt.dtype
            ).masked_fill_(tgt_key_padding_mask, float("-inf"))
        if mem_key_padding_mask is not None:
            mem_key_padding_mask = torch.zeros_like(
                mem_key_padding_mask, dtype=tgt.dtype
            ).masked_fill_(mem_key_padding_mask, float("-inf"))

        if (~nan_mask).any():
            logger_warn_once(logger=logger, msg=f"Applying `nan_mask` in {self}")
            out = tgt

            tgt = tgt[nan_mask]
            if tgt_mask is not None:
                raise NotImplementedError  # should work in any case, test ...
            tgt_key_padding_mask = tgt_key_padding_mask[nan_mask]
            tgt_positions = tgt_positions[nan_mask]

            mem = mem[nan_mask]
            if mem_mask is not None:
                raise NotImplementedError  # should work in any case, test ...
            mem_key_padding_mask = mem_key_padding_mask[nan_mask]
            mem_positions = mem_positions[nan_mask]

        for layer in self.layers:
            tgt, attn_scores = layer.forward(
                tgt=tgt,
                mem=mem,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_positions=tgt_positions,
                mem_mask=mem_mask,
                mem_key_padding_mask=mem_key_padding_mask,
                mem_positions=mem_positions,
            )

        if (~nan_mask).any():
            out[nan_mask] = tgt
            out_attn_scores = torch.full(
                [B, S_tgt, S_mem], fill_value=-torch.inf, device=attn_scores.device
            )
            out_attn_scores[nan_mask] = attn_scores
        else:
            out = tgt
            out_attn_scores = attn_scores

        return out, out_attn_scores


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        attn: DictConfig,
    ):
        super().__init__()

        self.self_attn: BaseMultiheadAttention = hydra.utils.instantiate(
            attn,
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        src_positions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        self_attn_output, self_attn_scores = self._sa_block(
            src=src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            src_positions=src_positions,
        )
        src = src + self_attn_output
        src = self.norm1(src)

        src = src + self._ff_block(x=src)
        src = self.norm2(src)

        return src, self_attn_scores

    def _sa_block(
        self,
        src: Tensor,
        src_mask: Tensor | None,
        src_key_padding_mask: Tensor | None,
        src_positions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        src, attn_scores = self.self_attn.forward(
            query=src,
            key=src,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            query_position=src_positions,
            key_position=src_positions,
        )
        return self.dropout1(src), attn_scores

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        attn: DictConfig,
    ):
        super().__init__()

        self.self_attn: BaseMultiheadAttention = hydra.utils.instantiate(
            attn,
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
        )
        self.cross_attn: BaseMultiheadAttention = hydra.utils.instantiate(
            attn,
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(
        self,
        tgt: Tensor,
        mem: Tensor,
        tgt_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        tgt_positions: Tensor | None = None,
        mem_mask: Tensor | None = None,
        mem_key_padding_mask: Tensor | None = None,
        mem_positions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:

        self_attn_output, _ = self._sa_block(
            tgt=tgt,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_positions=tgt_positions,
        )
        tgt = tgt + self_attn_output
        tgt = self.norm1(tgt)

        cross_attn_output, cross_attn_scores = self._ca_block(
            tgt=tgt,
            mem=mem,
            mem_mask=mem_mask,
            mem_key_padding_mask=mem_key_padding_mask,
            tgt_positions=tgt_positions,
            mem_positions=mem_positions,
        )
        tgt = tgt + cross_attn_output
        tgt = self.norm2(tgt)

        tgt = tgt + self._ff_block(x=tgt)
        tgt = self.norm3(tgt)

        return tgt, cross_attn_scores

    def _sa_block(
        self,
        tgt: Tensor,
        tgt_mask: Tensor | None,
        tgt_key_padding_mask: Tensor | None,
        tgt_positions: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        tgt, attn_scores = self.self_attn.forward(
            query=tgt,
            key=tgt,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            query_position=tgt_positions,
            key_position=tgt_positions,
        )
        return self.dropout1(tgt), attn_scores

    def _ca_block(
        self,
        tgt: Tensor,
        mem: Tensor,
        mem_mask: Tensor | None,
        mem_key_padding_mask: Tensor | None,
        tgt_positions: Tensor | None,
        mem_positions: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        tgt, attn_scores = self.cross_attn.forward(
            query=tgt,
            key=mem,
            value=mem,
            attn_mask=mem_mask,
            key_padding_mask=mem_key_padding_mask,
            query_position=tgt_positions,
            key_position=mem_positions,
        )
        return self.dropout2(tgt), attn_scores

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class BaseTransformer(nn.Module):

    def __init__(
        self,
        d_model: int,
        encoder: DictConfig,
        decoder: DictConfig,
        attn: DictConfig,
        max_position_idx: int,
        share_enc_dec_params: bool = True,
        share_all: bool = False,
    ) -> None:
        super().__init__()

        self.PADDING_VALUE = PADDING_VALUE
        self.d_model = d_model
        self.max_position_idx = max_position_idx
        self.share_params = share_enc_dec_params
        self.share_all = share_all

        # TODO: Refactor this, e.g. rotary does not need max_pos_idx
        attn.max_len = int(self.max_position_idx)

        self.use_ape: bool = attn.attn == "standard"
        if self.use_ape:
            self.ape = AbsolutePositionEmbedding(
                d_model=self.d_model, max_len=int(self.max_position_idx)
            )

        self.encoder: Encoder = hydra.utils.instantiate(
            encoder, d_model=d_model, attn=attn
        )
        self.decoder: Decoder = hydra.utils.instantiate(
            decoder, d_model=d_model, attn=attn
        )

        if self.share_params:
            num_enc_layers = len(self.encoder.layers)
            num_dec_layers = len(self.decoder.layers)
            assert num_enc_layers == num_dec_layers
            # TODO: assert that cfg is aligned, e.g. same heads etc.

            # TODO: make this a method?
            # copy encoder layers to decoder
            for i in range(num_enc_layers):
                self.decoder.layers[i].self_attn = self.encoder.layers[i].self_attn
                self.decoder.layers[i].linear1 = self.encoder.layers[i].linear1
                self.decoder.layers[i].linear2 = self.encoder.layers[i].linear2
                if self.share_all:
                    self.decoder.layers[i].norm1 = self.encoder.layers[i].norm1
                    self.decoder.layers[i].norm2 = self.encoder.layers[i].norm2

    def forward(
        self,
        x_given: Tensor,
        key_padding_mask_given: Tensor,
        positions_given: Tensor,
        x_partial: Tensor,
        key_padding_mask_partial: Tensor,
        positions_partial: Tensor,
    ) -> Tensor:
        raise NotImplementedError

    def get_updated_positions(
        self,
        positions: Tensor,
        mode: str,
        value: int = 0,
        offset: int = 0,
    ) -> tuple[Tensor, int]:

        B, S, m = positions.shape
        assert torch.equal(positions[..., 0].unsqueeze(-1).repeat(1, 1, m), positions)
        squeezed_positions = positions[..., 0]
        assert B, S == squeezed_positions.shape

        match mode:
            case "arange":
                new_squeezed_positions = (
                    torch.ones_like(squeezed_positions).cumsum(dim=1) + offset
                )
            case "constant":
                new_squeezed_positions = torch.full_like(
                    squeezed_positions, fill_value=value
                )
            case _:
                raise ValueError(f"{mode} is invalid")

        new_positions = new_squeezed_positions.unsqueeze(dim=-1).repeat(1, 1, m)
        max_position = new_positions.max().item()
        assert new_positions.shape == (B, S, m)

        return new_positions, max_position
