from omegaconf import DictConfig
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pkg.model.stedu.transformer import BaseTransformer


class Transformer(BaseTransformer):

    def forward(
        self,
        x_given: Tensor,
        key_padding_mask_given: Tensor,
        positions_given: Tensor,
        x_partial: Tensor,
        key_padding_mask_partial: Tensor,
        positions_partial: Tensor,
    ) -> Tensor:

        B, S_given, m, f = x_given.shape
        B_, S_partial, m_, f_ = x_partial.shape
        assert (B, m, f) == (B_, m_, f_)
        if S_given == 0:
            raise ValueError

        # TODO: We could randomly assign positions, ATTENTION: with current codebase arange for positions_partial leads to leakage
        # For now, we just use the question idx
        # -> Assumption: positions == questions, ATTENTION: this is assumed elsewhere in the code!
        DO_UPDATE_POSITIONS = False
        if DO_UPDATE_POSITIONS:
            positions_given, _ = self.get_updated_positions(
                positions=positions_given, mode="arange", offset=1
            )
            positions_partial, _ = self.get_updated_positions(
                positions=positions_partial,
                mode="constant",
                value=0,
            )  # ATTENTION: does not allow for discrimination in mem

        # TODO: check and implement: do not overwrite padded positions (should work in any case)
        # currently we overwrite `-1`s ...
        positions_partial[positions_partial == self.PADDING_VALUE] = 0

        if self.use_ape:
            x_given = self.ape.forward(x_given, positions_given)
            x_partial = self.ape.forward(x_partial, positions_partial)

        # Get correctly shaped src for encoder
        src, src_key_padding_mask, src_positions = self.get_src(
            x_partial=x_partial,
            key_padding_mask_partial=key_padding_mask_partial,
            positions_partial=positions_partial,
        )

        # Encoder
        src = self.encoder.forward(
            src=src,
            src_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            src_positions=src_positions,
        )

        # Get correctly shaped mem from encoder output
        mem, mem_key_padding_mask, mem_positions = self.get_mem(
            src=src,
            src_key_padding_mask=src_key_padding_mask,
            src_positions=positions_partial,
        )

        # Get correctly shaped tgt for decoder
        tgt, tgt_key_padding_mask, tgt_positions = self.get_tgt(
            x_given=x_given,
            key_padding_mask_given=key_padding_mask_given,
            positions_given=positions_given,
        )

        mem_mask, tgt_mask = self.generate_masks(
            S_given=S_given, S_partial=S_partial, m=m, device=mem.device
        )

        # Decoder
        _, attn_scores = self.decoder.forward(
            mem=mem,
            tgt=tgt,
            mem_mask=mem_mask,
            mem_key_padding_mask=mem_key_padding_mask,
            mem_positions=mem_positions,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_positions=tgt_positions,
        )

        # Get attention scores of question positions
        question_to_question_scores = attn_scores[:, ::m, ::m]
        assert question_to_question_scores.shape == (B, S_given, S_partial)

        return question_to_question_scores

    def get_tgt(
        self,
        x_given: Tensor,
        key_padding_mask_given: Tensor | None = None,
        positions_given: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        B, S, m, f = x_given.shape
        assert key_padding_mask_given.shape == (B, S, m)
        assert positions_given.shape == (B, S, m)

        x_given = x_given.view(B, S * m, f)
        key_padding_mask_given = key_padding_mask_given.view(B, S * m)
        positions_given = positions_given.reshape(B, S * m)

        return x_given, key_padding_mask_given, positions_given

    def get_src(
        self,
        x_partial: Tensor,
        key_padding_mask_partial: Tensor | None = None,
        positions_partial: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        B, S, m, f = x_partial.shape
        assert key_padding_mask_partial.shape == (B, S, m)
        assert positions_partial.shape == (B, S, m)

        x_partial = x_partial.view(B * S, m, f)
        key_padding_mask_partial = key_padding_mask_partial.view(B * S, m)
        positions_partial = positions_partial.reshape(B * S, m)

        return x_partial, key_padding_mask_partial, positions_partial

    def get_mem(
        self,
        src: Tensor,
        src_key_padding_mask: Tensor | None = None,
        src_positions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:

        B_partial, S_partial, m_partial = src_positions.shape
        BS_partial, m_partial, f_partial = src.shape
        assert src_key_padding_mask.shape == (BS_partial, m_partial)

        src = src.view(B_partial, S_partial * m_partial, f_partial)
        src_key_padding_mask = src_key_padding_mask.view(
            B_partial, S_partial * m_partial
        )
        src_positions = src_positions.reshape(B_partial, S_partial * m_partial)

        return src, src_key_padding_mask, src_positions

    def generate_masks(
        self, S_given: int, S_partial: int, m: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        mem_mask_given = torch.tril(
            torch.full((S_given, S_given), True, dtype=torch.bool, device=device),
            diagonal=-1,
        )
        mem_mask_partial = torch.full(
            (S_given, S_partial - S_given),
            False,
            dtype=torch.bool,
            device=device,
        )
        mem_mask = torch.cat([mem_mask_given, mem_mask_partial], dim=-1)
        mem_mask = F._canonical_mask(
            mask=mem_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=torch.float32,
            check_other=False,
        )
        mem_mask = mem_mask.repeat_interleave(repeats=m, dim=0)
        mem_mask = mem_mask.repeat_interleave(repeats=m, dim=1)
        assert mem_mask.shape == (S_given * m, S_partial * m)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            sz=S_given, device=device
        )
        tgt_mask = tgt_mask.repeat_interleave(repeats=m, dim=0)
        tgt_mask = tgt_mask.repeat_interleave(repeats=m, dim=1)
        assert tgt_mask.shape == (S_given * m, S_given * m)

        return mem_mask, tgt_mask
