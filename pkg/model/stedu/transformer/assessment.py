from torch import Tensor

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

        # TODO: We could randomly assign positions, ATTENTION: with current codebase arange for positions_partial leads to leakage
        # For now, we just use the question idx
        # -> Assumption: positions == questions, ATTENTION: this is assumed elsewhere in the code!
        DO_UPDATE_POSITIONS = False
        if DO_UPDATE_POSITIONS:
            positions_given, _ = self.get_updated_positions(
                positions=positions_given, mode="arange"
            )
            positions_partial, _ = self.get_updated_positions(
                positions=positions_partial,
                mode="constant",
                value=0,
            )

        # TODO: check and implement: do not overwrite padded positions (should work in any case)
        # currently we overwrite `-1`s ...
        positions_partial[positions_partial == self.PADDING_VALUE] = 0

        if self.use_ape:
            x_given = self.ape.forward(x_given, positions_given)
            x_partial = self.ape.forward(x_partial, positions_partial)

        # Get correctly shaped src for encoder
        src, src_key_padding_mask, src_positions = self.get_src(
            x_given=x_given,
            key_padding_mask_given=key_padding_mask_given,
            positions_given=positions_given,
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
            S_partial=S_partial,
            src_key_padding_mask=src_key_padding_mask,
            src_positions=src_positions,
        )

        # Get correctly shaped tgt for decoder
        tgt, tgt_key_padding_mask, tgt_positions = self.get_tgt(
            x_partial=x_partial,
            key_padding_mask_partial=key_padding_mask_partial,
            positions_partial=positions_partial,
        )

        mem_mask, tgt_mask = None, None

        # Decoder
        out, _ = self.decoder.forward(
            mem=mem,
            tgt=tgt,
            mem_mask=mem_mask,
            mem_key_padding_mask=mem_key_padding_mask,
            mem_positions=mem_positions,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_positions=tgt_positions,
        )

        # Get correctly shaped output for classifier
        out = out.reshape(-1, S_partial, m, f)

        return out

    def get_src(
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

    def get_tgt(
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
        S_partial: int,
        src_key_padding_mask: Tensor | None = None,
        src_positions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:

        B_given, Sm_given, f_given = src.shape
        assert src_key_padding_mask.shape == (B_given, Sm_given)
        assert src_positions.shape == (B_given, Sm_given)

        src = src.repeat_interleave(S_partial, dim=0)
        src_key_padding_mask = src_key_padding_mask.repeat_interleave(S_partial, dim=0)
        src_positions = src_positions.repeat_interleave(S_partial, dim=0)

        return src, src_key_padding_mask, src_positions
