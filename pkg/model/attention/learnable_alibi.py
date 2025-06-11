import logging
import math
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter

logger: logging.Logger = logging.getLogger(__name__)


def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def inverse_softplus(x):
    return x + torch.log(-torch.expm1(-x))


class LearnableALiBiAttention(Module):

    def __init__(self, num_heads: int, use_multi_theta: bool = True) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.use_multi_theta = use_multi_theta

        assert use_multi_theta == True

        self.thetas = Parameter(
            inverse_softplus(
                torch.tensor(get_slopes(self.num_heads)).reshape(self.num_heads, 1, 1)
            )
        )
        # self.register_buffer("thetas", thetas)

        # logger.info(f"{self.thetas.squeeze()=}")

    def forward(
        self,
        attn_output_weights: torch.Tensor,
        attn_mask: torch.Tensor | None,
        positions: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size_times_num_head, S, _ = attn_output_weights.shape
        assert (batch_size_times_num_head % self.num_heads) == 0
        assert attn_mask is not None

        if positions is not None:
            # we want to use block attention
            seq_ = positions.unsqueeze(1).expand(-1, S, -1)
            do_not_attend = (seq_ - seq_.transpose(-2, -1)).repeat_interleave(
                self.num_heads, dim=0
            ) > 0.0
            do_not_attend = torch.cat(
                [
                    do_not_attend[:, :, 1:],
                    torch.ones_like(do_not_attend[:, :, :1], dtype=torch.bool),
                ],
                dim=-1,
            ).cumsum(dim=-1)
            attn_mask = torch.where(do_not_attend > 0, -torch.inf, 0.0)

        attn_output_weights = attn_output_weights + attn_mask
        # dim should be: (h1_n1, h2_n1, h3_n1, h4_n1, h1_n2, h2_n2, h3_n2, h4_n2, ...), (s0, s1, s2, ...), (d0, d1, d2, ...)

        if positions is not None:
            assert self.use_multi_theta == True

            # seq = torch.arange(S).expand(S, -1).to(attn_output_weights.device).unsqueeze(dim=0).repeat(position.shape[0], 1, 1)
            seq = positions.unsqueeze(1).expand(-1, S, -1)

            distance_matrix = torch.tril(seq - seq.transpose(-2, -1), diagonal=-1)

            # self.thetas => 0,1,2,3, 0,1,2,3, 0,1,2,3
            # distance_matrix => 0,0,0,0, 1,1,1,1, 2,2,2,2
            position_effect = self.thetas.repeat(
                positions.shape[0], 1, 1
            ) * distance_matrix.repeat_interleave(self.num_heads, dim=0)

        else:  # position is `None`
            seq = torch.arange(S).expand(S, -1).to(attn_output_weights.device)

            # .tril helps with NaNs b/c of multiplication with parameters
            distance_matrix = torch.tril(seq - seq.t(), diagonal=-1)

        _position_effect = F.softplus(self.thetas) * distance_matrix

        # clone _position_effect to prevent torch inplace modification error
        if self.use_multi_theta:
            batch_size = batch_size_times_num_head // self.num_heads
            position_effect = _position_effect.clone().repeat(batch_size, 1, 1)
        else:
            position_effect = _position_effect.clone().repeat(
                batch_size_times_num_head, 1, 1
            )

        # # only relevant for MAM with multiplication, does not make a difference if using addition
        # # account for padding mask => prevent multiplication of -Inf and 0.0
        # if attn_mask is not None:
        #     position_effect[attn_mask == float("-Inf")] = 1.0

        attn_output_weights = attn_output_weights + position_effect

        # ensure that code above does not escape masking
        attn_output_weights = attn_output_weights + attn_mask

        return attn_output_weights
