import torch
from torch import Tensor

from pkg.data.utils import PADDING_VALUE


def gather_randomly_and_split_sequences(
    x: Tensor, num_given_positions: int
) -> tuple[Tensor, Tensor]:

    B, S, m, f = x.shape
    positions = x[..., 0, 2].clone()

    random_numbers = torch.rand_like(positions.float())
    random_numbers[positions == PADDING_VALUE] = torch.inf
    idx = torch.argsort(random_numbers, dim=-1)
    idx = idx.reshape(*idx.shape, 1, 1).repeat(1, 1, m, f)
    x = torch.gather(input=x, dim=1, index=idx)

    # Split into given and partially visible partitions, properly padded
    split_idx_1 = num_given_positions
    x_given = x[:, :split_idx_1]
    x_partial = x[:, split_idx_1:]

    # Assert shapes
    B_given, S_given, m_given, f_given = x_given.shape
    B_partial, S_partial, m_partial, f_partial = x_partial.shape
    assert B == B_given == B_partial
    assert S == S_given + S_partial
    assert m == m_given == m_partial
    assert f == f_given == f_partial == 3

    return x_given, x_partial


def gather_and_split_sequences(
    x: Tensor,
    given_positions: Tensor,
) -> tuple[Tensor, Tensor]:

    B, S, m, f = x.shape
    B_given, num_given_positions = given_positions.shape
    assert B == B_given

    positions = x[..., 0, 2].clone()

    if num_given_positions == 0:
        x_given = x[:, :0, :, :]
        x_partial = x
    else:
        # Get gather_idx
        gather_idx_matrix = given_positions.unsqueeze(dim=1) == positions.unsqueeze(
            dim=-1
        )
        # Check if all positions are found
        assert (gather_idx_matrix.sum(dim=1) == 1).all()

        # Gather x_given
        gather_idx_given = gather_idx_matrix.float().argmax(dim=1)
        gather_idx_given = gather_idx_given.reshape(
            *gather_idx_given.shape, 1, 1
        ).repeat(1, 1, m, f)
        x_given = torch.gather(input=x, dim=1, index=gather_idx_given)

        # Gather x_partial
        # True if idx is in given_positions
        positions_given_mask = gather_idx_matrix.sum(dim=-1)
        gather_idx_partial = torch.argsort(
            positions_given_mask, descending=False, dim=-1
        )[..., :-num_given_positions]
        gather_idx_partial = gather_idx_partial.reshape(
            *gather_idx_partial.shape, 1, 1
        ).repeat(1, 1, m, f)
        x_partial = torch.gather(input=x, dim=1, index=gather_idx_partial)
    # end else

    # Assert shapes
    B_given, S_given, m_given, f_given = x_given.shape
    B_partial, S_partial, m_partial, f_partial = x_partial.shape
    assert B == B_given == B_partial
    assert S == S_given + S_partial
    assert m == m_given == m_partial
    assert f == f_given == f_partial == 3

    return x_given, x_partial


def get_key_padding_mask(x: Tensor) -> Tensor:
    return (x[..., 0] == PADDING_VALUE).detach()
