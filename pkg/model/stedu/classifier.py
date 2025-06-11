import torch
import torch.distributions as Dist
from torch import nn
from torchvision import ops


class Classifier(nn.Module):
    """Base classifier"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Linear(Classifier):
    """Wrapper around nn.Linear"""

    def __init__(self, d_in: int, d_out) -> None:
        super().__init__()
        self.net = nn.Linear(in_features=d_in, out_features=d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLP(Classifier):
    """Wrapper around opt.MLP"""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_hidden: int = 32,
        num_hidden_layers: int = 1,
        use_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # construct hidden channels as input to ops.MLP
        hidden_channels = [d_hidden] * num_hidden_layers + [d_out]

        self.net = nn.Sequential(
            ops.MLP(
                in_channels=d_in,
                hidden_channels=hidden_channels,
                norm_layer=nn.LayerNorm if use_norm else None,
                dropout=dropout,
            )
        )

        # pop last dropout layer
        self.net[0].pop(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
