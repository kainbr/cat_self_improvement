import logging
from logging import Logger

import torch
from torch import Tensor, nn

logger: Logger = logging.getLogger(__name__)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self) -> tuple[Tensor, dict]:
        raise NotImplementedError
