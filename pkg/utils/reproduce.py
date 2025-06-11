import logging
import random

import numpy as np
import torch
from omegaconf import OmegaConf

from pkg.utils.logging import CONFIG_FILE_NAME

logger = logging.getLogger(name=__name__)


def seed_everything(seed: int, full_det: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Setting seed of random, numpy and torch to: {seed}")

    if full_det:
        make_deterministic_even_if_it_hurts_performance()
        logger.info(f"Setting CuDNN to deterministic and benchmark to: False")


def make_deterministic_even_if_it_hurts_performance() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def save_config(config) -> None:
    logger.info(f"Saving current config in {CONFIG_FILE_NAME}")
    logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")
    with open(file=CONFIG_FILE_NAME, mode="w") as file:
        file.write(OmegaConf.to_yaml(config))
