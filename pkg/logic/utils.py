import json
import logging
from dataclasses import dataclass
from logging import Logger
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from pkg.utils.logging import BEST_FILE_NAME, RESULTS_FILE_NAME

logger: Logger = logging.getLogger(__name__)


@dataclass
class State:
    epoch: int = 0
    iteration: int = 0
    best_iteration: int = 0
    best_val_metric: float = -np.inf
    num_epochs_not_improved: int = 0
    num_updates: int = 0


@dataclass
class SampleState:
    iteration: int = 0


def log_results(
    result: dict[str, float], state: State, writer: SummaryWriter, model: str = ""
) -> None:

    assert model in ["assessment", "policy", "policy_training", "policy_imitation"]

    result.update(epoch=state.epoch, iteration=state.iteration)

    # add model to results
    result = {f"{model}_{key}": value for key, value in result.items()}

    # write to json + log
    result_json: str = json.dumps(result, skipkeys=True)
    with open(file=RESULTS_FILE_NAME, mode="a") as file:
        file.write(f"{result_json}\n")
    if (model == "policy") and (state.best_iteration == state.iteration):
        with open(file=BEST_FILE_NAME, mode="w") as file:
            file.write(result_json)
    logger.info(result_json)

    # log results to tensorboard
    for key, value in result.items():
        writer.add_scalar(tag=key, scalar_value=value, global_step=state.iteration)
    writer.flush()
