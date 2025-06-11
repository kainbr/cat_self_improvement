import logging
import os
import shutil
from datetime import datetime
from functools import lru_cache
from logging import Logger
from pathlib import Path

RUN_DIR = os.getcwd()
OUTPUT_DIR = "outputs"

# # adding `root` logger with stream_handler
# root_logger = logging.getLogger()
# stream_handler = logging.StreamHandler()
# formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
# stream_handler.setFormatter(formatter)
# root_logger.addHandler(stream_handler)
# root_logger.setLevel(logging.INFO)

# exp_hash = uuid.uuid4().hex
# log_dir: Path = (
#     Path(OUTPUT_DIR).absolute()
#     / datetime.now().strftime(format="%Y-%m-%d")
#     / (datetime.now().strftime(format="%H-%M-%S") + f"-{exp_hash[:8]}")
# )
# log_dir.mkdir(parents=True)
# os.chdir(
#     log_dir,
# )
# print(f"Changed into logging directory: {os.getcwd()}")

# # adding file_handler to `root` logger
# file_handler = logging.FileHandler(filename="log.log")
# file_handler.setFormatter(formatter)
# root_logger.addHandler(file_handler)

logger = logging.getLogger(name=__name__)

CONFIG_FILE_NAME = "config.yaml"
CHECKPOINT_NAME = "checkpoint"
BEST_FILE_NAME = "best.yaml"
RESULTS_FILE_NAME = "results.json"


def save_snapshot_of_source_code(
    source_path: str = ".source", pkg: str = "pkg", file_name: str = "main.py"
) -> None:
    dst = Path(source_path).absolute()
    code_dir = Path(
        RUN_DIR
    ).absolute()  # this introduces a convention: run code from code directory (workaround for vscode bug to give relative path of `__file__` even for python 3.9+)
    shutil.copytree(src=code_dir / pkg, dst=dst / pkg)
    print(file_name)
    shutil.copy(src=code_dir / file_name, dst=dst / file_name)

    logger.info(f"Saving source code")


class Timer:
    def __init__(self):
        self.start = datetime.now()

    def get_duration_in_seconds(self) -> float:
        self.stop = datetime.now()
        duration = self.stop - self.start
        return duration.total_seconds()


@lru_cache(None)
def logger_warn_once(logger: Logger, msg: str):
    logger.warning(msg)
