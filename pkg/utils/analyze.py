from glob import glob
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from pkg.utils.logging import BEST_FILE_NAME, CONFIG_FILE_NAME


def summarize_outputs_in_df(outputs_dir: str) -> pd.DataFrame:
    outputs_path = Path(outputs_dir)

    best_files = glob(
        pathname=f"**/{BEST_FILE_NAME}",
        root_dir=str(outputs_path),
        recursive=True,
        include_hidden=False,
    )
    best_files = [(outputs_path / best_file).absolute() for best_file in best_files]
    run_dirs = [str(best_file.parent) for best_file in best_files]

    joined_dfs = []
    for best_file in best_files:
        result_df = pd.read_json(path_or_buf=best_file, lines=True)
        config = OmegaConf.load(best_file.with_name(CONFIG_FILE_NAME))
        config_df = pd.json_normalize(OmegaConf.to_container(config))  # type: ignore
        joined_df = pd.concat([result_df, config_df], axis=1)
        joined_dfs.append(joined_df)

    df = pd.concat(joined_dfs, axis=0)
    df["run_dir"] = run_dirs
    df["outputs_dir"] = str(outputs_path)
    df = df.reset_index()

    return df
