import json
import logging
import random
import warnings
from logging import Logger
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from educat.CAT import dataset as educat_dataset
from educat.CAT import model as educat_model
from educat.CAT import strategy as educat_strategy
from pkg.data.export_for_educat import Data
from pkg.utils.logging import BEST_FILE_NAME, Timer
from pkg.utils.reproduce import save_config

warnings.filterwarnings("ignore")
logger: Logger = logging.getLogger(__name__)


@hydra.main(
    config_path="pkg/config", config_name="educat_reproduce", version_base="1.3"
)
def main(config: DictConfig) -> float:

    # save code + config
    save_config(config=config)

    # dispatch to trainer
    logic = hydra.utils.instantiate(config=config)
    best_metric = logic.run()

    logger.info(f"Fin: {best_metric=}")


class Logic:
    def __init__(
        self,
        data_path: str,
        data_seed: int,
        cdm: str,
        test_length: int,
        train_config: DictConfig,
        test_config: DictConfig,
        device: str,
        pretrained_model_path: str | None = None,
        time_inference: bool = False,
        export_data_to_run_dir: bool = False,
        data: DictConfig | None = None,
    ) -> None:

        # preprocess data according to educat standards, but with our splits + configs
        # saves data into the current working path as a side-effect
        if export_data_to_run_dir:
            logger.info("Preprocess data")
            assert data_path == "."
            assert data is not None
            _: Data = hydra.utils.instantiate(config=data)

        self.time_inference = time_inference

        self.data_path = Path(data_path)
        self.pretrained_model_path = pretrained_model_path
        self.data_seed = data_seed

        assert cdm in ["IRT", "NCD"]
        self.cdm = cdm

        self.test_length = test_length

        # turning configs into dicts
        self.train_config = OmegaConf.to_container(train_config)
        self.test_config = OmegaConf.to_container(test_config)
        assert self.test_config["strategy"] in ["random", "mfi", "kli", "becat"]

        # add device to configs
        self.device = device
        self.train_config.update(device=self.device)
        self.test_config.update(device=self.device)

    def run(self):
        if self.pretrained_model_path is None:
            logger.info("train ...")
            self.train()
        else:
            logger.info("using pretrained model ...")

        logger.info("test ...")
        self.test()

    def train(self):
        # adapted from train.ipynb from educat
        config = self.train_config

        # load data
        logger.info("loading train data ...")
        with open(
            self.data_path / f"train_val_test_split_indices_seed_{self.data_seed}.json",
            "r",
        ) as f:
            split_indices = json.load(f)
        train_triplets = pd.read_csv(
            self.data_path / f"train_triplets_seed_{self.data_seed}.csv",
            encoding="utf-8",
        ).to_records(index=False)
        concept_map = json.load(open(self.data_path / "concept_map.json", "r"))
        concept_map = {int(k): v for k, v in concept_map.items()}
        metadata = json.load(open(self.data_path / "metadata.json", "r"))
        num_train_students = len(split_indices["train_indices"])
        metadata.update(num_train_students=num_train_students)
        logger.info("renumbering student_ids ...")  # is required for educat datasets
        train_triplets_renumbered = renumber_student_id(
            data=train_triplets, student_ids=sorted(split_indices["train_indices"])
        )

        logger.info("initializing datasets and model")
        train_data = educat_dataset.TrainDataset(
            train_triplets_renumbered,
            concept_map,
            metadata["num_train_students"],
            metadata["num_questions"],
            metadata["num_concepts"],
        )
        if self.cdm == "IRT":
            model = educat_model.IRTModel(**config)
        elif self.cdm == "NCD":
            model = educat_model.NCDModel(**config)
        else:
            raise NotImplementedError

        # train model
        logger.info("train ...")
        model.init_model(train_data)
        model.train(train_data)

        # save model
        model.adaptest_save(f"model_{self.cdm}_seed_{self.data_seed}.pt")

    def test(self):
        # adapted from test.ipynb from the educat repository
        seed = 0
        np.random.seed(seed)
        torch.manual_seed(seed)

        config = self.test_config
        test_length = self.test_length  # = num_given

        if self.pretrained_model_path is not None:
            ckpt_path = (
                Path(self.pretrained_model_path)
                / f"model_{self.cdm}_seed_{self.data_seed}.pt"
            )
        else:
            ckpt_path = f"model_{self.cdm}_seed_{self.data_seed}.pt"

        # choose strategies here
        strategy_map = {
            "random": educat_strategy.RandomStrategy(),
            "becat": educat_strategy.BECATstrategy(),
            "mfi": educat_strategy.MFIStrategy(),
            "kli": educat_strategy.KLIStrategy(),
        }
        strategy: educat_strategy.AbstractStrategy = strategy_map[config["strategy"]]

        overall_results = {}

        for mode in ["test"]:

            if mode == "val":
                # load data
                logger.info("loading val data ...")
                with open(
                    self.data_path
                    / f"train_val_test_split_indices_seed_{self.data_seed}.json",
                    "r",
                ) as f:
                    split_indices = json.load(f)
                triplets = pd.read_csv(
                    self.data_path / f"val_triplets_seed_{self.data_seed}.csv",
                    encoding="utf-8",
                ).to_records(index=False)
                indices = split_indices["val_indices"]
            elif mode == "test":
                # load data
                logger.info("loading test data ...")
                with open(
                    self.data_path
                    / f"train_val_test_split_indices_seed_{self.data_seed}.json",
                    "r",
                ) as f:
                    split_indices = json.load(f)
                triplets = pd.read_csv(
                    self.data_path / f"test_triplets_seed_{self.data_seed}.csv",
                    encoding="utf-8",
                ).to_records(index=False)
                indices = split_indices["test_indices"]

            concept_map = json.load(open(self.data_path / "concept_map.json", "r"))
            concept_map = {int(k): v for k, v in concept_map.items()}
            metadata = json.load(open(self.data_path / "metadata.json", "r"))
            num_students = len(indices)
            metadata.update({f"num_{mode}_students": num_students})
            logger.info(
                "renumbering student_ids ..."
            )  # is required for educat datasets
            triplets_renumbered = renumber_student_id(
                data=triplets, student_ids=sorted(indices)
            )
            dataset = educat_dataset.AdapTestDataset(
                triplets_renumbered,
                concept_map,
                metadata[f"num_{mode}_students"],
                metadata["num_questions"],
                metadata["num_concepts"],
            )

            if self.cdm == "IRT":
                model = educat_model.IRTModel(**config)
            elif self.cdm == "NCD":
                model = educat_model.NCDModel(**config)
            else:
                raise NotImplementedError

            model.init_model(dataset)
            model.adaptest_load(ckpt_path)
            dataset.reset()

            # testing
            logging.info(
                f"start adaptive testing with {strategy.name} strategy for {mode=}"
            )

            logging.info(f"Iteration 0")
            results = model.evaluate(dataset)
            for name, value in results.items():
                logging.info(f"{name}:{value}")

            # run strategy ...
            S_sel = {}
            for sid in range(dataset.num_students):
                key = sid
                S_sel[key] = []
            selected_questions = {}

            if self.time_inference:
                timer = Timer()

            for it in range(1, test_length + 1):
                logging.info(f"Iteration {it}")

                # select question
                if it == 1 and strategy.name == "BECAT Strategy":
                    for sid in range(dataset.num_students):
                        untested_questions = np.array(list(dataset.untested[sid]))
                        random_index = random.randint(0, len(untested_questions) - 1)
                        selected_questions[sid] = untested_questions[random_index]
                        S_sel[sid].append(untested_questions[random_index])
                elif strategy.name == "BECAT Strategy":
                    selected_questions = strategy.adaptest_select(model, dataset, S_sel)
                    for sid in range(dataset.num_students):
                        S_sel[sid].append(selected_questions[sid])
                else:
                    selected_questions = strategy.adaptest_select(model, dataset)
                for student, question in selected_questions.items():
                    dataset.apply_selection(student, question)

                # update models
                model.adaptest_update(dataset)

                if not self.time_inference:
                    # evaluate models
                    results = model.evaluate(dataset)

                    # log results
                    logger.info(f"{results=}")

                    overall_results.update(
                        {
                            f"{mode}_num_given_{it}_acc": results["acc"],
                            f"{mode}_num_given_{it}_auc": results["auc"],
                            f"{mode}_num_given_{it}_len_pred": results["len_pred"],
                        }
                    )
                elif self.time_inference:
                    logger.info(f"{it=}, {timer.get_duration_in_seconds()=}")
                else:
                    raise Exception

            overall_results.update({f"num_{mode}_triplets": len(triplets)})
            overall_results.update({f"num_{mode}_students": num_students})

        if self.time_inference:
            overall_results.update({"timer": timer.get_duration_in_seconds()})

        logger.info(f"{overall_results=}")
        with open(BEST_FILE_NAME, "w") as f:
            json.dump(overall_results, f)


def renumber_student_id(
    data: np.recarray, student_ids: list[int]
) -> list[tuple[int, int, int]]:

    # adapted from educat
    student_ids = sorted(set(t[0] for t in data))
    renumber_map = {sid: i for i, sid in enumerate(student_ids)}
    data = [(renumber_map[t[0]], t[1], t[2]) for t in data]

    return data


if __name__ == "__main__":
    main()
