import logging

from pkg.data.neurips.utils import Interaction, Metadata, pivot_df
import numpy as np
import pandas as pd
from pkg.data import BaseDataset
from tqdm import tqdm

logger = logging.getLogger(name=__name__)


class Dataset(BaseDataset):
    def __init__(
        self,
        user_features: list[str],
        interaction_features: list[str],
        position_type: str,
        data_path: str,
        cache_path: str | None,
        max_num_sequences: int | None,
        max_sequence_len: int | None,
        incl_test_data: bool = False,
    ) -> None:

        dataset = "neurips_education_challenge"

        assert position_type in ["question", "arange"]

        super().__init__(
            dataset=(
                f"{dataset}_incl_test" if incl_test_data else dataset
            ),  # for caching
            user_features=user_features,
            interaction_features=interaction_features,
            position_type=position_type,
            max_num_sequences=max_num_sequences,
            max_sequence_len=max_sequence_len,
            data_path=data_path,
            cache_path=cache_path,
        )

        meta_path = self.data_path / dataset / "metadata"
        self.metadata = Metadata(
            user_metadata_path=meta_path / "student_metadata_task_3_4.csv",
            subject_metadata_path=meta_path / "subject_metadata.csv",
            question_metadata_path=meta_path / "question_metadata_task_3_4.csv",
            answer_metadata_path=meta_path / "answer_metadata_task_3_4.csv",
        )

        if self.sequences is None:

            dataset_path = self.data_path / dataset
            df = pd.read_csv(dataset_path / "train_data" / "train_task_3_4.csv")

            if incl_test_data:
                df_public = pd.read_csv(
                    dataset_path / "test_data" / "test_public_task_4_more_splits.csv"
                )
                df_private = pd.read_csv(
                    dataset_path / "test_data" / "test_private_task_4_more_splits.csv"
                )

                # filter out splits, not used in BECAT as it seems
                df_cols = list(df.columns)
                df_public = df_public[df_cols]
                df_private = df_private[df_cols]
                assert (df.columns == df_public.columns).all()
                assert (df.columns == df_private.columns).all()

                # make sure that no user overlap
                df_unique_uids = df["UserId"].unique()
                df_public_unique_uids = df_public["UserId"].unique()
                df_private_unique_uids = df_private["UserId"].unique()
                for student in df_unique_uids:
                    assert student not in df_public_unique_uids
                    assert student not in df_private_unique_uids
                for student in df_public_unique_uids:
                    assert student not in df_private_unique_uids

                df_all = pd.concat([df, df_public, df_private])
                assert len(df_all) == len(df) + len(df_public) + len(df_private)

                # overwrite df
                df = df_all
                logger.info(f"{df.shape=}")

            is_correct, user_id = pivot_df(df=df, values="IsCorrect")
            answer_values, user_id_ = pivot_df(df=df, values="AnswerValue")
            assert (
                user_id == user_id_
            ).all()  # this information might not be available at eval (e.g. in neurips eval)
            logger.info(f"{is_correct.shape=}, {answer_values.shape=}")

            if max_num_sequences is not None:
                is_correct = is_correct[:max_num_sequences]
                answer_values = answer_values[:max_num_sequences]

            logger.info(f"Constructing user sequences")
            # IDEA: pass in user_id to get user_metadata ... could return dict[int, tuple[User, list[Interaction]]]
            interaction_sequences: dict[int, list[Interaction]] = {
                i: self._get_user_sequence(
                    user=i,
                    responses=is_correct[i],
                    answers=answer_values[i],
                    metadata=self.metadata,
                    user_features=self.user_features,
                    interaction_features=self.interaction_features,
                )
                for i in tqdm(range(len(is_correct)))
            }

            self.max_interaction_len = max(
                [len(i) for s in interaction_sequences.values() for i in s]
            )
            logger.info(f"Encoding user sequences")
            self.sequences = {
                i: np.concatenate(
                    [
                        np.expand_dims(
                            s.encode(
                                vectorizer=self.vectorizer,
                                max_len=self.max_interaction_len,
                            ),
                            axis=0,
                        )
                        for s in sequences
                    ],
                    axis=0,
                )
                for i, sequences in tqdm(interaction_sequences.items())
            }

            self.save_to_cache()

        # statistics
        # TODO: Refactor that max_position_idx is removed if it is not needed for positional embedding anymore
        # ATTENTION! +1 because of 0 position. look out! dangerous when refactoring
        # ATTENTION! Assumes that we have encoded positions
        self.max_position_idx = max(s[:, -1].max() for s in self.sequences.values()) + 1

        # logging statistics
        logger.info(f"{len(self.sequences)=}")
        logger.info(f"{self.max_position_idx=}")
        logger.info(f"{self.vectorizer.data=}")

    def _get_user_sequence(
        self,
        user: np.ndarray,
        responses: np.ndarray,
        answers: np.ndarray,
        metadata: Metadata,
        user_features: list[str],
        interaction_features: list[str],
    ) -> list[Interaction]:
        mask = responses != -1
        assert np.allclose(mask, answers != -1)
        questions = np.argwhere(mask).squeeze()

        # user = User(u_id=user).as_array(pos=0) if include_user else None
        # user = None  # TODO: Implement this, so far we do not use any user information

        # check answer position
        if "answer" in interaction_features:
            assert (
                interaction_features[-1] == "answer"
            ), "`answer` supposed to be last `feature`"

        answers = [int(a) for a in answers[mask]]
        responses = [int(r) for r in responses[mask]]

        if self.position_type.startswith("question"):
            positions = [
                self.vectorizer.encode_value(group="question", value=q)
                for q in questions
            ]
        elif self.position_type.startswith("arange"):
            positions = np.arange(len(questions))
        else:
            raise ValueError

        interactions = [
            Interaction(question=q, response=r, answer=a, position=p).load_metadata(
                metadata=metadata, features=interaction_features
            )
            for q, a, r, p in zip(questions, answers, responses, positions)
        ]

        return interactions
