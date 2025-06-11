import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pkg.data.utils import PADDING_VALUE, Vectorizer

logger = logging.getLogger(name=__name__)


def pivot_df(df: pd.DataFrame, values: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Copied from NeuIPS challenge boilerplate code. (+ user_id)
    """
    data = df.pivot(index="UserId", columns="QuestionId", values=values)

    data_cols = data.columns
    all_cols = np.arange(948)
    missing = set(all_cols) - set(data_cols)
    for i in missing:
        data[i] = np.nan
    data = data.reindex(sorted(data.columns), axis=1)

    # Naive classifier achieves 0.63 accuracy
    # weight = (data.count(axis=0).values / data.count(axis=0).sum())
    # accs = data.mean(axis=0).values
    # accs[data.mean(axis=0).values <= 0.5] = 1 - accs[data.mean(axis=0).values <= 0.5]
    # weight @ accs = 0.6287871720158787

    user_id = data.index.to_numpy()
    data = data.to_numpy()
    data[np.isnan(data)] = -1
    return data, user_id


@dataclass
class User:
    # TODO: refactor, extend
    # This id is currently only the row id and not the real u_id from metadata
    u_id: int

    # def as_tensor(self, pos: int | None = None) -> torch.Tensor:
    #     data_as_tensor = torch.Tensor([self.u_id])
    #     groups_as_tensor = torch.Tensor([group_int.get("u_id")])

    #     if pos == None:
    #         return torch.stack([data_as_tensor, groups_as_tensor], dim=-1)
    #     else:
    #         pos_as_tensor = pos * torch.ones_like(data_as_tensor)
    #         return torch.stack([data_as_tensor, groups_as_tensor, pos_as_tensor], dim=-1)


@dataclass
class Interaction:
    question: int
    response: int
    answer: int | None = None
    position: int | None = None
    _features: list[str] = field(default_factory=list)

    def load_metadata(self, metadata: "Metadata", features: list[str]):
        for f in features:
            match f:
                case "subjects":
                    if "subjects" in self._features:
                        raise ValueError
                    self.subjects = metadata.get_question_subjects(self.question)
                    self._features.append("subjects")
                case "answer":
                    if self.answer is not None:
                        self.answer = [self.answer]
                        self._features.append("answer")
                case _:
                    raise ValueError
        return self

    def __len__(self) -> int:
        num_question, num_response = 1, 1
        num_total = num_question + num_response
        for f in self._features:
            num_total += len(getattr(self, f))
        return num_total

    def encode(self, vectorizer: Vectorizer, max_len: int) -> np.ndarray:
        # Init groups and values and add question
        groups = [vectorizer.encode_group("question")]
        values = [vectorizer.encode_value(group="question", value=self.question)]

        # Loop over available features and encode them
        for f in self._features:
            for v in getattr(self, f):
                groups.append(vectorizer.encode_group(group=f))
                values.append(vectorizer.encode_value(group=f, value=v))

        # Pad til max_len - 1
        groups += [PADDING_VALUE] * (max_len - len(groups) - 1)
        values += [PADDING_VALUE] * (max_len - len(values) - 1)

        groups.append(vectorizer.encode_group("response"))
        values.append(vectorizer.encode_value(group="response", value=self.response))

        if self.position is not None:
            positions = [self.position] * len(values)
            return np.stack([values, groups, positions], axis=-1)
        else:
            return np.stack([values, groups], axis=-1)


class Metadata:
    def __init__(
        self,
        user_metadata_path: Path,
        subject_metadata_path: Path,
        question_metadata_path: Path,
        answer_metadata_path: Path,
    ):
        df_user_metadata = pd.read_csv(user_metadata_path)
        df_subject_metadata = pd.read_csv(subject_metadata_path)
        df_question_metadata = pd.read_csv(question_metadata_path)
        df_answer_metadata = pd.read_csv(answer_metadata_path)

        # Process questions
        self.question_metadata = df_question_metadata.set_index("QuestionId")
        self.question_metadata["SubjectId"] = self.question_metadata["SubjectId"].apply(
            lambda x: json.loads(x)[1:]
        )
        # self.question_metadata.rename(columns={"SubjectId": "KCs"}, inplace=True)
        # TODO: Add information from image folder

    def get_question_subjects(self, question: int) -> list[int]:
        return self.question_metadata["SubjectId"].loc[question]
