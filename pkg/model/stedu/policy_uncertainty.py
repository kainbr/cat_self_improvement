import logging
from logging import Logger

import torch
from torch import Tensor, nn

from pkg.model.stedu.stedu_assessment import Model as AssessmentModel

logger: Logger = logging.getLogger(__name__)


class Model(nn.Module):
    def __init__(self, assessment_model: AssessmentModel):
        super().__init__()
        self.assessment_model = assessment_model

    @torch.no_grad()
    def forward(
        self,
        x: Tensor,
        given_positions: Tensor | None,
        sampling_scheme: str = "argmax",
        temperature: float = 1.0,
    ) -> tuple[Tensor, dict]:

        assessment_model_is_in_training = self.assessment_model.training
        if assessment_model_is_in_training:
            self.assessment_model.eval()

        _, info_dict = self.assessment_model.forward(
            x=x,
            given_positions=given_positions,
            sampling_scheme=sampling_scheme,
            temperature=temperature,
        )
        next_x_given: torch.Tensor = info_dict["most_uncertain_position"]

        # overwrite info_dict
        info_dict = {"next": next_x_given}

        if assessment_model_is_in_training:
            self.assessment_model.train()

        return None, info_dict
