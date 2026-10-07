from __future__ import annotations

from abc import ABC

import torch
from torch import nn

from knowlight.datasets.batch import InteractionBatch
from knowlight.datasets.interactions import InteractionDataset
from knowlight.teachers.state import TeacherState


class Distiller(nn.Module, ABC):
    def on_train_start(
        self,
        teacher_state: TeacherState,
        dataset: InteractionDataset,
    ) -> None:
        del teacher_state, dataset

    def on_epoch_start(self) -> None:
        return None

    def build_aux_batch(
        self,
        batch: InteractionBatch,
        device: torch.device,
    ) -> object | None:
        del batch, device
        return None

    def compute_loss(
        self,
        student: nn.Module,
        batch: InteractionBatch,
        aux_batch: object | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError
