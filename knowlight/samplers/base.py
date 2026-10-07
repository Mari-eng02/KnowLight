from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class AuxiliarySampler(ABC):
    def initialize(self, dataset, teacher_state) -> None:
        del dataset, teacher_state

    def refresh(self) -> None:
        return None

    @abstractmethod
    def sample(self, indices: torch.Tensor, device: torch.device) -> object:
        raise NotImplementedError
