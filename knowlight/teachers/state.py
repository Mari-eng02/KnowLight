from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
import numpy as np
import torch


@runtime_checkable
class ScorerProtocol(Protocol):
    def score_items_for_user(self, user: int, num_items: int) -> torch.Tensor:
        ...


@dataclass
class TeacherState:
    """
    Encapsulates teacher representations and scoring capabilities for distillation.
    Compatible with Graph teachers (LightGCN) and LLM teachers (LLMEmb, Qwen, LLaMA), etc.
    """
    user_embeddings: torch.Tensor | None = None
    item_embeddings: torch.Tensor | None = None
    scorer: Any | None = None
    num_users: int | None = None
    num_items: int | None = None

    def __post_init__(self):
        if self.user_embeddings is not None:
            if isinstance(self.user_embeddings, np.ndarray):
                self.user_embeddings = torch.from_numpy(self.user_embeddings).float()
            if self.num_users is None:
                self.num_users = int(self.user_embeddings.size(0))

        if self.item_embeddings is not None:
            if isinstance(self.item_embeddings, np.ndarray):
                self.item_embeddings = torch.from_numpy(self.item_embeddings).float()
            if self.num_items is None:
                self.num_items = int(self.item_embeddings.size(0))

    @property
    def has_embeddings(self) -> bool:
        return self.user_embeddings is not None and self.item_embeddings is not None

    @property
    def embedding_dim(self) -> int:
        if self.user_embeddings is not None:
            return int(self.user_embeddings.size(-1))
        if self.item_embeddings is not None:
            return int(self.item_embeddings.size(-1))
        return 0

    @property
    def device(self) -> torch.device:
        if self.user_embeddings is not None:
            return self.user_embeddings.device
        if self.item_embeddings is not None:
            return self.item_embeddings.device
        return torch.device("cpu")

    def to(self, device: torch.device | str) -> "TeacherState":
        dev = torch.device(device) if isinstance(device, str) else device
        u_emb = self.user_embeddings.to(dev) if self.user_embeddings is not None else None
        i_emb = self.item_embeddings.to(dev) if self.item_embeddings is not None else None
        return TeacherState(
            user_embeddings=u_emb,
            item_embeddings=i_emb,
            scorer=self.scorer,
            num_users=self.num_users,
            num_items=self.num_items,
        )
