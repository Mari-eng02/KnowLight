from __future__ import annotations

import torch
from torch import nn

from knowlight.datasets.batch import InteractionBatch
from knowlight.distillation.base import Distiller
from knowlight.teachers.state import TeacherState


class Expert(nn.Module):
    def __init__(self, dims: list[int]):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2]),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mlp(inputs)


class DEDistiller(Distiller):
    """
    Distillation Experts (DE) distiller.
    """
    def __init__(
        self,
        teacher_dim: int,
        student_dim: int,
        num_experts: int = 20,
        lambda_de: float = 0.01,
        temperature: float = 0.1,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1.")
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        if lambda_de < 0:
            raise ValueError("lambda_de must be >= 0.")

        self.teacher_dim = teacher_dim
        self.student_dim = student_dim
        self.num_experts = num_experts
        self.lambda_de = lambda_de
        self.temperature = temperature

        hidden_dim = (teacher_dim + student_dim) // 2
        if teacher_dim == student_dim:
            hidden_dim = max(1, student_dim // 2)

        dims = [student_dim, hidden_dim, teacher_dim]
        self.user_experts = nn.ModuleList(Expert(dims) for _ in range(num_experts))
        self.item_experts = nn.ModuleList(Expert(dims) for _ in range(num_experts))
        self.user_gate = nn.Sequential(nn.Linear(teacher_dim, num_experts), nn.Softmax(dim=1))
        self.item_gate = nn.Sequential(nn.Linear(teacher_dim, num_experts), nn.Softmax(dim=1))

        self.register_buffer("_teacher_users", torch.empty(0), persistent=False)
        self.register_buffer("_teacher_items", torch.empty(0), persistent=False)
        self.softmax = nn.Softmax(dim=1)

    def on_train_start(self, teacher_state: TeacherState, dataset) -> None:
        del dataset
        device = self._teacher_users.device if self._teacher_users.numel() else self.user_experts[0].mlp[0].weight.device
        self._teacher_users = teacher_state.user_embeddings.detach().to(device)
        self._teacher_items = teacher_state.item_embeddings.detach().to(device)

    def set_temperature(self, temperature: float) -> None:
        self.temperature = temperature

    def compute_loss(
        self,
        student: nn.Module,
        batch: InteractionBatch,
        aux_batch: object | None = None,
    ) -> torch.Tensor:
        del aux_batch
        user_loss = self._entity_loss(
            student_indices=batch.unique_users,
            student_table=student.get_all_user_embeddings(),
            teacher_table=self._teacher_users,
            experts=self.user_experts,
            gate=self.user_gate,
        )
        item_loss = self._entity_loss(
            student_indices=batch.unique_items,
            student_table=student.get_all_item_embeddings(),
            teacher_table=self._teacher_items,
            experts=self.item_experts,
            gate=self.item_gate,
        )
        return self.lambda_de * (user_loss + item_loss)

    def _entity_loss(
        self,
        student_indices: torch.Tensor,
        student_table: torch.Tensor,
        teacher_table: torch.Tensor,
        experts: nn.ModuleList,
        gate: nn.Module,
    ) -> torch.Tensor:
        student_emb = student_table[student_indices]
        teacher_emb = teacher_table[student_indices]
        selection = gate(teacher_emb)

        if self.num_experts == 1:
            selection_result = 1.0
        else:
            noise = torch.distributions.Gumbel(0, 1).sample(selection.size()).to(selection.device)
            selection = selection + 1e-10
            selection = self.softmax((selection.log() + noise) / self.temperature)
            selection = selection.unsqueeze(1).repeat(1, self.teacher_dim, 1)
            selection_result = selection

        expert_outputs = [experts[idx](student_emb).unsqueeze(-1) for idx in range(self.num_experts)]
        expert_outputs = torch.cat(expert_outputs, dim=-1)
        mixed = expert_outputs * selection_result
        mixed = mixed.sum(dim=2)
        return ((teacher_emb - mixed) ** 2).sum(dim=-1).mean()
