from __future__ import annotations

import torch

from knowlight.datasets.batch import InteractionBatch, RRDAuxBatch
from knowlight.distillation.base import Distiller
from knowlight.samplers.rrd import RRDSampler


def relaxed_ranking_loss(
    interesting_scores: torch.Tensor,
    uninteresting_scores: torch.Tensor,
) -> torch.Tensor:
    diff = interesting_scores.unsqueeze(-1) - uninteresting_scores.unsqueeze(1)
    return -torch.nn.functional.logsigmoid(diff).mean()


def _can_score_rrd_items_together(student: torch.nn.Module) -> bool:
    if not bool(getattr(student, "can_score_items_together", True)):
        return False
    return not any(
        isinstance(module, torch.nn.modules.dropout._DropoutNd)
        and module.training
        and module.p > 0
        for module in student.modules()
    )


class RRDDistiller(Distiller):
    def __init__(self, sampler: RRDSampler, lambda_rrd: float):
        super().__init__()
        self.sampler = sampler
        self.lambda_rrd = lambda_rrd

    def on_train_start(self, teacher_state, dataset) -> None:
        self.sampler.initialize(dataset=dataset, teacher_state=teacher_state)

    def on_epoch_start(self) -> None:
        self.sampler.refresh()

    def build_aux_batch(
        self,
        batch: InteractionBatch,
        device: torch.device,
    ) -> RRDAuxBatch:
        return self.sampler.sample(batch.unique_users, device=device)

    def compute_loss(
        self,
        student: torch.nn.Module,
        batch: InteractionBatch,
        aux_batch: RRDAuxBatch | None = None,
    ) -> torch.Tensor:
        if aux_batch is None:
            return torch.zeros((), device=batch.users.device)

        if _can_score_rrd_items_together(student):
            items = torch.cat([aux_batch.interesting_items, aux_batch.uninteresting_items], dim=1)
            scores = student.score_items(aux_batch.users, items)
            interesting_scores, uninteresting_scores = scores.split(
                [aux_batch.interesting_items.size(1), aux_batch.uninteresting_items.size(1)],
                dim=1,
            )
        else:
            interesting_scores = student.score_items(aux_batch.users, aux_batch.interesting_items)
            uninteresting_scores = student.score_items(aux_batch.users, aux_batch.uninteresting_items)
        return self.lambda_rrd * relaxed_ranking_loss(interesting_scores, uninteresting_scores)
