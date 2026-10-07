from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from knowlight.datasets.batch import InteractionBatch, UnKDAuxBatch
from knowlight.datasets.interactions import InteractionDataset
from knowlight.distillation.base import Distiller
from knowlight.teachers.state import TeacherState


def unkd_ranking_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def _can_score_unkd_items_together(student: torch.nn.Module) -> bool:
    if not bool(getattr(student, "can_score_items_together", True)):
        return False
    return not any(
        isinstance(module, torch.nn.modules.dropout._DropoutNd)
        and module.training
        and module.p > 0
        for module in student.modules()
    )


class UnKDDistiller(Distiller):
    """
    Unbiased Knowledge Distillation (UnKD) distiller.

    The distiller samples teacher-ranked item pairs inside popularity groups to mitigate
    popularity bias, asking the student to preserve teacher preference ordering.
    """

    def __init__(
        self,
        lambda_unkd: float = 0.1,
        sample_num: int = 30,
        group_count: int = 4,
        popularity_lambda: float = 1.0,
        rank_top_k: int = 1000,
        rank_temperature: float = 20.0,
    ):
        super().__init__()
        if lambda_unkd < 0:
            raise ValueError("lambda_unkd must be >= 0.")
        if sample_num < 1:
            raise ValueError("sample_num must be >= 1.")
        if group_count < 1:
            raise ValueError("group_count must be >= 1.")
        if popularity_lambda < 0:
            raise ValueError("popularity_lambda must be >= 0.")
        if rank_top_k < 1:
            raise ValueError("rank_top_k must be >= 1.")
        if rank_temperature <= 0:
            raise ValueError("rank_temperature must be > 0.")

        self.lambda_unkd = float(lambda_unkd)
        self.sample_num = int(sample_num)
        self.group_count = int(group_count)
        self.popularity_lambda = float(popularity_lambda)
        self.rank_top_k = int(rank_top_k)
        self.rank_temperature = float(rank_temperature)

        self.dataset: InteractionDataset | None = None
        self.teacher_state: TeacherState | None = None
        self._group_items: list[torch.Tensor] = []
        self._group_sample_counts: list[int] = []
        self._ranked_group_items: list[list[torch.Tensor]] = []
        self._ranked_group_weights: list[list[torch.Tensor]] = []
        self._epoch_pos_items: torch.Tensor | None = None
        self._epoch_neg_items: torch.Tensor | None = None
        self._device_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    def on_train_start(self, teacher_state: TeacherState, dataset: InteractionDataset) -> None:
        self.teacher_state = teacher_state
        self.dataset = dataset
        self._group_items = self._build_popularity_groups(dataset)
        group_ratios = self._build_group_ratios(dataset, self._group_items)
        self._group_sample_counts = self._allocate_group_samples(group_ratios)
        self._ranked_group_items = self._build_ranked_group_items(teacher_state, dataset)
        self._ranked_group_weights = self._build_ranked_group_weights(self._ranked_group_items)
        self.refresh()

    def on_epoch_start(self) -> None:
        self.refresh()

    def build_aux_batch(
        self,
        batch: InteractionBatch,
        device: torch.device,
    ) -> UnKDAuxBatch:
        if self._epoch_pos_items is None or self._epoch_neg_items is None:
            self.refresh()

        device = torch.device(device)
        users = batch.unique_users.long().to(device)
        if device.type == "cpu":
            pos_by_user = self._epoch_pos_items
            neg_by_user = self._epoch_neg_items
        else:
            cached = self._device_cache.get(device)
            if cached is None:
                cached = (
                    self._epoch_pos_items.to(device),
                    self._epoch_neg_items.to(device),
                )
                self._device_cache[device] = cached
            pos_by_user, neg_by_user = cached

        return UnKDAuxBatch(
            users=users,
            pos_items=pos_by_user[users],
            neg_items=neg_by_user[users],
        )

    def compute_loss(
        self,
        student: torch.nn.Module,
        batch: InteractionBatch,
        aux_batch: UnKDAuxBatch | None = None,
    ) -> torch.Tensor:
        if aux_batch is None:
            return torch.zeros((), device=batch.users.device)

        if _can_score_unkd_items_together(student):
            items = torch.cat([aux_batch.pos_items, aux_batch.neg_items], dim=1)
            scores = student.score_items(aux_batch.users, items)
            pos_scores, neg_scores = scores.split(
                [aux_batch.pos_items.size(1), aux_batch.neg_items.size(1)],
                dim=1,
            )
        else:
            pos_scores = student.score_items(aux_batch.users, aux_batch.pos_items)
            neg_scores = student.score_items(aux_batch.users, aux_batch.neg_items)
        return self.lambda_unkd * unkd_ranking_loss(pos_scores, neg_scores)

    def refresh(self) -> None:
        if self.dataset is None or not self._ranked_group_items:
            raise RuntimeError("UnKDDistiller must be initialized before sampling.")

        pos_rows: list[torch.Tensor] = []
        neg_rows: list[torch.Tensor] = []
        for user in range(self.dataset.num_users):
            user_pos: list[torch.Tensor] = []
            user_neg: list[torch.Tensor] = []
            for group_idx, sample_count in enumerate(self._group_sample_counts):
                if sample_count <= 0:
                    continue
                ranked_items = self._ranked_group_items[user][group_idx]
                if ranked_items.numel() < 2:
                    continue
                weights = self._ranked_group_weights[user][group_idx]
                pos_items, neg_items = self._sample_ranked_pairs(ranked_items, weights, sample_count)
                user_pos.append(pos_items)
                user_neg.append(neg_items)

            if user_pos:
                pos_row = torch.cat(user_pos, dim=0)
                neg_row = torch.cat(user_neg, dim=0)
            else:
                pos_row, neg_row = self._fallback_user_pairs(user)

            pos_row, neg_row = self._fit_sample_num(pos_row, neg_row)
            pos_rows.append(pos_row)
            neg_rows.append(neg_row)

        self._epoch_pos_items = torch.stack(pos_rows, dim=0).long()
        self._epoch_neg_items = torch.stack(neg_rows, dim=0).long()
        self._device_cache = {}

    def _build_popularity_groups(self, dataset: InteractionDataset) -> list[torch.Tensor]:
        counts = torch.ones(dataset.num_items, dtype=torch.float32)
        for _, item in dataset.interactions:
            counts[int(item)] += 1.0

        sorted_items = torch.argsort(counts, descending=True).tolist()
        target_mass = float(counts.sum().item()) / self.group_count
        groups: list[list[int]] = []
        current: list[int] = []
        current_mass = 0.0
        for item in sorted_items:
            item_mass = float(counts[item].item())
            if current and current_mass + item_mass > target_mass:
                groups.append(current)
                current = []
                current_mass = 0.0
            current.append(int(item))
            current_mass += item_mass
        if current:
            groups.append(current)

        if len(groups) > self.group_count:
            head = groups[: self.group_count - 1]
            tail = [item for group in groups[self.group_count - 1 :] for item in group]
            groups = head + [tail]
        while len(groups) < self.group_count:
            groups.append([])
        return [torch.tensor(group, dtype=torch.long) for group in groups]

    def _build_group_ratios(
        self,
        dataset: InteractionDataset,
        group_items: list[torch.Tensor],
    ) -> list[float]:
        counts = torch.ones(dataset.num_items, dtype=torch.float32)
        for _, item in dataset.interactions:
            counts[int(item)] += 1.0

        avg_popularity = []
        for items in group_items:
            if items.numel() == 0:
                avg_popularity.append(0.0)
            else:
                avg_popularity.append(float(counts[items].mean().item()))

        max_pop = max(avg_popularity) if avg_popularity else 1.0
        min_pop = min(avg_popularity) if avg_popularity else 0.0
        inverse = [max((max_pop + min_pop) - value, 0.0) for value in avg_popularity]
        total = sum(inverse)
        if total <= 0:
            ratios = [1.0 / len(group_items) for _ in group_items]
        else:
            ratios = [max(value / total, 0.1) for value in inverse]

        ratios = [math.pow(value, self.popularity_lambda) for value in ratios]
        total = sum(ratios)
        return [value / total for value in ratios]

    def _allocate_group_samples(self, group_ratios: list[float]) -> list[int]:
        raw = [ratio * self.sample_num for ratio in group_ratios]
        counts = [int(math.floor(value)) for value in raw]
        remaining = self.sample_num - sum(counts)
        order = sorted(range(len(raw)), key=lambda idx: raw[idx] - counts[idx], reverse=True)
        for idx in order[:remaining]:
            counts[idx] += 1
        return counts

    def _build_ranked_group_items(
        self,
        teacher_state: TeacherState,
        dataset: InteractionDataset,
    ) -> list[list[torch.Tensor]]:
        ranked_by_user: list[list[torch.Tensor]] = []
        for user in range(dataset.num_users):
            seen = dataset.seen_items(user)
            user_groups = []
            all_scores = None
            if teacher_state.scorer is not None:
                all_scores = teacher_state.scorer.score_items_for_user(user, teacher_state.num_items).detach().cpu()
            for group_items in self._group_items:
                candidates = [int(item) for item in group_items.tolist() if int(item) not in seen]
                if not candidates:
                    user_groups.append(torch.empty(0, dtype=torch.long))
                    continue
                candidate_tensor = torch.tensor(candidates, dtype=torch.long)
                if all_scores is not None:
                    scores = all_scores[candidate_tensor]
                else:
                    scores = self._teacher_scores(teacher_state, user, candidate_tensor)
                top_k = min(self.rank_top_k, candidate_tensor.numel())
                ranked_idx = torch.topk(scores, k=top_k, dim=0).indices.cpu()
                user_groups.append(candidate_tensor[ranked_idx])
            ranked_by_user.append(user_groups)
        return ranked_by_user

    def _build_ranked_group_weights(
        self,
        ranked_group_items: list[list[torch.Tensor]],
    ) -> list[list[torch.Tensor]]:
        weights_by_user: list[list[torch.Tensor]] = []
        for user_groups in ranked_group_items:
            group_weights = []
            for ranked_items in user_groups:
                rank_count = ranked_items.numel()
                if rank_count == 0:
                    group_weights.append(torch.empty(0, dtype=torch.float32))
                    continue
                ranks = torch.arange(1, rank_count + 1, dtype=torch.float32)
                group_weights.append(torch.exp(-ranks / self.rank_temperature))
            weights_by_user.append(group_weights)
        return weights_by_user

    def _teacher_scores(
        self,
        teacher_state: TeacherState,
        user: int,
        items: torch.Tensor,
    ) -> torch.Tensor:
        if teacher_state.scorer is not None:
            all_scores = teacher_state.scorer.score_items_for_user(user, teacher_state.num_items).detach().cpu()
            return all_scores[items]

        if getattr(self, "teacher_scores", None) is not None and user in self.teacher_scores:
            u_dict = self.teacher_scores[user]
            return torch.tensor([u_dict.get(int(i), 0.0) for i in items.tolist()], dtype=torch.float32)

        if teacher_state.has_embeddings:
            user_emb = teacher_state.user_embeddings[user].detach().cpu()
            item_emb = teacher_state.item_embeddings[items].detach().cpu()
            return torch.matmul(item_emb, user_emb)

        return torch.rand(len(items), dtype=torch.float32)

    def _sample_ranked_pairs(
        self,
        ranked_items: torch.Tensor,
        weights: torch.Tensor,
        sample_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rank_count = ranked_items.numel()
        first = torch.multinomial(weights, sample_count, replacement=True)
        second = torch.multinomial(weights, sample_count, replacement=True)
        same = first == second
        if same.any() and rank_count > 1:
            second[same] = (second[same] + 1) % rank_count
        better = torch.minimum(first, second)
        worse = torch.maximum(first, second)
        return ranked_items[better], ranked_items[worse]

    def _fallback_user_pairs(self, user: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.dataset is not None
        seen = self.dataset.seen_items(user)
        available = [item for item in range(self.dataset.num_items) if item not in seen]
        if len(available) < 2:
            raise ValueError(f"User {user} has fewer than two unseen items for UnKD sampling.")
        items = torch.tensor(available, dtype=torch.long)
        shuffled = items[torch.randperm(items.numel())]
        return shuffled[:1], shuffled[1:2]

    def _fit_sample_num(
        self,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pos_items.numel() >= self.sample_num:
            return pos_items[: self.sample_num], neg_items[: self.sample_num]

        repeats = math.ceil(self.sample_num / max(1, pos_items.numel()))
        pos_items = pos_items.repeat(repeats)[: self.sample_num]
        neg_items = neg_items.repeat(repeats)[: self.sample_num]
        return pos_items, neg_items
