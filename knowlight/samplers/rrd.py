from __future__ import annotations

import math

import torch

from knowlight.datasets.batch import RRDAuxBatch
from knowlight.datasets.interactions import InteractionDataset
from knowlight.samplers.base import AuxiliarySampler
from knowlight.samplers.teacher_topk import TeacherTopKProvider
from knowlight.teachers.state import TeacherState


class RRDSampler(AuxiliarySampler):
    def __init__(
        self,
        interesting_size: int = 40,
        uninteresting_size: int = 50,
        temperature: float = 1.0,
        teacher_topk: dict[int, list[int]] | None = None,
        topk_provider: TeacherTopKProvider | None = None,
    ):
        self.interesting_size = interesting_size
        self.uninteresting_size = uninteresting_size
        self.temperature = temperature
        self.teacher_topk = teacher_topk
        self.topk_provider = topk_provider or TeacherTopKProvider(top_k=500)
        self.dataset: InteractionDataset | None = None
        self._teacher_topk_tensors: dict[int, torch.Tensor] = {}
        self._ranking_weights: dict[int, torch.Tensor] = {}
        self._available_uninteresting: dict[int, torch.Tensor] = {}
        self._epoch_interesting: torch.Tensor | None = None
        self._epoch_uninteresting: torch.Tensor | None = None
        self._device_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    def initialize(self, dataset: InteractionDataset, teacher_state: TeacherState) -> None:
        self.dataset = dataset
        if self.teacher_topk is None:
            self.teacher_topk = self.topk_provider.build(teacher_state=teacher_state, dataset=dataset)

        all_items = set(range(dataset.num_items))
        self._teacher_topk_tensors = {}
        self._ranking_weights = {}
        self._available_uninteresting = {}
        self._epoch_interesting = None
        self._epoch_uninteresting = None
        self._device_cache = {}
        for user in range(dataset.num_users):
            topk_items = self.teacher_topk.get(user, [])
            if len(topk_items) == 0:
                unseen = list(all_items - dataset.seen_items(user))
                topk_items = unseen[: max(1, min(len(unseen), self.interesting_size))]
            self._teacher_topk_tensors[user] = torch.tensor(topk_items, dtype=torch.long)
            weights = [math.exp(-(idx + 1) / self.temperature) for idx in range(len(topk_items))]
            self._ranking_weights[user] = torch.tensor(weights, dtype=torch.float32)

            blocked = set(topk_items) | dataset.seen_items(user)
            available = sorted(all_items - blocked)
            if len(available) == 0:
                available = list(all_items - dataset.seen_items(user))
            self._available_uninteresting[user] = torch.tensor(available, dtype=torch.long)

    def refresh(self) -> None:
        if self.dataset is None or self.teacher_topk is None or not self._ranking_weights:
            raise RuntimeError("RRDSampler must be initialized before sampling.")

        interesting_rows = []
        uninteresting_rows = []

        for user in range(self.dataset.num_users):
            num_topk = len(self._ranking_weights[user])
            use_replacement = num_topk < self.interesting_size
            sampled_positions = torch.multinomial(
                self._ranking_weights[user],
                self.interesting_size,
                replacement=use_replacement,
            )
            sampled_positions = sampled_positions.sort().values
            interesting_rows.append(self._teacher_topk_tensors[user][sampled_positions])

            available = self._available_uninteresting[user]
            if len(available) >= self.uninteresting_size:
                sampled_uninteresting_idx = torch.randperm(len(available))[: self.uninteresting_size]
            else:
                sampled_uninteresting_idx = torch.randint(0, len(available), (self.uninteresting_size,))
            uninteresting_rows.append(available[sampled_uninteresting_idx])

        self._epoch_interesting = torch.stack(interesting_rows, dim=0)
        self._epoch_uninteresting = torch.stack(uninteresting_rows, dim=0)
        self._device_cache = {}

    def sample(self, indices: torch.Tensor, device: torch.device) -> RRDAuxBatch:
        if self.dataset is None or self.teacher_topk is None or not self._ranking_weights:
            raise RuntimeError("RRDSampler must be initialized before sampling.")
        if self._epoch_interesting is None or self._epoch_uninteresting is None:
            self.refresh()

        device = torch.device(device)
        users = indices.long().to(device)
        if device.type == "cpu":
            interesting_by_user = self._epoch_interesting
            uninteresting_by_user = self._epoch_uninteresting
        else:
            cached = self._device_cache.get(device)
            if cached is None:
                cached = (
                    self._epoch_interesting.to(device),
                    self._epoch_uninteresting.to(device),
                )
                self._device_cache[device] = cached
            interesting_by_user, uninteresting_by_user = cached

        interesting_items = interesting_by_user[users]
        uninteresting_items = uninteresting_by_user[users]
        return RRDAuxBatch(
            users=users,
            interesting_items=interesting_items,
            uninteresting_items=uninteresting_items,
        )
