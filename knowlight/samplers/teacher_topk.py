from __future__ import annotations

import torch

from knowlight.datasets.interactions import InteractionDataset
from knowlight.teachers.state import TeacherState


class TeacherTopKProvider:
    def __init__(self, top_k: int):
        self.top_k = top_k

    def build(
        self,
        teacher_state: TeacherState,
        dataset: InteractionDataset,
    ) -> dict[int, list[int]]:
        if teacher_state.scorer is not None:
            return self._build_with_exact_scorer(teacher_state=teacher_state, dataset=dataset)
        if not teacher_state.has_embeddings:
            # Fallback: rank items by dataset frequency for unseen candidates
            counts = torch.zeros(dataset.num_items, dtype=torch.float32)
            for _, item in dataset.interactions:
                counts[int(item)] += 1.0
            sorted_by_pop = torch.argsort(counts, descending=True).tolist()
            topk_by_user: dict[int, list[int]] = {}
            for user in range(dataset.num_users):
                seen = dataset.seen_items(user)
                candidates = [i for i in sorted_by_pop if i not in seen]
                topk_by_user[user] = candidates[: min(self.top_k, len(candidates))]
            return topk_by_user

        user_emb = teacher_state.user_embeddings
        item_emb = teacher_state.item_embeddings
        scores = torch.matmul(user_emb, item_emb.T)
        topk_by_user: dict[int, list[int]] = {}
        num_teacher_users = int(user_emb.size(0))
        num_teacher_items = int(item_emb.size(0))

        for user in range(min(dataset.num_users, num_teacher_users)):
            seen = sorted(item for item in dataset.seen_items(user) if 0 <= item < num_teacher_items)
            user_scores = scores[user].clone()
            if seen:
                user_scores[seen] = -1e9
            k = min(self.top_k, num_teacher_items - len(seen))
            if k <= 0:
                topk_by_user[user] = []
                continue
            top_items = torch.topk(user_scores, k=k, dim=0).indices.tolist()
            topk_by_user[user] = [int(item) for item in top_items]

        return topk_by_user

    def _build_with_exact_scorer(
        self,
        teacher_state: TeacherState,
        dataset: InteractionDataset,
    ) -> dict[int, list[int]]:
        topk_by_user: dict[int, list[int]] = {}
        num_teacher_users = teacher_state.num_users
        num_teacher_items = teacher_state.num_items

        for user in range(min(dataset.num_users, num_teacher_users)):
            seen = sorted(item for item in dataset.seen_items(user) if 0 <= item < num_teacher_items)
            user_scores = teacher_state.scorer.score_items_for_user(user, num_teacher_items).clone()
            if seen:
                user_scores[seen] = -1e9
            k = min(self.top_k, num_teacher_items - len(seen))
            if k <= 0:
                topk_by_user[user] = []
                continue
            top_items = torch.topk(user_scores, k=k, dim=0).indices.tolist()
            topk_by_user[user] = [int(item) for item in top_items]

        return topk_by_user
