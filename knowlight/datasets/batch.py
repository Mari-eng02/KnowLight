from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


@dataclass
class InteractionBatch:
    users: torch.Tensor
    pos_items: torch.Tensor
    neg_items: torch.Tensor
    item_seq: torch.Tensor | None = None
    item_seq_len: torch.Tensor | None = None

    @property
    def unique_users(self) -> torch.Tensor:
        return self.users.unique()

    @property
    def unique_items(self) -> torch.Tensor:
        return torch.cat([self.pos_items, self.neg_items]).unique()


@dataclass
class RRDAuxBatch:
    users: torch.Tensor
    interesting_items: torch.Tensor
    uninteresting_items: torch.Tensor


@dataclass
class UnKDAuxBatch:
    users: torch.Tensor
    pos_items: torch.Tensor
    neg_items: torch.Tensor


class BPRDataset(Dataset):
    """Dataset class for BPR triplet sampling (user, pos_item, neg_item)."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        train_dict: dict[int, set[int]] | None = None,
        train_matrix: sp.csr_matrix | None = None,
        interactions: list[tuple[int, int]] | None = None
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.train_dict = train_dict if train_dict is not None else {}

        if train_matrix is not None and len(self.train_dict) == 0:
            for u in range(num_users):
                indices = train_matrix[u].indices
                if len(indices) > 0:
                    self.train_dict[u] = set(indices.tolist())

        if interactions is not None and len(interactions) > 0:
            self.pairs = np.array(interactions, dtype=np.int64)
        else:
            pairs = []
            for u, items in self.train_dict.items():
                for i in items:
                    pairs.append((u, i))
            self.pairs = np.array(pairs, dtype=np.int64) if len(pairs) > 0 else np.zeros((0, 2), dtype=np.int64)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        u, pos_i = self.pairs[idx]
        pos_set = self.train_dict.get(int(u), {int(pos_i)})
        while True:
            neg_i = np.random.randint(0, self.num_items)
            if neg_i not in pos_set:
                break
        return torch.tensor(u, dtype=torch.long), torch.tensor(pos_i, dtype=torch.long), torch.tensor(neg_i, dtype=torch.long)
