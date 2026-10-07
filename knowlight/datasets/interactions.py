from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class InteractionDataset:
    num_users: int
    num_items: int
    train_dict: dict[int, set[int]]
    interactions: list[tuple[int, int]]
    # Optional chronological sequences per user (preserves duplicates and order)
    train_sequences: dict[int, list[int]] | None = None
    # Optional timestamps aligned with `train_sequences`
    train_timestamps: dict[int, list[int]] | None = None

    @classmethod
    def from_train_dict(
        cls,
        train_dict: dict[int, set[int]] | dict[int, list[int]],
        num_users: int,
        num_items: int,
        *,
        train_sequences: dict[int, list[int]] | None = None,
    ) -> "InteractionDataset":
        normalized = {int(u): set(map(int, items)) for u, items in train_dict.items()}
        interactions = [
            (user, item) for user, items in normalized.items() for item in sorted(items)
        ]
        # Normalize sequences if provided
        if train_sequences is not None:
            normalized_sequences: dict[int, list[int]] = {
                int(u): [int(i) for i in seq] for u, seq in train_sequences.items()
            }
        else:
            normalized_sequences = None
        # train_timestamps are set by caller if available; default to None here
        normalized_timestamps = None
        return cls(
            num_users=num_users,
            num_items=num_items,
            train_dict=normalized,
            interactions=interactions,
            train_sequences=normalized_sequences,
            train_timestamps=normalized_timestamps,
        )

    def seen_items(self, user: int) -> set[int]:
        return self.train_dict.get(int(user), set())
