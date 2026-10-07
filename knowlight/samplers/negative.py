from __future__ import annotations

import random

from knowlight.datasets.interactions import InteractionDataset


class BPRNegativeSampler:
    def __init__(self, dataset: InteractionDataset):
        self.dataset = dataset

    def sample(self, user: int) -> int:
        seen = self.dataset.seen_items(user)
        while True:
            item = random.randrange(self.dataset.num_items)
            if item not in seen:
                return item
