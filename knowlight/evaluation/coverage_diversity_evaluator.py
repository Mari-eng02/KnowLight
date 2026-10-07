import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional


class CoverageDiversityEvaluator:
    """
    Evaluator for Beyond-Accuracy metrics in Recommender Systems:
    - Catalog Coverage@K: Percentage of the item catalog recommended to at least one user.
    - Personalization@K (Inter-User Diversity): Disimilarity across recommendation lists of different users.
    - Intra-List Diversity@K (ILD): Average pairwise cosine distance between recommended items within each user list.
    - Novelty@K: Average self-information of recommended items based on training popularity (bits).
    """

    def __init__(self, top_k_list: List[int] = [5, 10, 20]):
        self.top_k_list = sorted(list(set(top_k_list)))

    def generate_topk_recommendations(
        self,
        model,
        data_loader,
        device: torch.device,
        batch_size: int = 1024
    ) -> torch.Tensor:
        """
        Computes the Top-K item recommendations for all test users,
        masking seen training and validation interactions.

        Returns:
            torch.Tensor of shape (num_test_users, max_k) with recommended item IDs.
        """
        model.eval()
        test_dict = getattr(data_loader, "test_pos_items", getattr(data_loader, "test_dict", {}))
        val_dict = getattr(data_loader, "val_pos_items", getattr(data_loader, "val_dict", {}))
        train_matrix = data_loader.train_matrix

        all_test_users = list(test_dict.keys())
        max_k = max(self.top_k_list)

        topk_list = []
        with torch.no_grad():
            for i in range(0, len(all_test_users), batch_size):
                batch_users = all_test_users[i : i + batch_size]
                users_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
                scores = model.predict_all_items(users_tensor) # (batch_size, num_items)

                # Mask out seen training items
                for local_idx, u in enumerate(batch_users):
                    seen_train = train_matrix[u].indices
                    if len(seen_train) > 0:
                        scores[local_idx, seen_train] = -1e9

                    if u in val_dict:
                        seen_val = val_dict[u]
                        if len(seen_val) > 0:
                            scores[local_idx, seen_val] = -1e9

                _, batch_topk = torch.topk(scores, k=max_k, dim=-1)
                topk_list.append(batch_topk.cpu())

        return torch.cat(topk_list, dim=0) # (num_users, max_k)

    def evaluate_model(
        self,
        model,
        data_loader,
        device: torch.device,
        batch_size: int = 1024
    ) -> Dict[str, float]:
        """
        Evaluates Beyond-Accuracy metrics (Coverage, Personalization, ILD, Novelty, Entropy, Gini)
        for a LightGCN model on the test user population.

        Returns:
            Dictionary containing metrics for each K in top_k_list.
        """
        topk_preds = self.generate_topk_recommendations(
            model=model,
            data_loader=data_loader,
            device=device,
            batch_size=batch_size
        ) # (num_users, max_k) on CPU

        num_users, max_k = topk_preds.shape
        num_items = data_loader.num_items

        # Normalized item embeddings from LightGCN student for Intra-List Diversity (ILD)
        with torch.no_grad():
            item_embeds = model.item_embedding.weight.detach().cpu()
            norm_item_embeds = F.normalize(item_embeds, p=2, dim=-1) # (num_items, d)

        # Compute item frequencies in training data for Novelty (Self-Information)
        train_matrix = data_loader.train_matrix
        train_item_counts = np.array(train_matrix.sum(axis=0)).flatten()
        total_train_interactions = max(np.sum(train_item_counts), 1.0)
        item_probs = np.maximum(train_item_counts, 1.0) / total_train_interactions
        item_self_info = torch.tensor(-np.log2(item_probs), dtype=torch.float32) # (num_items,)

        metrics = {}

        for k in self.top_k_list:
            sub_topk = topk_preds[:, :k] # (num_users, k)

            # 1. Catalog Coverage@K (% of item catalog recommended at least once)
            unique_recommended = torch.unique(sub_topk).numel()
            catalog_coverage_pct = (unique_recommended / max(num_items, 1)) * 100.0
            metrics[f"Coverage@{k}"] = float(catalog_coverage_pct)

            # 2. Personalization@K (Inter-User Diversity / 1 - mean pairwise cosine similarity)
            # Efficient O(N*K + num_items) vectorization via sum of squared recommendation counts
            counts = torch.bincount(sub_topk.reshape(-1), minlength=num_items).float()
            sum_c_sq = torch.sum(counts ** 2).item()
            if num_users > 1 and k > 0:
                avg_cosine_sim = (sum_c_sq - num_users * k) / (k * num_users * (num_users - 1))
                personalization = max(0.0, min(1.0, 1.0 - avg_cosine_sim))
            else:
                personalization = 0.0
            metrics[f"Personalization@{k}"] = float(personalization)

            # 3. Intra-List Diversity@K (ILD@K)
            # Vectorized pairwise cosine distance across items within each user recommendation list
            user_item_embeds = norm_item_embeds[sub_topk] # (num_users, k, d)
            sim_matrices = torch.bmm(user_item_embeds, user_item_embeds.transpose(1, 2)) # (num_users, k, k)
            triu_mask = torch.triu(torch.ones(k, k, dtype=torch.bool), diagonal=1)
            pairwise_dists = 1.0 - sim_matrices[:, triu_mask] # (num_users, k*(k-1)/2)
            ild = pairwise_dists.mean().item()
            metrics[f"IntraListDiversity@{k}"] = float(ild)

            # 4. Novelty@K (Average Self-Information in bits)
            novelty = item_self_info[sub_topk].mean().item()
            metrics[f"Novelty@{k}"] = float(novelty)

        return metrics

    @staticmethod
    def compute_deltas(
        baseline_metrics: Dict[str, float],
        pipeline_metrics: Dict[str, float],
        round_decimals: int = 4,
        delta_decimals: int = 2
    ) -> Dict[str, float]:
        """
        Computes relative percentage improvement (Delta %) for each metric:
        Delta % = ((Pipeline - Baseline) / Baseline) * 100
        Using values rounded to round_decimals (default: 4) and delta rounded to delta_decimals (default: 2).
        """
        deltas = {}
        for metric_name, b_val in baseline_metrics.items():
            p_val = pipeline_metrics.get(metric_name, 0.0)
            b_round = round(b_val, round_decimals) if round_decimals is not None else b_val
            p_round = round(p_val, round_decimals) if round_decimals is not None else p_val
            if abs(b_round) > 1e-12:
                delta_pct = ((p_round - b_round) / abs(b_round)) * 100.0
            else:
                delta_pct = 0.0
            if delta_decimals is not None:
                delta_pct = round(delta_pct, delta_decimals)
            deltas[metric_name] = float(delta_pct)
        return deltas
