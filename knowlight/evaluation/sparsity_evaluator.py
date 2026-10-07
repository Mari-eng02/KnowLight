import numpy as np
import torch
from typing import Dict, List, Any, Tuple
from knowlight.utils.metrics import compute_topk_metrics


class SparsityEvaluator:
    """
    Evaluator for Degree-Stratified / Sparsity Analysis.
    Partitions test users into homogeneous activity buckets (based on training degree)
    and evaluates ranking metrics per bucket to assess cold/sparse robustness.
    """

    def __init__(self, top_k_list: List[int] = [5, 10, 20]):
        self.top_k_list = top_k_list

    @staticmethod
    def partition_users_into_homogeneous_buckets(
        data_loader,
        num_buckets: int = 3
    ) -> Dict[str, Dict[str, Any]]:
        """
        Partitions test set users into N homogeneous buckets (terciles/quantiles)
        based on their interaction degree in the training graph.

        Returns:
            Dictionary containing bucket definitions and statistics.
        """
        test_dict = getattr(data_loader, "test_pos_items", getattr(data_loader, "test_dict", {}))
        all_test_users = list(test_dict.keys())
        train_matrix = data_loader.train_matrix

        # Calculate training degree for each test user
        user_degrees = []
        for u in all_test_users:
            deg = int(train_matrix[u].nnz)
            user_degrees.append((u, deg))

        # Sort users by degree ascending (fewest interactions first)
        user_degrees.sort(key=lambda x: x[1])

        total_users = len(user_degrees)
        chunk_size = total_users // num_buckets
        remainder = total_users % num_buckets

        bucket_names = {
            0: "Bucket 1 (Sparse / Near-Cold)",
            1: "Bucket 2 (Medium)",
            2: "Bucket 3 (Dense / Warm)"
        }

        buckets = {}
        start_idx = 0
        for b_idx in range(num_buckets):
            # Distribute remainder evenly across first few buckets
            current_size = chunk_size + (1 if b_idx < remainder else 0)
            end_idx = start_idx + current_size
            chunk = user_degrees[start_idx:end_idx]
            start_idx = end_idx

            b_users = [item[0] for item in chunk]
            b_degs = [item[1] for item in chunk]

            b_key = f"bucket_{b_idx + 1}"
            b_label = bucket_names.get(b_idx, f"Bucket {b_idx + 1}")

            buckets[b_key] = {
                "key": b_key,
                "label": b_label,
                "users": b_users,
                "num_users": len(b_users),
                "deg_min": int(np.min(b_degs)) if b_degs else 0,
                "deg_max": int(np.max(b_degs)) if b_degs else 0,
                "deg_mean": float(np.mean(b_degs)) if b_degs else 0.0,
                "deg_median": float(np.median(b_degs)) if b_degs else 0.0,
            }

        return buckets

    def evaluate_model_on_buckets(
        self,
        model,
        data_loader,
        buckets: Dict[str, Dict[str, Any]],
        device: torch.device,
        batch_size: int = 1024
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluates a LightGCN model across all defined buckets and overall.

        Returns:
            Dictionary mapping bucket keys (and 'overall') to metric dictionaries.
        """
        model.eval()
        test_dict = getattr(data_loader, "test_pos_items", getattr(data_loader, "test_dict", {}))
        val_dict = getattr(data_loader, "val_pos_items", getattr(data_loader, "val_dict", {}))
        train_matrix = data_loader.train_matrix

        all_test_users = list(test_dict.keys())
        user_to_idx = {u: i for i, u in enumerate(all_test_users)}

        # 1. Compute prediction scores for all test users in batches
        all_scores_list = []
        with torch.no_grad():
            for i in range(0, len(all_test_users), batch_size):
                batch_users = all_test_users[i : i + batch_size]
                users_tensor = torch.tensor(batch_users, dtype=torch.long).to(device)
                scores = model.predict_all_items(users_tensor) # (batch_size, num_items)
                all_scores_list.append(scores.cpu())

        all_scores = torch.cat(all_scores_list, dim=0) # (num_users, num_items)

        # 2. Mask out seen interactions (train items + val items)
        for idx, u in enumerate(all_test_users):
            seen_train = train_matrix[u].indices
            if len(seen_train) > 0:
                all_scores[idx, seen_train] = -1e9

            if u in val_dict:
                seen_val = val_dict[u]
                if len(seen_val) > 0:
                    all_scores[idx, seen_val] = -1e9

        results = {}

        # 3. Compute metrics for each bucket
        for b_key, b_info in buckets.items():
            b_users = b_info["users"]
            indices = [user_to_idx[u] for u in b_users if u in user_to_idx]
            if len(indices) == 0:
                continue

            b_scores = all_scores[indices]
            b_pos_items = [test_dict[u] for u in b_users]

            b_metrics = compute_topk_metrics(
                pos_items=b_pos_items,
                rank_scores=b_scores,
                top_k_list=self.top_k_list
            )
            results[b_key] = b_metrics

        # 4. Compute overall metrics
        overall_pos = [test_dict[u] for u in all_test_users]
        overall_metrics = compute_topk_metrics(
            pos_items=overall_pos,
            rank_scores=all_scores,
            top_k_list=self.top_k_list
        )
        results["overall"] = overall_metrics

        return results

    @staticmethod
    def compute_deltas(
        baseline_metrics: Dict[str, Dict[str, float]],
        pipeline_metrics: Dict[str, Dict[str, float]],
        round_decimals: int = 4,
        delta_decimals: int = 2
    ) -> Dict[str, Dict[str, float]]:
        """
        Computes percentage improvement: Delta% = ((Pipeline - Baseline) / Baseline) * 100.
        Calculated using input metrics rounded to round_decimals (default: 4) and
        with the final delta percentage rounded to delta_decimals (default: 2), ensuring
        total consistency with thesis tables.
        """
        deltas = {}
        for group in baseline_metrics.keys():
            if group not in pipeline_metrics:
                continue
            deltas[group] = {}
            for metric, base_val in baseline_metrics[group].items():
                pipe_val = pipeline_metrics[group].get(metric, 0.0)

                b_val = round(base_val, round_decimals) if round_decimals is not None else base_val
                p_val = round(pipe_val, round_decimals) if round_decimals is not None else pipe_val

                if b_val > 1e-9:
                    delta_pct = ((p_val - b_val) / b_val) * 100.0
                elif base_val > 1e-9:
                    delta_pct = ((pipe_val - base_val) / base_val) * 100.0
                else:
                    delta_pct = 0.0

                if delta_decimals is not None:
                    delta_pct = round(delta_pct, delta_decimals)

                deltas[group][metric] = delta_pct
        return deltas
