import numpy as np
import torch


def compute_topk_metrics(
    pos_items: torch.Tensor,
    rank_scores: torch.Tensor,
    top_k_list: list = [5, 10, 20]
) -> dict:
    """
    Computes Precision@K, Recall@K, NDCG@K, and Hit Ratio (HR@K).
    
    Args:
        pos_items: Tensor/array of ground-truth positive item indices for each user.
                   Shape: (num_users, max_pos) or list of lists.
        rank_scores: Tensor of predicted scores for candidate items per user.
                     Shape: (num_users, num_items)
        top_k_list: List of K thresholds.
        
    Returns:
        Dictionary mapping metric names (e.g. 'Precision@10') to float values.
    """
    num_candidates = rank_scores.size(-1)
    max_k = min(max(top_k_list), num_candidates) if num_candidates > 0 else max(top_k_list)
    _, topk_indices = torch.topk(rank_scores, k=max_k, dim=-1) # (num_users, max_k)
    
    topk_indices = topk_indices.cpu().numpy()
    
    metrics = {f"Precision@{k}": [] for k in top_k_list}
    metrics.update({f"Recall@{k}": [] for k in top_k_list})
    metrics.update({f"NDCG@{k}": [] for k in top_k_list})
    metrics.update({f"HR@{k}": [] for k in top_k_list})
    
    for user_idx, user_topk in enumerate(topk_indices):
        raw_pos = pos_items[user_idx]
        if isinstance(raw_pos, torch.Tensor):
            raw_pos = raw_pos.cpu().tolist()

        if isinstance(raw_pos, (list, set, tuple, np.ndarray)):
            user_pos = set(int(x) for x in raw_pos)
        else:
            user_pos = {int(raw_pos)}

        if len(user_pos) == 0:
            continue

        for k in top_k_list:
            top_k_preds = user_topk[:k]
            hits = [1 if int(item) in user_pos else 0 for item in top_k_preds]
            n_hits = sum(hits)
            
            # Precision@K
            precision = n_hits / k
            metrics[f"Precision@{k}"].append(precision)
            
            # Recall@K
            recall = n_hits / len(user_pos)
            metrics[f"Recall@{k}"].append(recall)
            
            # Hit Ratio@K
            hr = 1.0 if n_hits > 0 else 0.0
            metrics[f"HR@{k}"].append(hr)
            
            # NDCG@K
            dcg = sum([hit / np.log2(idx + 2) for idx, hit in enumerate(hits)])
            idcg = sum([1.0 / np.log2(idx + 2) for idx in range(min(len(user_pos), k))])
            ndcg = dcg / idcg if idcg > 0 else 0.0
            metrics[f"NDCG@{k}"].append(ndcg)
            
    # Compute mean over users
    final_metrics = {m: float(np.mean(vals)) if len(vals) > 0 else 0.0 for m, vals in metrics.items()}
    return final_metrics
