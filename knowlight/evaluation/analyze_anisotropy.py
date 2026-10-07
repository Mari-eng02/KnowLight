#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Anisotropy and Representation Degeneration Analysis
===================================================
Analyzes embedding distribution for:
- Pipeline 2: Teacher Item & User embeddings (LLMEmb)
- Pipeline 3: Teacher Item & User Intent embeddings (Llama-3-8B-Instruct)

Metrics:
1. Centroid norm (Cosine directional bias)
2. Pairwise Cosine Similarity Distribution (Item-Item, User-User, User-Item)
3. Singular Value Spectrum & Effective Rank (Roy & Vetterli)
4. Top Singular Value Explained Variance
5. Wang & Isola Uniformity Metric
6. Top-K Candidate Collision / Frequency skew (for Graph Augmentation)
"""

import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path("c:/Users/maria/OneDrive - Politecnico di Bari/Magistrale/Tesi/KnowLight")

def compute_anisotropy_metrics(embeddings: torch.Tensor, name: str, sample_size: int = 5000):
    """
    Computes rigorous anisotropy & representation geometry metrics on an embedding tensor.
    embeddings: (N, D) torch tensor
    """
    N, D = embeddings.shape
    
    # Subsample if large
    if N > sample_size:
        perm = torch.randperm(N)[:sample_size]
        sub_emb = embeddings[perm].float()
    else:
        sub_emb = embeddings.float()
        
    n_sub = sub_emb.shape[0]
    
    # 1. L2 Normalize
    norms = torch.norm(sub_emb, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
    normed = sub_emb / norms
    
    # 2. Centroid Norm (Directional bias)
    # If all vectors point in identical direction, centroid norm is 1.0. If uniformly spread, ~ 1/sqrt(N) -> ~0.0
    centroid = torch.mean(normed, dim=0)
    centroid_norm = float(torch.norm(centroid, p=2).item())
    
    # 3. Pairwise Cosine Similarity (Sampled)
    # Cosine sim matrix for subsample
    sim_matrix = torch.matmul(normed, normed.t())
    # Extract off-diagonal elements
    mask = ~torch.eye(n_sub, dtype=torch.bool, device=sim_matrix.device)
    off_diag_sims = sim_matrix[mask].cpu().numpy()
    
    mean_cos = float(np.mean(off_diag_sims))
    std_cos = float(np.std(off_diag_sims))
    median_cos = float(np.median(off_diag_sims))
    min_cos = float(np.min(off_diag_sims))
    max_cos = float(np.max(off_diag_sims))
    p5_cos = float(np.percentile(off_diag_sims, 5))
    p95_cos = float(np.percentile(off_diag_sims, 95))
    
    # 4. SVD & Effective Rank
    # Center the normalized embeddings or use raw normalized
    # Compute SVD of normalized embeddings (N_sub, D)
    try:
        # torch.linalg.svdvals
        s = torch.linalg.svdvals(normed)
        s_np = s.cpu().numpy()
        var_explained = (s_np ** 2) / np.sum(s_np ** 2)
        top1_var = float(var_explained[0])
        top5_var = float(np.sum(var_explained[:min(5, len(var_explained))]))
        top10_var = float(np.sum(var_explained[:min(10, len(var_explained))]))
        
        # Effective Rank (Roy & Vetterli): exp( - sum p_i ln p_i ) where p_i = s_i / sum s_i
        p_s = s_np / np.sum(s_np)
        p_s = p_s[p_s > 1e-12]
        entropy = -np.sum(p_s * np.log(p_s))
        effective_rank = float(np.exp(entropy))
    except Exception as e:
        top1_var, top5_var, top10_var, effective_rank = None, None, None, None
        
    # 5. Wang & Isola Uniformity Metric: log E[ exp(-2 * ||x - y||^2) ]
    # ||x - y||^2 = 2 - 2 * cos(x, y)
    # exp(-2 * ||x - y||^2) = exp(-4 + 4 * cos(x, y))
    sq_dist = 2.0 - 2.0 * off_diag_sims
    # Sample 100k pairs to avoid overflow / memory
    if len(sq_dist) > 200000:
        sample_pairs = np.random.choice(sq_dist, size=200000, replace=False)
    else:
        sample_pairs = sq_dist
    uniformity = float(np.log(np.mean(np.exp(-2.0 * sample_pairs))))
    
    return {
        "name": name,
        "total_count": N,
        "dimension": D,
        "sample_analyzed": n_sub,
        "centroid_norm": round(centroid_norm, 4),
        "cosine_similarity": {
            "mean": round(mean_cos, 4),
            "std": round(std_cos, 4),
            "median": round(median_cos, 4),
            "p5": round(p5_cos, 4),
            "p95": round(p95_cos, 4),
            "min": round(min_cos, 4),
            "max": round(max_cos, 4)
        },
        "spectrum": {
            "effective_rank": round(effective_rank, 2) if effective_rank is not None else None,
            "effective_rank_ratio": round(effective_rank / D, 4) if effective_rank is not None else None,
            "top1_var_explained_pct": round(top1_var * 100, 2) if top1_var is not None else None,
            "top5_var_explained_pct": round(top5_var * 100, 2) if top5_var is not None else None,
            "top10_var_explained_pct": round(top10_var * 100, 2) if top10_var is not None else None
        },
        "wang_isola_uniformity": round(uniformity, 4)
    }

def analyze_cross_similarity(users: torch.Tensor, items: torch.Tensor, sample_u: int = 2000, sample_i: int = 5000):
    """Analyzes User-Item cross cosine similarity distribution and top-1 / top-k collision."""
    Nu, Du = users.shape
    Ni, Di = items.shape
    
    idx_u = torch.randperm(Nu)[:min(sample_u, Nu)]
    idx_i = torch.randperm(Ni)[:min(sample_i, Ni)]
    
    u_sub = F.normalize(users[idx_u].float(), p=2, dim=-1)
    i_sub = F.normalize(items[idx_i].float(), p=2, dim=-1)
    
    # Cross similarity matrix (sample_u, sample_i)
    sims = torch.matmul(u_sub, i_sub.t()) # (Nu_sub, Ni_sub)
    sims_flat = sims.flatten().cpu().numpy()
    
    # Top-1 candidate frequency skew: do users all pick the exact same items?
    top1_indices = torch.argmax(sims, dim=-1).cpu().numpy()
    unique_top1 = len(np.unique(top1_indices))
    
    # Top-5 candidate unique count
    top5_indices = torch.topk(sims, k=min(5, sims.shape[1]), dim=-1).indices.cpu().numpy().flatten()
    unique_top5 = len(np.unique(top5_indices))
    
    return {
        "cross_cosine_mean": round(float(np.mean(sims_flat)), 4),
        "cross_cosine_std": round(float(np.std(sims_flat)), 4),
        "cross_cosine_median": round(float(np.median(sims_flat)), 4),
        "cross_cosine_p95": round(float(np.percentile(sims_flat, 95)), 4),
        "user_sample": len(idx_u),
        "item_sample": len(idx_i),
        "unique_top1_items": unique_top1,
        "unique_top1_ratio": round(unique_top1 / len(idx_u), 4),
        "unique_top5_items": unique_top5,
        "unique_top5_ratio": round(unique_top5 / (len(idx_u) * 5), 4)
    }

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    datasets = ["amazon_beauty", "movielens", "yelp", "amazon_cd_2014", "amazon_toys_and_games"]
    results = {}
    
    for ds in datasets:
        print(f"\n=======================================================")
        print(f" ANALYZING DATASET: {ds.upper()}")
        print(f"=======================================================")
        results[ds] = {}
        
        # ----------------------------------------------------
        # 1. PIPELINE 2 (LLMEmb)
        # ----------------------------------------------------
        p2_item_path = PROJECT_ROOT / "data" / "cache" / "llmemb" / ds / "item_emb_llmemb.npy"
        p2_user_path = PROJECT_ROOT / "data" / "cache" / "llmemb" / ds / "user_emb_llmemb.npy"
        
        if p2_item_path.exists() and p2_user_path.exists():
            print(f"[Pipeline 2 - LLMEmb] Loading embeddings...")
            p2_items = torch.from_numpy(np.load(p2_item_path, mmap_mode="r"))
            p2_users = torch.from_numpy(np.load(p2_user_path, mmap_mode="r"))
            
            p2_item_metrics = compute_anisotropy_metrics(p2_items, f"{ds} - Pipeline 2 Items (LLMEmb)")
            p2_user_metrics = compute_anisotropy_metrics(p2_users, f"{ds} - Pipeline 2 Users (LLMEmb)")
            p2_cross = analyze_cross_similarity(p2_users, p2_items)
            
            results[ds]["pipeline_2_llmemb"] = {
                "items": p2_item_metrics,
                "users": p2_user_metrics,
                "cross_user_item": p2_cross
            }
            print(f"  P2 Items: Dim={p2_items.shape[1]}, Mean Cos={p2_item_metrics['cosine_similarity']['mean']}, Centroid Norm={p2_item_metrics['centroid_norm']}, EffRank={p2_item_metrics['spectrum']['effective_rank']}")
            print(f"  P2 Users: Dim={p2_users.shape[1]}, Mean Cos={p2_user_metrics['cosine_similarity']['mean']}, Centroid Norm={p2_user_metrics['centroid_norm']}, EffRank={p2_user_metrics['spectrum']['effective_rank']}")
            print(f"  P2 Cross: Mean Cos={p2_cross['cross_cosine_mean']}, Unique Top-1={p2_cross['unique_top1_items']}/{p2_cross['user_sample']}")
        else:
            print(f"[Pipeline 2 - LLMEmb] Files not found for {ds}")
            
        # ----------------------------------------------------
        # 2. PIPELINE 3 (Llama-3-8B-Instruct)
        # ----------------------------------------------------
        p3_item_path = PROJECT_ROOT / "data" / "cache" / "Llama-3-8B-Instruct" / ds / "item_embeddings" / "item_text_emb.npy"
        p3_user_path = PROJECT_ROOT / "data" / "cache" / "Llama-3-8B-Instruct" / ds / "intents" / "user_intent_text_emb.npy"
        
        if p3_item_path.exists() and p3_user_path.exists():
            print(f"[Pipeline 3 - Llama-3] Loading embeddings...")
            p3_items = torch.from_numpy(np.load(p3_item_path, mmap_mode="r"))
            p3_users = torch.from_numpy(np.load(p3_user_path, mmap_mode="r"))
            
            p3_item_metrics = compute_anisotropy_metrics(p3_items, f"{ds} - Pipeline 3 Items (Llama-3)")
            p3_user_metrics = compute_anisotropy_metrics(p3_users, f"{ds} - Pipeline 3 Users (Llama-3)")
            p3_cross = analyze_cross_similarity(p3_users, p3_items)
            
            results[ds]["pipeline_3_llama3"] = {
                "items": p3_item_metrics,
                "users": p3_user_metrics,
                "cross_user_item": p3_cross
            }
            print(f"  P3 Items: Dim={p3_items.shape[1]}, Mean Cos={p3_item_metrics['cosine_similarity']['mean']}, Centroid Norm={p3_item_metrics['centroid_norm']}, EffRank={p3_item_metrics['spectrum']['effective_rank']}")
            print(f"  P3 Users: Dim={p3_users.shape[1]}, Mean Cos={p3_user_metrics['cosine_similarity']['mean']}, Centroid Norm={p3_user_metrics['centroid_norm']}, EffRank={p3_user_metrics['spectrum']['effective_rank']}")
            print(f"  P3 Cross: Mean Cos={p3_cross['cross_cosine_mean']}, Unique Top-1={p3_cross['unique_top1_items']}/{p3_cross['user_sample']}")
        else:
            print(f"[Pipeline 3 - Llama-3] Files not found for {ds}")

    # Output json results
    out_file = PROJECT_ROOT / "data" / "reports" / "anisotropy_analysis.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Done] Results successfully written to: {out_file}")

if __name__ == "__main__":
    main()
