import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Ensure root workspace is accessible
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

DEFAULT_DATASETS = ["movielens", "amazon_cd_2014", "amazon_toys_and_games", "amazon_beauty", "yelp"]


def analyze_dataset_similarity_distribution(
    dataset_name: str,
    teacher_name: str = "Llama-3-8B-Instruct",
    top_k: int = 5,
    sample_users: int = 2000,
    random_samples: int = 20000,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Analyzes the semantic cosine similarity distribution between LLM user intents
    and item embeddings:
    1. Background / Random noise distribution (uncorrelated user-item pairs)
    2. Candidate Top-K semantic similarities (potential pseudo-edges)
    3. Calculates optimal statistical threshold suggestions (tau).
    """
    cache_dir = PROJECT_ROOT / "data" / "cache" / Path(teacher_name).name / dataset_name
    u_path = cache_dir / "intents" / "user_intent_text_emb.npy"
    i_path = cache_dir / "item_embeddings" / "item_text_emb.npy"

    if not (u_path.exists() and i_path.exists()):
        raise FileNotFoundError(f"Embeddings not found for '{dataset_name}' at {cache_dir}.")

    logging.info(f"[{dataset_name}] Loading user intents from: {u_path.name}")
    u_raw = np.load(u_path)
    logging.info(f"[{dataset_name}] Loading item embeddings from: {i_path.name}")
    i_raw = np.load(i_path)

    u_tensor = torch.tensor(u_raw, dtype=torch.float32)
    i_tensor = torch.tensor(i_raw, dtype=torch.float32)

    num_users, dim_u = u_tensor.shape
    num_items, dim_i = i_tensor.shape
    logging.info(f"[{dataset_name}] Dataset size: {num_users} users, {num_items} items (dim={dim_u})")

    # L2 Normalization
    u_norm = F.normalize(u_tensor, p=2, dim=-1)
    i_norm = F.normalize(i_tensor, p=2, dim=-1)

    # Subsample users for efficiency if population is large
    n_sample = min(sample_users, num_users)
    torch.manual_seed(42)
    selected_u_idx = torch.randperm(num_users)[:n_sample]
    sample_u = u_norm[selected_u_idx]

    # Compute similarity matrix (n_sample, num_items)
    batch_size = 512
    topk_sims_list = []
    rand_sims_list = []

    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    i_norm_dev = i_norm.to(dev)

    for start in range(0, n_sample, batch_size):
        end = min(start + batch_size, n_sample)
        batch_u = sample_u[start:end].to(dev)
        sim_batch = torch.matmul(batch_u, i_norm_dev.T) # (batch, num_items)

        # Top-K candidate scores
        vals, _ = torch.topk(sim_batch, k=min(top_k, num_items), dim=-1)
        topk_sims_list.append(vals.cpu())

        # Random sample for background noise
        flat_sim = sim_batch.reshape(-1)
        n_rand = min(len(flat_sim), max(100, random_samples // (n_sample // batch_size + 1)))
        rand_idx = torch.randint(0, len(flat_sim), (n_rand,))
        rand_sims_list.append(flat_sim[rand_idx].cpu())

    topk_all = torch.cat(topk_sims_list, dim=0).flatten().numpy()
    rand_all = torch.cat(rand_sims_list, dim=0).numpy()

    # Statistical metrics
    rand_mean = float(np.mean(rand_all))
    rand_std = float(np.std(rand_all))
    rand_95 = float(np.percentile(rand_all, 95))
    rand_99 = float(np.percentile(rand_all, 99))

    topk_mean = float(np.mean(topk_all))
    topk_median = float(np.median(topk_all))
    topk_std = float(np.std(topk_all))
    topk_min = float(np.min(topk_all))
    topk_max = float(np.max(topk_all))
    topk_q1 = float(np.percentile(topk_all, 25))
    topk_q3 = float(np.percentile(topk_all, 75))
    topk_p10 = float(np.percentile(topk_all, 10))

    # Suggested Tau Thresholds
    # 1. Conservative: Mean of random noise + 2*sigma (exceeds ~97.7% of random noise)
    tau_noise_2sigma = float(rand_mean + 2 * rand_std)
    # 2. Moderate: 95th percentile of random noise or 25th percentile of candidates
    tau_noise_95p = float(rand_95)
    tau_candidate_q1 = float(topk_q1)
    tau_candidate_median = float(topk_median)

    stats = {
        "dataset": dataset_name,
        "teacher": teacher_name,
        "num_users": num_users,
        "num_items": num_items,
        "top_k": top_k,
        "random_noise": {
            "mean": rand_mean,
            "std": rand_std,
            "min": float(np.min(rand_all)),
            "max": float(np.max(rand_all)),
            "p95": rand_95,
            "p99": rand_99
        },
        "candidate_topk": {
            "mean": topk_mean,
            "median": topk_median,
            "std": topk_std,
            "min": topk_min,
            "max": topk_max,
            "p10": topk_p10,
            "q1": topk_q1,
            "q3": topk_q3
        },
        "suggested_tau": {
            "conservative_2sigma": tau_noise_2sigma,
            "noise_95_percentile": tau_noise_95p,
            "candidate_first_quartile": tau_candidate_q1,
            "candidate_median": tau_candidate_median
        }
    }

    return stats, topk_all, rand_all


def plot_similarity_distribution(
    dataset_name: str,
    topk_sims: np.ndarray,
    rand_sims: np.ndarray,
    stats: Dict[str, Any],
    save_path: Path
):
    """Plots probability density and cumulative distribution for noise vs candidates."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # 1. Probability Density (Histogram / KDE)
    bins = np.linspace(min(rand_sims.min(), topk_sims.min()), max(rand_sims.max(), topk_sims.max()), 60)
    ax1.hist(rand_sims, bins=bins, density=True, alpha=0.45, color="#7f7f7f", label="Random Pairs (Noise Floor)")
    ax1.hist(topk_sims, bins=bins, density=True, alpha=0.6, color="#1f77b4", label=f"Top-{stats['top_k']} Candidates")

    # Mark suggested thresholds
    tau_2s = stats["suggested_tau"]["conservative_2sigma"]
    tau_med = stats["suggested_tau"]["candidate_median"]
    ax1.axvline(tau_2s, color="#d62728", linestyle="--", linewidth=1.8, label=f"Noise + 2σ (τ={tau_2s:.3f})")
    ax1.axvline(tau_med, color="#2ca02c", linestyle=":", linewidth=1.8, label=f"Candidate Median (τ={tau_med:.3f})")

    ax1.set_title(f"Cosine Similarity Density: {dataset_name.upper()}", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Cosine Similarity", fontsize=10)
    ax1.set_ylabel("Probability Density", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. Cumulative Distribution Function (CDF)
    sorted_topk = np.sort(topk_sims)
    cdf_topk = np.arange(1, len(sorted_topk) + 1) / len(sorted_topk)
    sorted_rand = np.sort(rand_sims)
    cdf_rand = np.arange(1, len(sorted_rand) + 1) / len(sorted_rand)

    ax2.plot(sorted_rand, cdf_rand, color="#7f7f7f", linestyle="--", linewidth=1.5, label="Random Pairs CDF")
    ax2.plot(sorted_topk, cdf_topk, color="#1f77b4", linewidth=2.0, label=f"Top-{stats['top_k']} Candidates CDF")
    ax2.axvline(tau_2s, color="#d62728", linestyle="--", linewidth=1.8, label=f"Cutoff Noise+2σ (τ={tau_2s:.3f})")

    ax2.set_title("Cumulative Distribution Function (CDF)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Cosine Similarity", fontsize=10)
    ax2.set_ylabel("Cumulative Fraction", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved distribution plot to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Semantic Cosine Similarity Distribution and Determine Optimal Tau Threshold")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset name ('all' or specific dataset)")
    parser.add_argument("--teacher", type=str, default="Llama-3-8B-Instruct", help="Teacher LLM name")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K candidate similarities per user")
    parser.add_argument("--output_dir", type=str, default="data/reports/similarity_analysis", help="Output directory")
    args = parser.parse_args()

    datasets = DEFAULT_DATASETS if args.dataset == "all" else [args.dataset]
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_results = {}

    for ds in datasets:
        print("\n" + "=" * 80)
        print(f" ANALYZING COSINE SIMILARITY DISTRIBUTION: {ds.upper()}")
        print("=" * 80)
        try:
            stats, topk_sims, rand_sims = analyze_dataset_similarity_distribution(
                dataset_name=ds,
                teacher_name=args.teacher,
                top_k=args.top_k
            )
            summary_results[ds] = stats

            print(f"Background Noise (Random Pairs):  Mean={stats['random_noise']['mean']:.4f} +/- {stats['random_noise']['std']:.4f} | 95th%={stats['random_noise']['p95']:.4f}")
            print(f"Top-{args.top_k} Candidate Similarities: Mean={stats['candidate_topk']['mean']:.4f} | Median={stats['candidate_topk']['median']:.4f} | Q1={stats['candidate_topk']['q1']:.4f} | Q3={stats['candidate_topk']['q3']:.4f}")
            print(f"Suggested Thresholds:")
            print(f"  -> Noise Floor + 2*sigma (Conservative): tau = {stats['suggested_tau']['conservative_2sigma']:.4f}")
            print(f"  -> Noise 95th Percentile:                tau = {stats['suggested_tau']['noise_95_percentile']:.4f}")
            print(f"  -> Candidate Q1 (Bottom 25% filter):     tau = {stats['suggested_tau']['candidate_first_quartile']:.4f}")
            print(f"  -> Candidate Median:                     tau = {stats['suggested_tau']['candidate_median']:.4f}")

            # Plot
            plot_path = out_dir / f"sim_dist_{ds}.png"
            plot_similarity_distribution(ds, topk_sims, rand_sims, stats, plot_path)

        except Exception as e:
            logging.error(f"Failed analyzing {ds}: {e}")

    # Save summary JSON
    json_path = out_dir / "similarity_distributions_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    print(f"\n[Done] Summary JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
