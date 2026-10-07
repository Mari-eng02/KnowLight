import os
import sys
import re
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import concurrent.futures
import multiprocessing as mp

import torch
import scipy.sparse as sp
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowlight.paths import (
    DATA_DIR,
    RAW_DATA_DIR,
    CACHE_DIR,
    STUDENTS_DIR,
)
from knowlight.utils.helpers import set_seed, load_yaml_config
from knowlight.datasets.data_loader import KnowLightDataLoader
from knowlight.datasets import BPRDataset
from knowlight.student.lightgcn import LightGCN
from knowlight.training.evaluator import Evaluator
from knowlight.evaluation.sparsity_evaluator import SparsityEvaluator

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

CANONICAL_DATASETS = [
    "movielens",
    "amazon_cd_2014",
    "amazon_toys_and_games",
    "amazon_beauty",
    "yelp"
]


def _extract_run_number(path: Path) -> int:
    """Extracts trailing integer from directory name (e.g. 'baseline_run_3' -> 3)."""
    match = re.search(r'(\d+)$', path.name)
    return int(match.group(1)) if match else -1


def find_baseline_checkpoint(dataset_name: str) -> Optional[Path]:
    """Finds the latest existing LightGCN baseline checkpoint for a dataset (e.g. baseline_run_3)."""
    base_student_dir = STUDENTS_DIR / "lightgcn" / dataset_name
    if not base_student_dir.exists():
        return None

    # Sort all run directories by numerical index descending to prioritize the latest run
    subdirs = [d for d in base_student_dir.iterdir() if d.is_dir()]
    subdirs.sort(key=_extract_run_number, reverse=True)

    for run_dir in subdirs:
        for pt_file in ["best_student_lightgcn.pt", "checkpoint_best_student.pt", "best_student_model.pt"]:
            candidate = run_dir / pt_file
            if candidate.exists():
                logging.info(f"[Auto-Discovery] Selected latest baseline run: '{run_dir.name}' ({candidate.name})")
                return candidate

    return None


def get_next_report_files(target_dir: Path, base_prefix: str, dataset_name: str) -> Tuple[Path, Path]:
    """
    Finds the next available report file paths (.json and .md) using sequential numbering:
    - 1st: {base_prefix}_{dataset_name}.json / .md
    - 2nd: {base_prefix}_{dataset_name}_2.json / .md
    - 3rd: {base_prefix}_{dataset_name}_3.json / .md
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    base_stem = f"{base_prefix}_{dataset_name}"
    first_json = target_dir / f"{base_stem}.json"
    first_md = target_dir / f"{base_stem}.md"

    if not first_json.exists() and not first_md.exists():
        return first_json, first_md

    pattern = re.compile(rf"^{re.escape(base_stem)}_(\d+)\.(?:json|md)$")
    existing_nums = [1]
    for f in target_dir.iterdir():
        if f.is_file():
            m = pattern.match(f.name)
            if m:
                existing_nums.append(int(m.group(1)))

    next_num = max(existing_nums) + 1
    return target_dir / f"{base_stem}_{next_num}.json", target_dir / f"{base_stem}_{next_num}.md"


def find_best_intent_params(
    dataset_name: str,
    pipeline_ckpt_path: Optional[str] = None,
    aug_graph_path: Optional[str] = None
) -> Tuple[Dict[str, Any], Optional[Path]]:
    """
    Reads the grid search results for intent graph augmentation from the specified or best run
    and extracts best_params and the corresponding augmented graph .npz path.

    If pipeline_ckpt_path is provided, prioritizes reading grid_aug_results.json from
    the checkpoint's directory to ensure model weights and graph topology are strictly aligned.
    Otherwise, evaluates all available grid runs and selects the one with the highest NDCG.
    """
    intent_dir = STUDENTS_DIR / "intent_augmented" / dataset_name
    grid_res_file = None
    target_run_num = None

    # 1. If explicit pipeline_ckpt_path is provided, inspect its parent directory first
    if pipeline_ckpt_path:
        ckpt_p = Path(pipeline_ckpt_path).resolve()
        candidate = ckpt_p.parent / "grid_aug_results.json"
        if candidate.exists():
            grid_res_file = candidate
            target_run_num = _extract_run_number(ckpt_p.parent)
            logging.info(f"[Explicit Checkpoint] Found grid results matching checkpoint in: '{ckpt_p.parent.name}'")
        elif (ckpt_p.parent.parent / "grid_aug_results.json").exists():
            grid_res_file = ckpt_p.parent.parent / "grid_aug_results.json"
            target_run_num = _extract_run_number(ckpt_p.parent.parent)
            logging.info(f"[Explicit Checkpoint] Found grid results in parent dir: '{ckpt_p.parent.parent.name}'")

    # 2. Auto-discovery across all grid runs (select highest best_ndcg)
    if not grid_res_file and intent_dir.exists():
        subdirs = [d for d in intent_dir.iterdir() if d.is_dir() and (d / "grid_aug_results.json").exists()]
        if subdirs:
            runs_info = []
            for run_dir in subdirs:
                try:
                    with open(run_dir / "grid_aug_results.json", "r", encoding="utf-8") as f:
                        data_tmp = json.load(f)
                    ndcg = float(data_tmp.get("best_ndcg", 0.0))
                    r_num = _extract_run_number(run_dir)
                    runs_info.append((ndcg, r_num, run_dir))
                except Exception:
                    continue
            if runs_info:
                # Sort descending by NDCG, then by run number
                runs_info.sort(key=lambda x: (x[0], x[1]), reverse=True)
                best_ndcg_val, best_r_num, best_run_dir = runs_info[0]
                grid_res_file = best_run_dir / "grid_aug_results.json"
                target_run_num = best_r_num
                logging.info(f"[Auto-Discovery] Selected best grid results run: '{best_run_dir.name}' (Best NDCG@20: {best_ndcg_val:.4f})")

    if not grid_res_file or not grid_res_file.exists():
        raise FileNotFoundError(f"Could not find grid_aug_results.json in {intent_dir}")

    with open(grid_res_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    best_params = data.get("best_params")
    if not best_params:
        raise ValueError(f"No 'best_params' found in {grid_res_file}")

    top_k = int(best_params["top_k_pseudo"])
    weight = float(best_params["pseudo_weight"])
    sim_threshold = float(best_params.get("sim_threshold", data.get("sim_threshold", 0.0)))
    teacher_name = data.get("teacher", "Llama-3-8B-Instruct")

    # 3. Locate cached augmented graph
    if aug_graph_path:
        graph_path = Path(aug_graph_path)
        if not graph_path.exists():
            raise FileNotFoundError(f"Explicitly specified --aug_graph not found: {graph_path}")
        logging.info(f"[Explicit Graph] Using provided augmented graph: {graph_path}")
    else:
        candidate_cache_dirs = []
        for t_name in [teacher_name, "Llama-3-8B-Instruct", "Qwen3.5-4B"]:
            if not t_name:
                continue
            base_c = CACHE_DIR / Path(t_name).name / dataset_name
            if not base_c.exists():
                continue
            # Prioritize cache directory corresponding to the specific grid run number
            if target_run_num is not None and target_run_num != -1:
                target_cache = base_c / f"augmented_graphs_{target_run_num}"
                if target_cache.is_dir() and target_cache not in candidate_cache_dirs:
                    candidate_cache_dirs.append(target_cache)
            for d in sorted(base_c.glob("augmented_graphs*"), key=_extract_run_number, reverse=True):
                if d.is_dir() and d not in candidate_cache_dirs:
                    candidate_cache_dirs.append(d)

        graph_filename = f"aug_r_k{top_k}_w{weight}_th{sim_threshold}.npz"
        graph_path = None
        for c_dir in candidate_cache_dirs:
            possible_file = c_dir / graph_filename
            if possible_file.exists():
                graph_path = possible_file
                break
            # Fallback glob if exact float string formatting differs
            glob_matches = list(c_dir.glob(f"aug_r_k{top_k}_w{weight}_th*.npz"))
            if glob_matches:
                graph_path = glob_matches[0]
                break

    return {
        "top_k_pseudo": top_k,
        "pseudo_weight": weight,
        "sim_threshold": sim_threshold,
        "best_ndcg": data.get("best_ndcg", 0.0),
        "teacher": teacher_name,
        "grid_results_path": str(grid_res_file)
    }, graph_path


def train_single_intent_augmented_student(
    dataset_name: str,
    augmented_r: sp.csr_matrix,
    data_loader: KnowLightDataLoader,
    config: Dict[str, Any],
    save_path: Path,
    device: torch.device
) -> LightGCN:
    """
    Trains a single LightGCN student with early stopping using the best augmented graph
    and saves the weights to save_path.
    """
    logging.info(f"[Fast Trainer] Training best Pipeline 3 LightGCN student for '{dataset_name}'...")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    num_users = data_loader.num_users
    num_items = data_loader.num_items

    student_cfg = config.get("student", {})
    embedding_dim = student_cfg.get("embedding_dim", 64)
    n_layers = student_cfg.get("n_layers", 3)
    keep_prob = student_cfg.get("keep_prob", 0.6)

    student = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        keep_prob=keep_prob
    ).to(device)

    student.set_graph(augmented_r, device=device)

    # Read hyperparameters from dataset_hyperparams in config
    dataset_overrides = config.get("dataset_hyperparams", {}).get(dataset_name, {})
    lr = float(dataset_overrides.get("lr", config.get("training", {}).get("lr", 0.001)))
    weight_decay = float(dataset_overrides.get("weight_decay", config.get("training", {}).get("weight_decay", 0.001)))

    logging.info(f"[Fast Trainer] Hyperparameters -> LR: {lr} | Weight Decay: {weight_decay} | Embedding Dim: {embedding_dim}")

    optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=0.0)
    evaluator = Evaluator(top_k_list=config.get("evaluation", {}).get("top_k", [5, 10, 20]))

    bpr_ds = BPRDataset(num_users, num_items, train_dict=data_loader.train_dict, train_matrix=data_loader.train_matrix)
    batch_size = config.get("training", {}).get("batch_size", 1024)
    train_loader = DataLoader(bpr_ds, batch_size=batch_size, shuffle=True)

    epochs = config.get("training", {}).get("epochs", 1000)
    eval_step = config.get("training", {}).get("eval_step", 1)
    patience = config.get("training", {}).get("patience", 10)

    best_val_ndcg = 0.0
    patience_counter = 0
    best_weights = None

    val_target = getattr(data_loader, "val_pos_items", getattr(data_loader, "val_dict", {}))

    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0.0
        start_time = time.time()

        for batch_users, batch_pos, batch_neg in train_loader:
            batch_users = batch_users.to(device)
            batch_pos = batch_pos.to(device)
            batch_neg = batch_neg.to(device)

            optimizer.zero_grad()
            loss = student.bpr_loss(batch_users, batch_pos, batch_neg, decay=weight_decay)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_duration = time.time() - start_time
        avg_loss = total_loss / len(train_loader)

        if epoch % eval_step == 0:
            student.eval()
            with torch.no_grad():
                val_metrics = evaluator.evaluate(
                    model=student,
                    data_loader=data_loader,
                    test_pos_items=val_target,
                    device=device
                )
            val_ndcg = val_metrics.get("NDCG@20", 0.0)
            logging.info(f"Epoch {epoch:03d}/{epochs:03d} [{epoch_duration:.1f}s] | Loss: {avg_loss:.4f} | Val NDCG@20: {val_ndcg:.4f}")

            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                best_weights = {k: v.cpu().clone() for k, v in student.state_dict().items()}
                patience_counter = 0
                torch.save(best_weights, save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"[Fast Trainer] Early stopping at epoch {epoch} (Best Val NDCG@20: {best_val_ndcg:.4f})")
                    break

    if best_weights is not None:
        student.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    logging.info(f"[Fast Trainer] Model successfully trained and saved to: {save_path}")
    return student


def run_sparsity_evaluation(
    dataset_name: str = "movielens",
    config_path: str = "config/train_intent_augmented_lightgcn.yaml",
    baseline_ckpt_path: Optional[str] = None,
    pipeline_ckpt_path: Optional[str] = None,
    aug_graph_path: Optional[str] = None,
    output_dir: str = "data/reports",
    force_retrain_pipeline: bool = False,
    device_name: Optional[str] = None
):
    """
    Executes Degree-Stratified / Sparsity Analysis:
    1. Loads dataset and partitions test users into 3 homogeneous buckets.
    2. Loads Baseline LightGCN and evaluates on the buckets.
    3. Loads (or quickly trains) Pipeline 3 LightGCN and evaluates on the buckets.
    4. Computes Delta% improvements and generates JSON + Markdown reports in data/reports/.
    """
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info(f"=== SPARSITY ANALYSIS (DEGREE-STRATIFIED EVALUATION) ===")
    logging.info(f"Dataset: '{dataset_name}' | Device: {device}")

    # 1. Load config
    config = load_yaml_config(config_path, dataset_name=dataset_name)
    set_seed(config.get("experiment", {}).get("seed", 42))

    # 2. Load dataset
    data_dir = config.get("dataset", {}).get("data_dir", "./data/raw")
    data_loader = KnowLightDataLoader(dataset_name=dataset_name, data_dir=data_dir)
    data_loader.load_data()

    num_users = data_loader.num_users
    num_items = data_loader.num_items
    logging.info(f"Loaded {num_users} users and {num_items} items.")

    # 3. Partition users into 3 homogeneous buckets
    sp_evaluator = SparsityEvaluator(top_k_list=config.get("evaluation", {}).get("top_k", [5, 10, 20]))
    buckets = sp_evaluator.partition_users_into_homogeneous_buckets(data_loader=data_loader, num_buckets=3)

    print("\n" + "=" * 80)
    print(f" HOMOGENEOUS TEST USER BUCKETS (3 TERCILES) - {dataset_name.upper()}")
    print("=" * 80)
    print(f"{'Bucket':<32} | {'Users':<8} | {'Min Deg':<8} | {'Max Deg':<8} | {'Mean Deg':<10} | {'Median Deg':<10}")
    print("-" * 80)
    for b_key, b in buckets.items():
        print(f"{b['label']:<32} | {b['num_users']:<8} | {b['deg_min']:<8} | {b['deg_max']:<8} | {b['deg_mean']:<10.1f} | {b['deg_median']:<10.1f}")
    print("=" * 80 + "\n")

    # 4. Load Baseline Model
    if baseline_ckpt_path:
        base_ckpt = Path(baseline_ckpt_path)
    else:
        base_ckpt = find_baseline_checkpoint(dataset_name)

    if not base_ckpt or not base_ckpt.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found for dataset '{dataset_name}'. Looked in {STUDENTS_DIR / 'lightgcn' / dataset_name}")

    logging.info(f"Loading Baseline LightGCN from: {base_ckpt}")
    student_cfg = config.get("student", {})
    baseline_model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=student_cfg.get("embedding_dim", 64),
        n_layers=student_cfg.get("n_layers", 3),
        keep_prob=student_cfg.get("keep_prob", 0.6)
    ).to(device)

    base_state = torch.load(base_ckpt, map_location=device, weights_only=False)
    baseline_model.load_state_dict(base_state)
    baseline_model.set_graph(data_loader.train_matrix, device=device)

    # 5. Pipeline 3 Best Model Resolution
    if pipeline_ckpt_path:
        pipe_ckpt = Path(pipeline_ckpt_path)
    else:
        pipe_ckpt = None

    best_info, aug_graph_file = find_best_intent_params(
        dataset_name=dataset_name,
        pipeline_ckpt_path=pipeline_ckpt_path,
        aug_graph_path=aug_graph_path
    )
    logging.info(f"Pipeline 3 Configuration: Top-K={best_info['top_k_pseudo']}, Pseudo-Weight={best_info['pseudo_weight']}, Threshold={best_info['sim_threshold']} (Best NDCG@20: {best_info['best_ndcg']:.4f})")

    if not aug_graph_file or not aug_graph_file.exists():
        raise FileNotFoundError(f"Could not find cached augmented graph for top_k={best_info['top_k_pseudo']}, weight={best_info['pseudo_weight']}, threshold={best_info['sim_threshold']}")

    logging.info(f"Loading cached Augmented Graph from: {aug_graph_file.name}")
    augmented_r = sp.load_npz(aug_graph_file)

    # Determine Pipeline 3 checkpoint location if not explicitly provided
    if not pipe_ckpt:
        grid_run_dir = Path(best_info["grid_results_path"]).parent
        grid_run_ckpt = grid_run_dir / "best_student_model.pt"
        if grid_run_ckpt.exists():
            pipe_ckpt = grid_run_ckpt
        else:
            pipe_ckpt = STUDENTS_DIR / "intent_augmented" / dataset_name / "best_model" / "best_student_model.pt"

    pipeline_model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=student_cfg.get("embedding_dim", 64),
        n_layers=student_cfg.get("n_layers", 3),
        keep_prob=student_cfg.get("keep_prob", 0.6)
    ).to(device)

    if not pipe_ckpt.exists() or force_retrain_pipeline:
        logging.info(f"Pipeline 3 checkpoint not found at {pipe_ckpt}. Launching fast single-run training with best augmented graph...")
        pipeline_model = train_single_intent_augmented_student(
            dataset_name=dataset_name,
            augmented_r=augmented_r,
            data_loader=data_loader,
            config=config,
            save_path=pipe_ckpt,
            device=device
        )
    else:
        logging.info(f"Loading existing Pipeline 3 checkpoint from: {pipe_ckpt}")
        pipe_state = torch.load(pipe_ckpt, map_location=device, weights_only=False)
        pipeline_model.load_state_dict(pipe_state)

    pipeline_model.set_graph(augmented_r, device=device)

    # 6. Evaluate Models on Buckets
    logging.info("Evaluating Baseline LightGCN across all 3 buckets...")
    baseline_metrics = sp_evaluator.evaluate_model_on_buckets(
        model=baseline_model,
        data_loader=data_loader,
        buckets=buckets,
        device=device
    )

    logging.info("Evaluating Pipeline 3 (Graph Augmentation) across all 3 buckets...")
    pipeline_metrics = sp_evaluator.evaluate_model_on_buckets(
        model=pipeline_model,
        data_loader=data_loader,
        buckets=buckets,
        device=device
    )

    deltas = sp_evaluator.compute_deltas(baseline_metrics, pipeline_metrics)

    # 7. Print Terminal Comparative Table
    print("\n" + "=" * 95)
    print(f" SPARSITY EVALUATION RESULTS: BASELINE vs PIPELINE 3 (GRAPH AUGMENTATION)")
    print(f" Dataset: {dataset_name.upper()} | Best Params: k={best_info['top_k_pseudo']}, weight={best_info['pseudo_weight']}")
    print("=" * 95)
    print(f"{'Cohort / Bucket':<30} | {'Model':<12} | {'Recall@5':<10} | {'Recall@20':<10} | {'NDCG@5':<10} | {'NDCG@20':<10}")
    print("-" * 95)

    all_groups = list(buckets.keys()) + ["overall"]
    for g in all_groups:
        label = buckets[g]["label"] if g in buckets else "Full Test Set (Overall)"
        deg_info = f"(deg: {buckets[g]['deg_min']}-{buckets[g]['deg_max']})" if g in buckets else ""

        b_m = baseline_metrics[g]
        p_m = pipeline_metrics[g]
        d_m = deltas[g]

        print(f"{label + ' ' + deg_info:<30} | {'Baseline':<12} | {b_m['Recall@5']:<10.4f} | {b_m['Recall@20']:<10.4f} | {b_m['NDCG@5']:<10.4f} | {b_m['NDCG@20']:<10.4f}")
        print(f"{'':<30} | {'Pipeline 3':<12} | {p_m['Recall@5']:<10.4f} | {p_m['Recall@20']:<10.4f} | {p_m['NDCG@5']:<10.4f} | {p_m['NDCG@20']:<10.4f}")
        print(f"{'':<30} | {'Delta (%)':<12} | {d_m['Recall@5']:>+9.2f}% | {d_m['Recall@20']:>+9.2f}% | {d_m['NDCG@5']:>+9.2f}% | {d_m['NDCG@20']:>+9.2f}%")
        print("-" * 95)

    # 8. Save Reports (JSON and Markdown) in data/reports/sparsity_analysis/
    out_dir_path = PROJECT_ROOT / output_dir
    if "sparsity_analysis" not in str(out_dir_path):
        out_dir_path = out_dir_path / "sparsity_analysis"
    out_dir_path.mkdir(parents=True, exist_ok=True)

    json_report_file, md_report_file = get_next_report_files(out_dir_path, "sparsity_analysis", dataset_name)

    report_data = {
        "dataset": dataset_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_pipeline_params": best_info,
        "baseline_checkpoint": str(base_ckpt),
        "pipeline_checkpoint": str(pipe_ckpt),
        "buckets": {
            k: {
                "label": v["label"],
                "num_users": v["num_users"],
                "deg_min": v["deg_min"],
                "deg_max": v["deg_max"],
                "deg_mean": v["deg_mean"],
                "deg_median": v["deg_median"]
            } for k, v in buckets.items()
        },
        "baseline_metrics": baseline_metrics,
        "pipeline_metrics": pipeline_metrics,
        "delta_improvements_percent": deltas
    }

    with open(json_report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Markdown Report
    with open(md_report_file, "w", encoding="utf-8") as f:
        f.write(f"# Sparsity Analysis Report: {dataset_name.upper()}\n\n")
        f.write(f"- **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Dataset**: `{dataset_name}`\n")
        f.write(f"- **Pipeline 3 Best Params**: $k={best_info['top_k_pseudo']}$, $\\text{{weight}}={best_info['pseudo_weight']}$\n")
        f.write(f"- **Baseline Checkpoint**: `{base_ckpt.name}`\n\n")
        f.write("## 1. Test User Partitioning across Sparsity Buckets (Terciles)\n\n")
        f.write("| Bucket | Users | Min Degree | Max Degree | Mean Degree | Median Degree |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for b_key, b in buckets.items():
            f.write(f"| **{b['label']}** | {b['num_users']} | {b['deg_min']} | {b['deg_max']} | {b['deg_mean']:.1f} | {b['deg_median']:.1f} |\n")
        f.write("\n## 2. Comparative Performance (Baseline vs Pipeline 3)\n\n")
        f.write("| Cohort / Bucket | Model | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for g in all_groups:
            label = buckets[g]["label"] if g in buckets else "**Overall (All Test Users)**"
            bm = baseline_metrics[g]
            pm = pipeline_metrics[g]
            dm = deltas[g]
            f.write(f"| {label} | Baseline | {bm['Recall@5']:.4f} | {bm['Recall@10']:.4f} | {bm['Recall@20']:.4f} | {bm['NDCG@5']:.4f} | {bm['NDCG@10']:.4f} | {bm['NDCG@20']:.4f} |\n")
            f.write(f"| | **Pipeline 3** | **{pm['Recall@5']:.4f}** | **{pm['Recall@10']:.4f}** | **{pm['Recall@20']:.4f}** | **{pm['NDCG@5']:.4f}** | **{pm['NDCG@10']:.4f}** | **{pm['NDCG@20']:.4f}** |\n")
            f.write(f"| | *Delta (%)* | *{dm['Recall@5']:+.2f}%* | *{dm['Recall@10']:+.2f}%* | *{dm['Recall@20']:+.2f}%* | *{dm['NDCG@5']:+.2f}%* | *{dm['NDCG@10']:+.2f}%* | *{dm['NDCG@20']:+.2f}%* |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")

    logging.info(f"[{dataset_name}] Sparsity Analysis completed successfully!")
    logging.info(f"[{dataset_name}] JSON Report saved to: {json_report_file}")
    logging.info(f"[{dataset_name}] Markdown Report saved to: {md_report_file}")


def _run_single_dataset_worker(kwargs: Dict[str, Any]) -> Tuple[str, bool, Optional[str]]:
    """Worker wrapper to execute sparsity evaluation for a single dataset in a worker process."""
    ds = kwargs["dataset_name"]
    try:
        run_sparsity_evaluation(**kwargs)
        return ds, True, None
    except Exception as e:
        import traceback
        return ds, False, f"{str(e)}\n{traceback.format_exc()}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Degree-Stratified / Sparsity Analysis for KnowLight")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "movielens", "amazon_cd_2014", "amazon_toys_and_games", "amazon_beauty", "yelp"],
        help="Dataset name to evaluate, or 'all' to evaluate all 5 canonical datasets (default: all)"
    )
    parser.add_argument("--config", type=str, default="config/train_intent_augmented_lightgcn.yaml")
    parser.add_argument("--baseline_ckpt", type=str, default=None, help="Explicit baseline checkpoint path (only valid for single dataset)")
    parser.add_argument("--pipeline_ckpt", type=str, default=None, help="Explicit pipeline 3 checkpoint path (only valid for single dataset)")
    parser.add_argument("--aug_graph", type=str, default=None, help="Explicit path to augmented graph .npz (optional override)")
    parser.add_argument("--output_dir", type=str, default="data/reports")
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="Maximum parallel worker processes for --dataset all (default: 5)"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        default=False,
        help="Run datasets sequentially instead of in parallel when --dataset all is used"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Automatically generate sparsity line plots (PDF & PNG) after evaluation"
    )

    args = parser.parse_args()

    if args.dataset.lower() == "all":
        datasets_to_run = CANONICAL_DATASETS
        num_workers = min(len(datasets_to_run), args.max_workers)
        is_parallel = (not args.sequential) and (num_workers > 1)

        print("\n" + "#" * 80)
        print(" RUNNING DEGREE-STRATIFIED SPARSITY ANALYSIS ACROSS ALL 5 CANONICAL DATASETS")
        print(f" Datasets: {', '.join(datasets_to_run)}")
        print(f" Execution Mode: {'PARALLEL (max_workers=' + str(num_workers) + ')' if is_parallel else 'SEQUENTIAL'}")
        print("#" * 80 + "\n")

        if args.baseline_ckpt or args.pipeline_ckpt:
            logging.warning(
                "[Notice] Specific checkpoint paths were provided with --dataset all. "
                "Auto-discovery will be used for any dataset where the checkpoint path does not match."
            )

        tasks = []
        for ds in datasets_to_run:
            b_ckpt = args.baseline_ckpt if (args.baseline_ckpt and ds in args.baseline_ckpt) else None
            p_ckpt = args.pipeline_ckpt if (args.pipeline_ckpt and ds in args.pipeline_ckpt) else None
            a_graph = args.aug_graph if (args.aug_graph and ds in args.aug_graph) else None
            tasks.append({
                "dataset_name": ds,
                "config_path": args.config,
                "baseline_ckpt_path": b_ckpt,
                "pipeline_ckpt_path": p_ckpt,
                "aug_graph_path": a_graph,
                "output_dir": args.output_dir,
                "force_retrain_pipeline": args.force_retrain,
                "device_name": args.device
            })

        start_time = time.time()
        results = {}

        if is_parallel:
            ctx = mp.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
                future_to_ds = {
                    executor.submit(_run_single_dataset_worker, task): task["dataset_name"]
                    for task in tasks
                }
                for future in concurrent.futures.as_completed(future_to_ds):
                    ds = future_to_ds[future]
                    try:
                        _, success, err = future.result()
                        results[ds] = (success, err)
                        if success:
                            logging.info(f"==> [{ds}] Completed successfully.")
                        else:
                            logging.error(f"==> [{ds}] Failed:\n{err}")
                    except Exception as exc:
                        logging.error(f"==> [{ds}] Worker process exception: {exc}")
                        results[ds] = (False, str(exc))
        else:
            for task in tasks:
                ds = task["dataset_name"]
                try:
                    run_sparsity_evaluation(**task)
                    results[ds] = (True, None)
                    logging.info(f"==> [{ds}] Completed successfully.")
                except Exception as e:
                    logging.error(f"==> [{ds}] Failed: {e}")
                    results[ds] = (False, str(e))

        elapsed = time.time() - start_time
        print("\n" + "=" * 80)
        print(f" SPARSITY ANALYSIS SUMMARY (Total Elapsed: {elapsed:.1f}s)")
        print("=" * 80)
        for ds in datasets_to_run:
            ok, err = results.get(ds, (False, "Not executed"))
            status = "[OK] SUCCESS" if ok else f"[FAILED] ({err.splitlines()[-1] if err else 'Error'})"
            print(f"- {ds:<25}: {status}")
        print("=" * 80 + "\n")

        if args.plot:
            try:
                from knowlight.evaluation.graph_generator import generate_sparsity_graph
                out_reports_dir = PROJECT_ROOT / args.output_dir
                generate_sparsity_graph(reports_dir=out_reports_dir)
            except Exception as e:
                logging.error(f"Failed to generate sparsity graph: {e}")
    else:
        run_sparsity_evaluation(
            dataset_name=args.dataset,
            config_path=args.config,
            baseline_ckpt_path=args.baseline_ckpt,
            pipeline_ckpt_path=args.pipeline_ckpt,
            aug_graph_path=args.aug_graph,
            output_dir=args.output_dir,
            force_retrain_pipeline=args.force_retrain,
            device_name=args.device
        )
        if args.plot:
            try:
                from knowlight.evaluation.graph_generator import generate_sparsity_graph
                out_reports_dir = PROJECT_ROOT / args.output_dir
                generate_sparsity_graph(reports_dir=out_reports_dir)
            except Exception as e:
                logging.error(f"Failed to generate sparsity graph: {e}")

