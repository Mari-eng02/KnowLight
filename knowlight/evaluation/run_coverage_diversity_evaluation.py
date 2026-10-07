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
from knowlight.student.lightgcn import LightGCN
from knowlight.evaluation.coverage_diversity_evaluator import CoverageDiversityEvaluator
from knowlight.evaluation.graph_generator import generate_coverage_diversity_graph

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
    """Finds the latest existing LightGCN baseline checkpoint for a dataset."""
    base_student_dir = STUDENTS_DIR / "lightgcn" / dataset_name
    if not base_student_dir.exists():
        return None

    subdirs = [d for d in base_student_dir.iterdir() if d.is_dir()]
    subdirs.sort(key=_extract_run_number, reverse=True)

    for run_dir in subdirs:
        for pt_file in ["best_student_lightgcn.pt", "checkpoint_best_student.pt", "best_student_model.pt"]:
            candidate = run_dir / pt_file
            if candidate.exists():
                logging.info(f"[Auto-Discovery] Baseline checkpoint found: '{run_dir.name}/{candidate.name}'")
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


def run_coverage_diversity_for_dataset(
    dataset_name: str,
    config_path: str = "config/train_intent_augmented_lightgcn.yaml",
    baseline_ckpt_path: Optional[str] = None,
    pipeline_ckpt_path: Optional[str] = None,
    aug_graph_path: Optional[str] = None,
    top_k_list: List[int] = [5, 10, 20],
    output_dir: str = "data/reports",
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Executes Coverage and Diversity evaluation comparing Baseline LightGCN vs Pipeline 3.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"\n{'='*75}\n RUNNING COVERAGE & DIVERSITY EVALUATION: {dataset_name.upper()}\n{'='*75}")

    # 1. Load config and seed
    config = load_yaml_config(config_path, dataset_name=dataset_name)
    set_seed(config.get("experiment", {}).get("seed", 42))

    # 2. Load dataset
    data_dir = config.get("dataset", {}).get("data_dir", "./data/raw")
    data_loader = KnowLightDataLoader(dataset_name=dataset_name, data_dir=data_dir)
    data_loader.load_data()

    num_users = data_loader.num_users
    num_items = data_loader.num_items
    student_cfg = config.get("student", {})
    embedding_dim = student_cfg.get("embedding_dim", 64)
    n_layers = student_cfg.get("n_layers", 3)
    keep_prob = student_cfg.get("keep_prob", 0.6)

    logging.info(f"Loaded dataset '{dataset_name}': {num_users} users, {num_items} items.")

    # 3. Resolve and Load Baseline Checkpoint
    if baseline_ckpt_path:
        base_ckpt = Path(baseline_ckpt_path)
    else:
        base_ckpt = find_baseline_checkpoint(dataset_name)

    if not base_ckpt or not base_ckpt.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found for '{dataset_name}'. Check STUDENTS_DIR / lightgcn / {dataset_name}")

    logging.info(f"Loading Baseline LightGCN checkpoint from: {base_ckpt}")
    baseline_model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        keep_prob=keep_prob
    ).to(device)

    base_state = torch.load(base_ckpt, map_location=device, weights_only=False)
    baseline_model.load_state_dict(base_state)
    baseline_model.set_graph(data_loader.train_matrix, device=device)

    # 4. Resolve and Load Pipeline 3 Checkpoint & Augmented Graph
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
        raise FileNotFoundError(f"Cached augmented graph not found for top_k={best_info['top_k_pseudo']}, weight={best_info['pseudo_weight']}, threshold={best_info['sim_threshold']}")

    logging.info(f"Loading Augmented Graph from: {aug_graph_file.name}")
    augmented_r = sp.load_npz(aug_graph_file)

    if not pipe_ckpt:
        grid_run_dir = Path(best_info["grid_results_path"]).parent
        candidate_ckpt = grid_run_dir / "best_student_model.pt"
        if candidate_ckpt.exists():
            pipe_ckpt = candidate_ckpt
        else:
            pipe_ckpt = STUDENTS_DIR / "intent_augmented" / dataset_name / "best_model" / "best_student_model.pt"

    if not pipe_ckpt.exists():
        raise FileNotFoundError(
            f"Pipeline 3 checkpoint not found at '{pipe_ckpt}'. "
            f"Please ensure the Intent-Augmented student training has been executed."
        )

    logging.info(f"Loading Pipeline 3 LightGCN checkpoint from: {pipe_ckpt}")
    pipeline_model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        keep_prob=keep_prob
    ).to(device)

    pipe_state = torch.load(pipe_ckpt, map_location=device, weights_only=False)
    pipeline_model.load_state_dict(pipe_state)
    pipeline_model.set_graph(augmented_r, device=device)

    # 5. Evaluate Beyond-Accuracy Metrics
    evaluator = CoverageDiversityEvaluator(top_k_list=top_k_list)

    logging.info("Evaluating Beyond-Accuracy metrics for Baseline LightGCN...")
    baseline_metrics = evaluator.evaluate_model(baseline_model, data_loader, device=device)

    logging.info("Evaluating Beyond-Accuracy metrics for Pipeline 3 (Intent-Augmented)...")
    pipeline_metrics = evaluator.evaluate_model(pipeline_model, data_loader, device=device)

    deltas = evaluator.compute_deltas(baseline_metrics, pipeline_metrics)

    # 6. Display Comparison Table in Terminal
    print("\n" + "=" * 95)
    print(f" BEYOND-ACCURACY EVALUATION: BASELINE vs PIPELINE 3 (COVERAGE & DIVERSITY)")
    print(f" Dataset: {dataset_name.upper()} | Best Params: k={best_info['top_k_pseudo']}, weight={best_info['pseudo_weight']}")
    print("=" * 95)
    print(f"{'Metric':<30} | {'Baseline':<18} | {'Pipeline 3':<18} | {'Delta (%)':<15}")
    print("-" * 95)

    metric_categories = [
        ("Coverage", "Catalog Coverage (%) [Higher is better]"),
        ("Personalization", "Personalization / Inter-User Diversity [Higher is better]"),
        ("IntraListDiversity", "Intra-List Diversity (ILD) [Higher is better]"),
        ("Novelty", "Novelty / Self-Information (bits) [Higher is better]")
    ]

    for prefix, cat_label in metric_categories:
        print(f"--- {cat_label} ---")
        for k in top_k_list:
            m_key = f"{prefix}@{k}"
            b_val = baseline_metrics.get(m_key, 0.0)
            p_val = pipeline_metrics.get(m_key, 0.0)
            d_val = deltas.get(m_key, 0.0)

            # Formatting
            if "Coverage" in prefix:
                b_str = f"{b_val:.2f}%"
                p_str = f"{p_val:.2f}%"
            else:
                b_str = f"{b_val:.4f}"
                p_str = f"{p_val:.4f}"

            d_str = f"{d_val:>+7.2f}%"
            print(f"  {m_key:<28} | {b_str:<18} | {p_str:<18} | {d_str:<15}")

    print("=" * 95 + "\n")

    # 7. Generate and Save Reports (JSON and Markdown) in data/reports/coverage_analysis/
    out_dir_path = PROJECT_ROOT / output_dir
    if "coverage_analysis" not in str(out_dir_path):
        out_dir_path = out_dir_path / "coverage_analysis"
    out_dir_path.mkdir(parents=True, exist_ok=True)

    json_path, md_path = get_next_report_files(out_dir_path, "coverage_diversity", dataset_name)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    report_data = {
        "dataset": dataset_name,
        "timestamp": timestamp,
        "best_pipeline_params": best_info,
        "baseline_checkpoint": str(base_ckpt),
        "pipeline_checkpoint": str(pipe_ckpt),
        "top_k_list": top_k_list,
        "baseline_metrics": baseline_metrics,
        "pipeline_metrics": pipeline_metrics,
        "delta_percent": deltas
    }

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Save Markdown
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Beyond-Accuracy Report: Coverage & Diversity ({dataset_name.upper()})\n\n")
        f.write(f"- **Date**: {timestamp}\n")
        f.write(f"- **Dataset**: `{dataset_name}`\n")
        f.write(f"- **Pipeline 3 Best Params**: $k={best_info['top_k_pseudo']}$, $\\text{{weight}}={best_info['pseudo_weight']}$\n")
        f.write(f"- **Baseline Checkpoint**: `{base_ckpt.name}`\n")
        f.write(f"- **Pipeline 3 Checkpoint**: `{pipe_ckpt.name}`\n\n")
        f.write("## 1. Comparative Results (Baseline vs Pipeline 3)\n\n")
        f.write("| Metric | Baseline LightGCN | Pipeline 3 (Intent-Augmented) | Delta (%) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")

        for prefix, _ in metric_categories:
            for k in top_k_list:
                m_key = f"{prefix}@{k}"
                b_val = baseline_metrics.get(m_key, 0.0)
                p_val = pipeline_metrics.get(m_key, 0.0)
                d_val = deltas.get(m_key, 0.0)
                if "Coverage" in prefix:
                    b_str = f"{b_val:.2f}%"
                    p_str = f"{p_val:.2f}%"
                else:
                    b_str = f"{b_val:.4f}"
                    p_str = f"{p_val:.4f}"
                d_str = f"**{d_val:+.2f}%**"
                f.write(f"| **{m_key}** | {b_str} | **{p_str}** | {d_str} |\n")

    logging.info(f"Reports successfully generated:")
    logging.info(f"- JSON: {json_path}")
    logging.info(f"- Markdown: {md_path}")

    return report_data


def main():
    parser = argparse.ArgumentParser(
        description="KnowLight: Beyond-Accuracy (Coverage, Diversity, Personalization) Evaluation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="amazon_cd_2014",
        help="Dataset name ('movielens', 'amazon_cd_2014', 'amazon_toys_and_games', 'amazon_beauty', 'yelp', or 'all')"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_intent_augmented_lightgcn.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--baseline_ckpt",
        type=str,
        default=None,
        help="Explicit path to Baseline LightGCN checkpoint (.pt)"
    )
    parser.add_argument(
        "--pipeline_ckpt",
        type=str,
        default=None,
        help="Explicit path to Pipeline 3 Intent-Augmented checkpoint (.pt)"
    )
    parser.add_argument(
        "--aug_graph",
        type=str,
        default=None,
        help="Explicit path to augmented graph .npz (optional override)"
    )
    parser.add_argument(
        "--top_k",
        nargs="+",
        type=int,
        default=[5, 10, 20],
        help="Top-K cutoff thresholds (e.g. 5 10 20)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/reports",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device ('cuda' or 'cpu')"
    )
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
        help="Automatically generate 2x2 comparison plots (PDF & PNG) after evaluation"
    )

    args = parser.parse_args()

    if args.dataset.lower() == "all":
        datasets_to_run = CANONICAL_DATASETS
        num_workers = min(len(datasets_to_run), args.max_workers)
        is_parallel = (not args.sequential) and (num_workers > 1)

        print("\n" + "#" * 80)
        print(" RUNNING BEYOND-ACCURACY EVALUATION ACROSS ALL 5 CANONICAL DATASETS")
        print(f" Datasets: {', '.join(datasets_to_run)}")
        print(f" Execution Mode: {'PARALLEL (max_workers=' + str(num_workers) + ')' if is_parallel else 'SEQUENTIAL'}")
        print("#" * 80 + "\n")

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
                "top_k_list": args.top_k,
                "output_dir": args.output_dir,
                "device_name": args.device
            })

        start_time = time.time()
        results = {}

        if is_parallel:
            ctx = mp.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
                future_to_ds = {
                    executor.submit(_run_single_cov_div_worker, task): task["dataset_name"]
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
                    dev = torch.device(task["device_name"] if task["device_name"] else ("cuda" if torch.cuda.is_available() else "cpu"))
                    run_coverage_diversity_for_dataset(
                        dataset_name=ds,
                        config_path=task["config_path"],
                        baseline_ckpt_path=task["baseline_ckpt_path"],
                        pipeline_ckpt_path=task["pipeline_ckpt_path"],
                        aug_graph_path=task.get("aug_graph_path"),
                        top_k_list=task["top_k_list"],
                        output_dir=task["output_dir"],
                        device=dev
                    )
                    results[ds] = (True, None)
                    logging.info(f"==> [{ds}] Completed successfully.")
                except Exception as e:
                    logging.error(f"==> [{ds}] Failed: {e}")
                    results[ds] = (False, str(e))

        elapsed = time.time() - start_time
        print("\n" + "=" * 80)
        print(f" COVERAGE & DIVERSITY EVALUATION SUMMARY (Total Elapsed: {elapsed:.1f}s)")
        print("=" * 80)
        for ds in datasets_to_run:
            ok, err = results.get(ds, (False, "Not executed"))
            status = "[OK] SUCCESS" if ok else f"[FAILED] ({err.splitlines()[-1] if err else 'Error'})"
            print(f"- {ds:<25}: {status}")
        print("=" * 80 + "\n")

    else:
        dev = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        run_coverage_diversity_for_dataset(
            dataset_name=args.dataset,
            config_path=args.config,
            baseline_ckpt_path=args.baseline_ckpt,
            pipeline_ckpt_path=args.pipeline_ckpt,
            aug_graph_path=args.aug_graph,
            top_k_list=args.top_k,
            output_dir=args.output_dir,
            device=dev
        )

    if args.plot:
        out_reports_dir = PROJECT_ROOT / args.output_dir
        for k_val in args.top_k:
            generate_coverage_diversity_graph(k=k_val, reports_dir=out_reports_dir)


def _run_single_cov_div_worker(kwargs: Dict[str, Any]) -> Tuple[str, bool, Optional[str]]:
    """Worker wrapper to execute coverage & diversity evaluation in a child process."""
    ds = kwargs["dataset_name"]
    try:
        dev = torch.device(kwargs["device_name"] if kwargs["device_name"] else ("cuda" if torch.cuda.is_available() else "cpu"))
        run_coverage_diversity_for_dataset(
            dataset_name=ds,
            config_path=kwargs["config_path"],
            baseline_ckpt_path=kwargs["baseline_ckpt_path"],
            pipeline_ckpt_path=kwargs["pipeline_ckpt_path"],
            top_k_list=kwargs["top_k_list"],
            output_dir=kwargs["output_dir"],
            device=dev
        )
        return ds, True, None
    except Exception as e:
        import traceback
        return ds, False, f"{str(e)}\n{traceback.format_exc()}"


if __name__ == "__main__":
    main()
