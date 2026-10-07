import os
import sys
import json
import time
import argparse
import logging
import itertools
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
import numpy as np

# Ensure root workspace directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from knowlight.utils.helpers import set_seed, load_yaml_config, get_device
from knowlight.datasets.data_loader import KnowLightDataLoader
from knowlight.training.trainer import Trainer
from knowlight.paths import CHECKPOINTS_DIR, STUDENTS_DIR, DISTILLED_STUDENTS_DIR, get_next_run_dir, get_latest_or_next_run_dir

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ==============================================================================
# GRID SEARCH PARAMETER LISTS
# ==============================================================================
GRID_LR = [1e-4, 1e-3, 1e-2]
GRID_WEIGHT_DECAY = [1e-4, 1e-3, 1e-2]


def run_grid_search(
    base_config_path: str,
    dataset_name: str,
    lr_list: List[float] = None,
    weight_decay_list: List[float] = None,
    epochs: int = None,
    seed: int = 42,
    teacher_name: str = "Qwen3.5-4B"
):
    """
    Executes Grid Search over optimization hyperparameters (lr, weight_decay).
    """
    set_seed(seed)
    
    # Use code-defined default grid lists if not provided
    lr_list = lr_list or GRID_LR
    weight_decay_list = weight_decay_list or GRID_WEIGHT_DECAY
    
    # Load base configs
    train_config = load_yaml_config(base_config_path, dataset_name=dataset_name)
    llm_config = load_yaml_config("config/teacher_llm.yaml")

    if epochs:
        train_config["training"]["epochs"] = epochs

    device = get_device(train_config.get("experiment", {}).get("device", "cuda"))
    logging.info(f"Running Grid Search on device: {device} for dataset: '{dataset_name}'")

    # Load dataset once
    data_loader = KnowLightDataLoader(
        dataset_name=train_config["dataset"]["name"],
        data_dir=train_config["dataset"]["data_dir"],
        min_uc=train_config["dataset"]["min_uc"],
        min_ic=train_config["dataset"]["min_ic"]
    )
    data_loader.load_data()

    # Generate Cartesian product of all hyperparameter combinations
    param_grid = list(itertools.product(
        lr_list,
        weight_decay_list
    ))

    total_trials = len(param_grid)
    logging.info(f"Starting Grid Search with {total_trials} total parameter combinations...")

    # Determine saving path based on whether distillation is used or baseline
    model_name = train_config.get("student", {}).get("model", "lightgcn").lower()
    is_distillation_active = train_config.get("distillation", {}).get("active", True)
    if is_distillation_active:
        teacher_tag = Path(teacher_name).name.lower().replace("-", "_").replace(".", "_")
        prefix = f"{teacher_tag}_score_run" if teacher_name else "score_run"
        out_dir = get_latest_or_next_run_dir(base_dir, prefix, resume=True)
    else:
        base_dir = STUDENTS_DIR / model_name / dataset_name
        out_dir = get_latest_or_next_run_dir(base_dir, "baseline_run", resume=True)

    grid_ckpt_file = out_dir / "checkpoint_grid_state.json"
    ckpt_student_path = out_dir / "checkpoint_best_student.pt"
    ckpt_distiller_path = out_dir / "checkpoint_best_distiller.pt"

    results = []
    completed_keys = set()
    best_ndcg = -1.0
    best_trial_info = None
    best_model_state = None
    best_distiller_state = None

    if grid_ckpt_file.exists():
        try:
            with open(grid_ckpt_file, "r", encoding="utf-8") as f:
                saved_grid = json.load(f)
            results = saved_grid.get("completed_trials", [])
            completed_keys = {(float(t["hyperparameters"]["lr"]), float(t["hyperparameters"]["weight_decay"])) for t in results}
            best_ndcg = saved_grid.get("best_ndcg", -1.0)
            best_trial_info = saved_grid.get("best_trial", None)
            if ckpt_student_path.exists():
                best_model_state = torch.load(ckpt_student_path, map_location="cpu", weights_only=False)
            if ckpt_distiller_path.exists():
                best_distiller_state = torch.load(ckpt_distiller_path, map_location="cpu", weights_only=False)
            logging.info(f"[Grid Search] Resuming run '{out_dir.name}': {len(results)} trials already completed.")
        except Exception as e:
            logging.warning(f"[Grid Search] Failed loading grid checkpoint ({e}). Starting fresh.")
            results = []
            completed_keys = set()

    # Pre-load teacher scores and graph alignment once across all grid search trials if distillation is active
    preloaded_scores = None
    prompt_builder = None
    if is_distillation_active:
        logging.info(f"[Grid Search] Pre-loading Teacher Scores ({teacher_name}) and graph alignment once for all trials...")
        temp_trainer = Trainer(
            train_config=train_config,
            llm_config=llm_config,
            data_loader=data_loader,
            device=device,
            teacher_name=teacher_name
        )
        temp_trainer.preextract_teacher_knowledge()
        preloaded_scores = temp_trainer.cached_user_evaluated_items
        prompt_builder = temp_trainer.prompt_builder
        del temp_trainer

    search_start_time = time.time()

    for idx, (lr, wd) in enumerate(param_grid, start=1):
        key = (float(lr), float(wd))
        if key in completed_keys:
            print(f"\n[Grid Search] Trial {idx}/{total_trials} (LR: {lr}, WeightDecay: {wd}) already completed. Skipping!")
            continue

        # Reset random seed for every trial to ensure identical weight initialization
        # and identical negative sampling sequences across all hyperparameter configurations
        set_seed(seed)

        print("\n" + "=" * 75)
        print(f" Trial {idx}/{total_trials}")
        print(f" Hyperparameters -> LR: {lr}, WeightDecay: {wd} (Seed: {seed})")
        print("=" * 75)

        # Clone config for current trial
        cfg = json.loads(json.dumps(train_config))
        cfg["training"]["lr"] = float(lr)
        cfg["training"]["weight_decay"] = float(wd)

        trial_start = time.time()
        trainer = Trainer(
            train_config=cfg,
            llm_config=llm_config,
            data_loader=data_loader,
            device=device,
            teacher_name=teacher_name,
            preloaded_scores=preloaded_scores,
            prompt_builder=prompt_builder
        )

        test_metrics = trainer.train(out_dir=out_dir)
        trial_elapsed = time.time() - trial_start

        # Target optimization metric: NDCG@20
        current_ndcg = test_metrics.get("NDCG@20", 0.0)

        trial_record = {
            "trial_id": idx,
            "hyperparameters": {
                "lr": lr,
                "weight_decay": wd
            },
            "metrics": {k: float(v) for k, v in test_metrics.items()},
            "elapsed_seconds": round(trial_elapsed, 2)
        }

        if trainer.is_distillation_active and hasattr(trainer, 'distiller'):
            trial_record["distillation_params"] = {
                "alpha": getattr(trainer.distiller, "alpha", 1.0),
                "beta": getattr(trainer.distiller, "beta", 0.1)
            }

        results.append(trial_record)
        completed_keys.add(key)

        if current_ndcg > best_ndcg:
            best_ndcg = current_ndcg
            best_trial_info = trial_record
            best_model_state = {k: v.cpu().clone() for k, v in trainer.student.state_dict().items()}
            torch.save(best_model_state, ckpt_student_path)
            
            # Save distiller state as well
            if trainer.is_distillation_active and hasattr(trainer, 'distiller'):
                best_distiller_state = {k: v.cpu().clone() for k, v in trainer.distiller.state_dict().items()}
                torch.save(best_distiller_state, ckpt_distiller_path)
            else:
                best_distiller_state = None
                
            print(f" >>> New Best Trial Found! NDCG@20: {best_ndcg:.4f} <<<")

        grid_save_data = {
            "completed_trials": results,
            "best_ndcg": best_ndcg,
            "best_trial": best_trial_info
        }
        with open(grid_ckpt_file, "w", encoding="utf-8") as f:
            json.dump(grid_save_data, f, indent=4)

    # Sort results by NDCG@20 descending
    results.sort(key=lambda r: r["metrics"].get("NDCG@20", 0.0), reverse=True)

    total_search_time = time.time() - search_start_time

    report_file = out_dir / "grid_search_results.json"

    summary_output = {
        "dataset": dataset_name,
        "teacher": teacher_name if is_distillation_active else "None",
        "total_trials": total_trials,
        "total_elapsed_seconds": round(total_search_time, 2),
        "best_trial": best_trial_info,
        "all_trials": results
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=4)

    # Save best student model weights
    if best_model_state:
        model_save_path = out_dir / "best_student_lightgcn.pt"
        torch.save(best_model_state, model_save_path)
        print(f"[KnowLight Grid Search] Saved best student model weights to: {model_save_path}")
        
        # Determine best_distiller_state from the variable scope
        if 'best_distiller_state' in locals() and best_distiller_state is not None:
            distiller_save_path = out_dir / "best_distiller.pt"
            torch.save(best_distiller_state, distiller_save_path)
            print(f"[KnowLight Grid Search] Saved best distiller weights to: {distiller_save_path}")

    print("\n" + "=" * 70)
    print(f"  KnowLight Grid Search Final Summary ({dataset_name.upper()})")
    print("=" * 70)
    print(f"  Total Trials Tested : {total_trials}")
    print(f"  Total Time Elapsed  : {total_search_time / 60:.2f} minutes")
    if best_trial_info:
        bm = best_trial_info.get("metrics", {})
        bp = best_trial_info.get("hyperparameters", {})
        print(f"  Best Trial ID       : Trial #{best_trial_info.get('trial_id')}")
        print(f"  Best Hyperparameters: LR={bp.get('lr')}, WeightDecay={bp.get('weight_decay')}")
        print(f"   NDCG@20          : {bm.get('NDCG@20', 0.0):.4f}")
        print(f"   Recall@20        : {bm.get('Recall@20', 0.0):.4f}")
        print(f"   HR@20            : {bm.get('HR@20', 0.0):.4f}")
    print(f" Results saved to    : {report_file}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Grid Search Hyperparameter Tuning for KnowLight LightGCN Student")
    parser.add_argument("--config", type=str, default="config/train_student.yaml", help="Path to base config file")
    parser.add_argument("--dataset", type=str, default="movielens", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs per trial (default: 20)")
    
    # Hyperparameter Grid Lists (defaults mapped to top-level constants)
    parser.add_argument("--lr", nargs="+", type=float, default=GRID_LR, help="Learning rates to test")
    parser.add_argument("--weight_decay", nargs="+", type=float, default=GRID_WEIGHT_DECAY, help="Weight decays to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run_grid_search(
        base_config_path=args.config,
        dataset_name=args.dataset,
        lr_list=args.lr,
        weight_decay_list=args.weight_decay,
        epochs=args.epochs,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
