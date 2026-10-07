import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch

# Ensure root workspace directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from knowlight.utils.helpers import set_seed, load_yaml_config, get_device
from knowlight.datasets.data_loader import KnowLightDataLoader
from knowlight.training.trainer import Trainer
from knowlight.paths import DISTILLED_STUDENTS_DIR, get_latest_or_next_run_dir

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

DEFAULT_BETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
DEFAULT_LAMBDA_RRD = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
DEFAULT_LAMBDA_UNKD = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def run_score_grid_search(
    base_config_path: str = "config/train_student_score_distill.yaml",
    dataset_name: str = "movielens",
    distiller_type: str = "score",
    beta_list: List[float] = None,
    lambda_rrd_list: List[float] = None,
    lambda_unkd_list: List[float] = None,
    epochs: int = None,
    seed: int = 42,
    teacher_name: str = "Qwen3.5-4B"
):
    """
    Executes a Distillation Grid Search for Pipeline 1:
    - If score: grid over beta in [0.0, 1.0]
    - If rrd: grid over lambda_rrd with interesting_size=40, uninteresting_size=50, temp=1.0, topk=500
    - If unkd: grid over lambda_unkd with sample_num=30, group_count=4, pop_lambda=1.0, rank_top_k=1000, rank_temp=20.0
    """
    set_seed(seed)
    distiller_type = str(distiller_type).lower()

    train_config = load_yaml_config(base_config_path, dataset_name=dataset_name)
    llm_config = load_yaml_config("config/teacher_llm.yaml")

    if epochs:
        train_config["training"]["epochs"] = epochs

    lr_val = train_config.get("training", {}).get("lr")
    wd_val = train_config.get("training", {}).get("weight_decay")
    device = get_device(train_config.get("experiment", {}).get("device", "cuda"))
    logging.info(f"[Pipeline 1 Grid Search] Distiller: {distiller_type.upper()} | Device: {device} | Dataset: '{dataset_name}' | LR: {lr_val} | Weight Decay: {wd_val} | Teacher: '{teacher_name}'")
    print(f"[{distiller_type.upper()} Grid Search] Dataset: '{dataset_name}' | LR: {lr_val} | Weight Decay: {wd_val} | Teacher: '{teacher_name}'")

    # Load dataset once
    data_loader = KnowLightDataLoader(
        dataset_name=train_config["dataset"]["name"],
        data_dir=train_config["dataset"]["data_dir"],
        min_uc=train_config["dataset"]["min_uc"],
        min_ic=train_config["dataset"]["min_ic"]
    )
    data_loader.load_data()

    model_name = train_config.get("student", {}).get("model", "lightgcn").lower()
    base_dir = DISTILLED_STUDENTS_DIR / model_name / dataset_name
    teacher_tag = Path(teacher_name).name.lower().replace("-", "_").replace(".", "_")
    prefix = f"{teacher_tag}_{distiller_type}_grid"
    out_dir = get_latest_or_next_run_dir(base_dir, prefix, resume=True)
    grid_ckpt_file = out_dir / "checkpoint_grid_state.json"
    ckpt_student_path = out_dir / "checkpoint_best_student.pt"
    ckpt_distiller_path = out_dir / "checkpoint_best_distiller.pt"

    results = []
    completed_keys = set()
    start_time = time.time()
    best_ndcg = -1.0
    best_student_state = None
    best_distiller_state = None
    best_trial_params = None

    if grid_ckpt_file.exists():
        try:
            with open(grid_ckpt_file, "r", encoding="utf-8") as f:
                saved_grid = json.load(f)
            results = saved_grid.get("completed_trials", [])
            completed_keys = {str(t["param_key"]) for t in results}
            best_ndcg = saved_grid.get("best_ndcg", -1.0)
            best_trial_params = saved_grid.get("best_trial_params", None)
            if ckpt_student_path.exists():
                best_student_state = torch.load(ckpt_student_path, map_location="cpu", weights_only=False)
            if ckpt_distiller_path.exists():
                best_distiller_state = torch.load(ckpt_distiller_path, map_location="cpu", weights_only=False)
            logging.info(f"[{distiller_type.upper()} Grid Search] Resuming run '{out_dir.name}': {len(results)} trials already completed.")
        except Exception as e:
            logging.warning(f"[{distiller_type.upper()} Grid Search] Failed loading grid checkpoint ({e}). Starting fresh.")
            results = []
            completed_keys = set()

    # Build trial param grid
    dist_cfg = train_config.get("distillation", {})
    if distiller_type == "rrd":
        lambda_rrd_list = lambda_rrd_list or DEFAULT_LAMBDA_RRD
        rrd_int_size = int(dist_cfg.get("interesting_size", 40))
        rrd_unint_size = int(dist_cfg.get("uninteresting_size", 50))
        rrd_temp = float(dist_cfg.get("temperature_rrd", 1.0))
        rrd_topk = int(dist_cfg.get("teacher_topk", 500))
        param_grid = [{
            "param_key": str(l),
            "lambda_rrd": float(l),
            "interesting_size": rrd_int_size,
            "uninteresting_size": rrd_unint_size,
            "temperature_rrd": rrd_temp,
            "teacher_topk": rrd_topk
        } for l in lambda_rrd_list]
    elif distiller_type == "unkd":
        lambda_unkd_list = lambda_unkd_list or DEFAULT_LAMBDA_UNKD
        u_sample_num = int(dist_cfg.get("sample_num", 30))
        u_group_count = int(dist_cfg.get("group_count", 4))
        u_pop_lambda = float(dist_cfg.get("popularity_lambda", 1.0))
        u_rank_topk = int(dist_cfg.get("rank_top_k", 1000))
        u_rank_temp = float(dist_cfg.get("rank_temperature", 20.0))
        param_grid = [{
            "param_key": str(l),
            "lambda_unkd": float(l),
            "sample_num": u_sample_num,
            "group_count": u_group_count,
            "popularity_lambda": u_pop_lambda,
            "rank_top_k": u_rank_topk,
            "rank_temperature": u_rank_temp
        } for l in lambda_unkd_list]
    else:
        beta_list = beta_list or DEFAULT_BETAS
        param_grid = [{"param_key": str(b), "beta": float(b)} for b in beta_list]

    total_trials = len(param_grid)

    # Pre-load teacher scores once across all grid search trials
    preloaded_scores = None
    prompt_builder = None
    if distiller_type in ["score", "knowlight", "rrd"]:
        logging.info(f"[{distiller_type.upper()} Grid Search] Pre-loading Teacher Scores ({teacher_name}) and graph alignment once for all trials...")
        temp_cfg = json.loads(json.dumps(train_config))
        temp_cfg.setdefault("distillation", {})["type"] = distiller_type
        temp_cfg["distillation"]["active"] = True
        temp_trainer = Trainer(
            train_config=temp_cfg,
            llm_config=llm_config,
            data_loader=data_loader,
            device=device,
            teacher_name=teacher_name
        )
        temp_trainer.preextract_teacher_knowledge()
        preloaded_scores = temp_trainer.cached_user_evaluated_items
        prompt_builder = temp_trainer.prompt_builder
        del temp_trainer

    for idx, params in enumerate(param_grid, start=1):
        key = params["param_key"]
        if key in completed_keys:
            print(f"\n[{distiller_type.upper()} Grid Search] Trial {idx}/{total_trials} ({key}) already completed. Skipping!")
            continue

        # Reset random seed for every trial to ensure identical weight initialization
        # and identical negative sampling sequences across all distillation configurations
        set_seed(seed)

        print("\n" + "=" * 75)
        print(f" {distiller_type.upper()} Grid Search Trial {idx}/{total_trials} -> Params: {params} (Seed: {seed})")
        print("=" * 75)

        cfg = json.loads(json.dumps(train_config))
        cfg.setdefault("distillation", {})
        cfg["distillation"]["active"] = True
        cfg["distillation"]["type"] = distiller_type

        if distiller_type == "rrd":
            cfg["distillation"]["lambda_rrd"] = params["lambda_rrd"]
            cfg["distillation"]["interesting_size"] = params["interesting_size"]
            cfg["distillation"]["uninteresting_size"] = params["uninteresting_size"]
            cfg["distillation"]["temperature_rrd"] = params["temperature_rrd"]
            cfg["distillation"]["teacher_topk"] = params["teacher_topk"]
        elif distiller_type == "unkd":
            cfg["distillation"]["lambda_unkd"] = params["lambda_unkd"]
            cfg["distillation"]["sample_num"] = params["sample_num"]
            cfg["distillation"]["group_count"] = params["group_count"]
            cfg["distillation"]["popularity_lambda"] = params["popularity_lambda"]
            cfg["distillation"]["rank_top_k"] = params["rank_top_k"]
            cfg["distillation"]["rank_temperature"] = params["rank_temperature"]
        else:
            b_val = params["beta"]
            if b_val == 0.0:
                cfg["distillation"]["active"] = False
            else:
                cfg["distillation"]["active"] = True
                cfg["distillation"]["beta"] = b_val

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
        current_ndcg = test_metrics.get("NDCG@20", 0.0)

        trial_record = {
            "trial_id": idx,
            "param_key": key,
            "params": params,
            "lr": cfg["training"].get("lr"),
            "weight_decay": cfg["training"].get("weight_decay"),
            "metrics": {k: float(v) for k, v in test_metrics.items()},
            "elapsed_seconds": round(trial_elapsed, 2)
        }
        results.append(trial_record)
        completed_keys.add(key)

        if current_ndcg > best_ndcg:
            best_ndcg = current_ndcg
            best_trial_params = params
            best_student_state = {k: v.cpu().clone() for k, v in trainer.student.state_dict().items()}
            torch.save(best_student_state, ckpt_student_path)
            if cfg["distillation"]["active"] and hasattr(trainer, 'distiller') and trainer.distiller is not None:
                best_distiller_state = {k: v.cpu().clone() for k, v in trainer.distiller.state_dict().items()}
                torch.save(best_distiller_state, ckpt_distiller_path)
            else:
                best_distiller_state = None
            print(f" >>> New Best Trial Found! Params={params} (NDCG@20: {best_ndcg:.4f}) <<<")

        grid_save_data = {
            "completed_trials": results,
            "best_ndcg": best_ndcg,
            "best_trial_params": best_trial_params
        }
        with open(grid_ckpt_file, "w", encoding="utf-8") as f:
            json.dump(grid_save_data, f, indent=4)

    total_elapsed = time.time() - start_time
    report_file = out_dir / f"grid_{distiller_type}_results.json"

    summary_output = {
        "pipeline": f"score_distillation_{distiller_type}",
        "distiller": distiller_type,
        "dataset": dataset_name,
        "teacher": teacher_name,
        "lr": lr_val,
        "weight_decay": wd_val,
        "total_trials": total_trials,
        "total_elapsed_seconds": round(total_elapsed, 2),
        "best_params": best_trial_params,
        "best_ndcg": best_ndcg,
        "trials": results
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=4)

    if best_student_state:
        model_save_path = out_dir / f"best_student_lightgcn_{distiller_type}.pt"
        torch.save(best_student_state, model_save_path)
        print(f"[KnowLight Grid Search] Saved best student weights (Params={best_trial_params}) to: {model_save_path}")
        
        if best_distiller_state is not None:
            distiller_save_path = out_dir / f"best_{distiller_type}_distiller.pt"
            torch.save(best_distiller_state, distiller_save_path)
            print(f"[KnowLight Grid Search] Saved best distiller weights to: {distiller_save_path}")

    # Cleanup intermediate grid checkpoint files
    for temp_p in [grid_ckpt_file, ckpt_student_path, ckpt_distiller_path]:
        if temp_p.exists():
            try:
                temp_p.unlink()
            except Exception:
                pass

    print("\n" + "=" * 75)
    print(f"  KnowLight {distiller_type.upper()} Distillation Grid Search Final Summary ({dataset_name.upper()})")
    print("=" * 75)
    print(f"{'Params':<30} | {'NDCG@20':<12} | {'Recall@20':<12} | {'HR@20':<12}")
    print("-" * 75)
    for r in results:
        m = r["metrics"]
        print(f"{r['param_key']:<30} | {m.get('NDCG@20', 0.0):<12.4f} | {m.get('Recall@20', 0.0):<12.4f} | {m.get('HR@20', 0.0):<12.4f}")
    print("=" * 75)
    print(f" Best Params overall : {best_trial_params} (NDCG@20: {best_ndcg:.4f})")
    print(f" Results saved to    : {report_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Pipeline 1 Score/Ranking Distillation Grid Search (Score, RRD, UnKD)")
    parser.add_argument("--config", type=str, default="config/train_student_score_distill.yaml", help="Path to base config file")
    parser.add_argument("--dataset", type=str, default="movielens", help="Dataset name")
    parser.add_argument("--teacher", type=str, default="Qwen3.5-4B", help="Teacher model name (default: Qwen3.5-4B)")
    parser.add_argument("--distiller", type=str, default="score", choices=["score", "knowlight", "rrd", "unkd"], help="Distillation method to use")
    parser.add_argument("--beta", nargs="+", type=float, default=None, help="Beta values to test for score distillation")
    parser.add_argument("--lambda_rrd", nargs="+", type=float, default=None, help="Lambda RRD values to test in grid search")
    parser.add_argument("--lambda_unkd", nargs="+", type=float, default=None, help="Lambda UnKD values to test in grid search")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs per trial")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run_score_grid_search(
        base_config_path=args.config,
        dataset_name=args.dataset,
        distiller_type=args.distiller,
        beta_list=args.beta,
        lambda_rrd_list=args.lambda_rrd,
        lambda_unkd_list=args.lambda_unkd,
        epochs=args.epochs,
        seed=args.seed,
        teacher_name=args.teacher
    )


if __name__ == "__main__":
    main()
