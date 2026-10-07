import argparse
import os
import sys
import torch

# Ensure root workspace directory is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from knowlight.utils.helpers import set_seed, load_yaml_config, get_device
from knowlight.datasets.data_loader import KnowLightDataLoader
from knowlight.training.trainer import Trainer
from knowlight.training.grid_search import run_grid_search
from knowlight.training.run_score_grid import run_score_grid_search, DEFAULT_BETAS


def main():
    parser = argparse.ArgumentParser(description="KnowLight Student LightGCN Training and Grid Search")
    parser.add_argument("--config", type=str, default="config/train_student.yaml", help="Path to main train config file")
    parser.add_argument("--llm_config", type=str, default="config/teacher_llm.yaml", help="Path to LLM teacher config file")
    parser.add_argument("--teacher", type=str, default="Qwen3.5-4B", help="Name of the LLM teacher (used for finding cache subdirectory). Ignored if distillation is off.")
    parser.add_argument("--dataset", type=str, default="movielens", help="Dataset name ('amazon_beauty', 'amazon_toys_and_games', 'amazon_cd_2014', 'movielens', 'yelp')")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs per trial")
    parser.add_argument("--grid", action="store_true", default=False, help="Enable Distillation Grid Search over lambda/beta values")
    parser.add_argument("--grid_search", action="store_true", default=False, help="Enable Grid Search over LR and Weight Decay")
    parser.add_argument("--single_run", action="store_true", default=False, help="Run a single training trial using YAML config (default mode if no flags given)")
    parser.add_argument("--beta", nargs="+", type=float, default=DEFAULT_BETAS, help="Beta values to test in score grid search")
    parser.add_argument("--no_distillation", action="store_true", default=False, help="Disable distillation losses for pure LightGCN baseline")
    parser.add_argument("--distiller", type=str, default="score", choices=["score", "knowlight", "rrd", "unkd"], help="Distillation method: 'score' (default), 'rrd', 'unkd'")
    parser.add_argument("--lambda_rrd", nargs="+", type=float, default=None, help="Distillation weight(s) for RRD (grid default: [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0])")
    parser.add_argument("--interesting_size", type=int, default=None, help="Interesting items per user for RRD (default: 40)")
    parser.add_argument("--uninteresting_size", type=int, default=None, help="Uninteresting items per user for RRD (default: 50)")
    parser.add_argument("--temperature_rrd", type=float, default=None, help="Sampling temperature for RRD (default: 1.0)")
    parser.add_argument("--lambda_unkd", nargs="+", type=float, default=None, help="Distillation weight(s) for UnKD (grid default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])")
    parser.add_argument("--sample_num", type=int, default=None, help="Sample number for UnKD (default: 30)")
    parser.add_argument("--group_count", type=int, default=None, help="Popularity group count for UnKD (default: 4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Load configuration YAMLs
    train_config = load_yaml_config(args.config, dataset_name=args.dataset)
    llm_config = load_yaml_config(args.llm_config)

    # Overrides
    if args.epochs:
        train_config["training"]["epochs"] = args.epochs
    train_config["experiment"]["seed"] = args.seed

    lr_val = train_config.get("training", {}).get("lr")
    wd_val = train_config.get("training", {}).get("weight_decay")
    print(f"[KnowLight] Target Dataset: '{train_config['dataset']['name']}' | LR: {lr_val} | Weight Decay: {wd_val}")

    dist_cfg = train_config.setdefault("distillation", {})
    if args.distiller:
        dist_cfg["type"] = args.distiller
    if args.lambda_rrd is not None:
        dist_cfg["lambda_rrd"] = float(args.lambda_rrd[0] if isinstance(args.lambda_rrd, list) else args.lambda_rrd)
    if args.interesting_size is not None:
        dist_cfg["interesting_size"] = args.interesting_size
    if args.uninteresting_size is not None:
        dist_cfg["uninteresting_size"] = args.uninteresting_size
    if args.temperature_rrd is not None:
        dist_cfg["temperature_rrd"] = args.temperature_rrd
    if args.lambda_unkd is not None:
        dist_cfg["lambda_unkd"] = float(args.lambda_unkd[0] if isinstance(args.lambda_unkd, list) else args.lambda_unkd)
    if args.sample_num is not None:
        dist_cfg["sample_num"] = args.sample_num
    if args.group_count is not None:
        dist_cfg["group_count"] = args.group_count

    set_seed(args.seed)

    if args.grid:
        grid_config_path = "config/train_student_score_distill.yaml" if args.config == "config/train_student.yaml" else args.config
        distiller_name = dist_cfg.get("type", "score").lower()
        print(f"[KnowLight] Launching {distiller_name.upper()} Distillation Grid Search using config: {grid_config_path}...")
        run_score_grid_search(
            base_config_path=grid_config_path,
            dataset_name=train_config["dataset"]["name"],
            distiller_type=distiller_name,
            beta_list=args.beta,
            lambda_rrd_list=args.lambda_rrd if isinstance(args.lambda_rrd, list) else ([args.lambda_rrd] if args.lambda_rrd is not None else None),
            lambda_unkd_list=args.lambda_unkd if isinstance(args.lambda_unkd, list) else ([args.lambda_unkd] if args.lambda_unkd is not None else None),
            epochs=args.epochs,
            seed=args.seed,
            teacher_name=args.teacher
        )
    elif args.grid_search:
        if args.no_distillation:
            print("[KnowLight] Launching Grid Search for Baseline LightGCN (WITHOUT Distillation, LR & WD tuning)...")
            train_config.setdefault("distillation", {})["active"] = False
        else:
            print("[KnowLight] Launching Grid Search for Student LightGCN Training (Full Distillation)...")
            train_config.setdefault("distillation", {})["active"] = True

        run_grid_search(
            base_config_path=args.config,
            dataset_name=train_config["dataset"]["name"],
            epochs=train_config["training"].get("epochs", 100),
            seed=args.seed,
            teacher_name=args.teacher
        )
    else:
        # Single Run Mode (default)
        if args.no_distillation:
            print("[KnowLight] Running single Baseline LightGCN training trial (WITHOUT Distillation)...")
            train_config.setdefault("distillation", {})["active"] = False
        else:
            distiller_name = dist_cfg.get("type", "score").upper()
            print(f"[KnowLight] Running Student LightGCN with Distiller: {distiller_name}...")
            train_config.setdefault("distillation", {})["active"] = True

        device = get_device(train_config.get("experiment", {}).get("device", "cuda"))
        print(f"[KnowLight] Running on device: {device}")

        data_loader = KnowLightDataLoader(
            dataset_name=train_config["dataset"]["name"],
            data_dir=train_config["dataset"]["data_dir"],
            min_uc=train_config["dataset"]["min_uc"],
            min_ic=train_config["dataset"]["min_ic"]
        )
        data_loader.load_data()

        from knowlight.paths import STUDENTS_DIR, DISTILLED_STUDENTS_DIR, get_latest_or_next_run_dir
        import json

        model_name = train_config.get("student", {}).get("model", "lightgcn").lower()
        dataset_name = train_config["dataset"]["name"]
        is_distilled = train_config.get("distillation", {}).get("active", True)
        teacher_name = getattr(args, "teacher", "Qwen3.5-4B")
        distiller_tag = dist_cfg.get("type", "score").lower()

        if is_distilled:
            base_dir = DISTILLED_STUDENTS_DIR / model_name / dataset_name
            teacher_tag = Path(teacher_name).name.lower().replace("-", "_").replace(".", "_")
            prefix = f"{teacher_tag}_{distiller_tag}_run"
            out_dir = get_latest_or_next_run_dir(base_dir, prefix, resume=True)
        else:
            base_dir = STUDENTS_DIR / model_name / dataset_name
            out_dir = get_latest_or_next_run_dir(base_dir, "baseline_run", resume=True)

        trainer = Trainer(
            train_config=train_config,
            llm_config=llm_config,
            data_loader=data_loader,
            device=device,
            teacher_name=args.teacher
        )

        test_metrics = trainer.train(out_dir=out_dir)

        model_save_path = out_dir / f"best_student_lightgcn_{distiller_tag}.pt" if is_distilled else out_dir / "best_student_lightgcn.pt"
        torch.save(trainer.student.state_dict(), model_save_path)
        print(f"[KnowLight] Saved student model weights to: {model_save_path}")

        if is_distilled and hasattr(trainer, "distiller") and trainer.distiller is not None:
            distiller_save_path = out_dir / "best_distiller.pt"
            torch.save(trainer.distiller.state_dict(), distiller_save_path)
            print(f"[KnowLight] Saved distiller weights to: {distiller_save_path}")

        report_file_name = f"{distiller_tag}_run_results.json" if is_distilled else "baseline_run_results.json"
        report_file = out_dir / report_file_name
        report_data = {
            "model": model_name,
            "dataset": dataset_name,
            "distillation": is_distilled,
            "distiller": distiller_tag if is_distilled else None,
            "teacher": teacher_name if is_distilled else None,
            "config": train_config,
            "metrics": {k: float(v) for k, v in test_metrics.items()}
        }
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)
        print(f"[KnowLight] Saved evaluation report to: {report_file}")


if __name__ == "__main__":
    main()
