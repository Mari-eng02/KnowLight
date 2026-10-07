import os
import sys
import json
import argparse
import logging
from pathlib import Path

import torch

# Ensure root workspace directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from knowlight.utils.helpers import set_seed, load_yaml_config, get_device
from knowlight.datasets.data_loader import KnowLightDataLoader
from knowlight.student.lightgcn import LightGCN
from knowlight.training.evaluator import Evaluator
from knowlight.paths import STUDENTS_DIR, DISTILLED_STUDENTS_DIR, CHECKPOINTS_DIR

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def evaluate_student(
    dataset_name: str,
    model_path: str = None,
    config_path: str = "config/train_student.yaml",
    is_distilled: bool = True,
    split: str = "test"
):
    """
    Evaluates a trained Student LightGCN model (distilled or baseline) on test/val set.
    """
    train_config = load_yaml_config(config_path, dataset_name=dataset_name)
    
    device = get_device(train_config.get("experiment", {}).get("device", "cuda"))
    logging.info(f"Evaluating Student LightGCN model on dataset '{dataset_name}' ({split.upper()} split) on device: {device}...")

    # Load dataset
    data_loader = KnowLightDataLoader(
        dataset_name=train_config["dataset"]["name"],
        data_dir=train_config["dataset"]["data_dir"],
        min_uc=train_config["dataset"]["min_uc"],
        min_ic=train_config["dataset"]["min_ic"]
    )
    data_loader.load_data()

    # Auto-resolve checkpoint model path if not provided
    if not model_path:
        model_name = train_config.get("student", {}).get("model", "lightgcn").lower()
        base_dir = (DISTILLED_STUDENTS_DIR / model_name / dataset_name) if is_distilled else (STUDENTS_DIR / model_name / dataset_name)
        
        found_ckpt = None
        if base_dir.exists():
            run_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()], key=lambda d: d.name, reverse=True)
            for rdir in run_dirs:
                for pt_name in ["best_student_lightgcn.pt", "best_student_lightgcn_contrastive.pt"]:
                    pt_file = rdir / pt_name
                    if pt_file.exists():
                        found_ckpt = pt_file
                        break
                if found_ckpt:
                    break

        if found_ckpt:
            model_path = str(found_ckpt)
        else:
            raise FileNotFoundError(f"Could not automatically locate student checkpoint under: {base_dir}. Please specify --model_path explicitly.")

    logging.info(f"Loading Student model weights from: '{model_path}'...")

    # Initialize LightGCN model
    student_cfg = train_config.get("student", {})
    student_model = LightGCN(
        num_users=data_loader.num_users,
        num_items=data_loader.num_items,
        embedding_dim=student_cfg.get("embedding_dim", 64),
        n_layers=student_cfg.get("n_layers", 3),
        keep_prob=student_cfg.get("keep_prob", 0.6)
    ).to(device)

    # Set graph adjacency
    student_model.set_graph(data_loader.train_matrix, device=device)

    # Load state dict
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    student_model.load_state_dict(state_dict)
    student_model.eval()

    # Choose split target items
    if split in ("val", "validation"):
        target_pos_items = data_loader.val_pos_items
    else:
        target_pos_items = data_loader.test_pos_items

    # Evaluate using Evaluator
    top_k_list = train_config.get("evaluation", {}).get("top_k", [1, 5, 10, 20])
    evaluator = Evaluator(top_k_list=top_k_list)
    is_test = split not in ("val", "validation")
    results = evaluator.evaluate(student_model, data_loader, target_pos_items, device, is_test=is_test)

    # Display results
    ckpt_dir_name = Path(model_path).parent.parent.name # 'distilled_students' or 'students'
    model_type_label = "Distilled Student" if ckpt_dir_name == "distilled_students" else "Baseline Student (Non-distilled)"

    print("=" * 75)
    print(f"  KnowLight {model_type_label} Evaluation Results")
    print(f"  Dataset: {dataset_name.upper()} | Split: {split.upper()} | Checkpoint: {Path(model_path).name}")
    print("=" * 75)
    for k, v in results.items():
        print(f"  {k:<15} : {v:.4f}")
    print("=" * 75)

    # Save metrics to metrics.json in model directory
    metrics_file = Path(model_path).parent / f"metrics_{split}.json"
    metrics_dict = {
        "dataset": dataset_name,
        "split": split,
        "model_type": model_type_label,
        "model_path": str(model_path),
        "evaluated_users": len(target_pos_items),
        "metrics": {k: float(v) for k, v in results.items()}
    }
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4)
    logging.info(f"Successfully saved evaluation metrics to: {metrics_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained LightGCN Student Model (Distilled or Baseline)")
    parser.add_argument("--dataset", type=str, default="movielens", help="Dataset name ('movielens', 'amazon_beauty', etc.)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to best_student_lightgcn.pt checkpoint file (optional)")
    parser.add_argument("--config", type=str, default="config/train_student.yaml", help="Path to student config YAML")
    parser.add_argument("--distilled", action="store_true", default=True, help="Evaluate distilled student (default: True)")
    parser.add_argument("--baseline", action="store_true", default=False, help="Evaluate baseline non-distilled student")
    parser.add_argument("--split", type=str, default="test", help="Data split to evaluate on ('test' or 'val')")

    args = parser.parse_args()
    is_distilled = not args.baseline

    evaluate_student(
        dataset_name=args.dataset,
        model_path=args.model_path,
        config_path=args.config,
        is_distilled=is_distilled,
        split=args.split
    )


if __name__ == "__main__":
    main()
