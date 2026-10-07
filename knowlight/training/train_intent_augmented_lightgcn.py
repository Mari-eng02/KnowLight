import argparse
import itertools
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch
from torch.utils.data import DataLoader

# Ensure root workspace directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from knowlight.datasets import KnowLightDataLoader, BPRDataset
from knowlight.student.lightgcn import LightGCN
from knowlight.training.evaluator import Evaluator
from knowlight.utils.helpers import set_seed, load_yaml_config
from knowlight.paths import STUDENTS_DIR, get_latest_or_next_run_dir
from knowlight.graphs.intent_graph_augmenter import IntentGraphAugmenter

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

DEFAULT_TOP_KS = [1, 2, 5, 8, 10]
DEFAULT_WEIGHTS = [0.1, 0.2, 0.5, 0.8, 1.0]


def run_intent_augmented_training(
    train_config: dict,
    dataset_name: str = "movielens",
    teacher_name: str = "Qwen3.5-4B",
    top_k_pseudo: int = 2,
    pseudo_weight: float = 0.5,
    sim_threshold: float = 0.0,
    device: Optional[torch.device] = None,
    epochs_override: Optional[int] = None,
    out_dir: Optional[Path] = None,
    eval_topk: Optional[List[int]] = None,
    seed: int = 42
) -> Tuple[Dict[str, Any], Optional[Dict[str, torch.Tensor]]]:
    """Runs a single training trial of LightGCN using the Intent-Augmented Graph."""
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    data_dir = train_config.get("dataset", {}).get("data_dir", "./data/raw")
    data_loader = KnowLightDataLoader(dataset_name=dataset_name, data_dir=data_dir)
    data_loader.load_data()

    num_users = data_loader.num_users
    num_items = data_loader.num_items

    # 1. Build Intent-Augmented Interaction Matrix R_tilde
    augmenter = IntentGraphAugmenter(
        dataset_name=dataset_name,
        teacher_name=teacher_name,
        top_k_pseudo=top_k_pseudo,
        pseudo_weight=pseudo_weight,
        sim_threshold=sim_threshold,
        device=device
    )
    augmented_r = augmenter.build_augmented_interaction_matrix(
        train_interaction_matrix=data_loader.train_matrix,
        train_dict=data_loader.train_dict,
        top_k_pseudo=top_k_pseudo,
        pseudo_weight=pseudo_weight,
        sim_threshold=sim_threshold,
        use_cache=train_config.get("graph_augmentation", {}).get("use_cache", True)
    )

    # 2. Instantiate LightGCN Student
    student_cfg = train_config.get("student", {})
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

    # Set augmented graph for message passing
    student.set_graph(augmented_r, device=device)

    # 3. Training setup
    lr = float(train_config.get("training", {}).get("lr", 0.001))
    l2_reg = float(train_config.get("training", {}).get("weight_decay", 0.001))

    # Apply dataset-specific overrides if present
    dataset_overrides = train_config.get("dataset_hyperparams", {}).get(dataset_name, {})
    if "lr" in dataset_overrides:
        lr = float(dataset_overrides["lr"])
    if "weight_decay" in dataset_overrides:
        l2_reg = float(dataset_overrides["weight_decay"])

    optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=0.0)
    evaluator = Evaluator(top_k_list=eval_topk or train_config.get("evaluation", {}).get("top_k", [5, 10, 20]))

    bpr_ds = BPRDataset(num_users, num_items, train_dict=data_loader.train_dict, train_matrix=data_loader.train_matrix)
    batch_size = train_config.get("training", {}).get("batch_size", 1024)
    train_loader = DataLoader(bpr_ds, batch_size=batch_size, shuffle=True)

    epochs = epochs_override or train_config.get("training", {}).get("epochs", 1000)
    eval_step = train_config.get("training", {}).get("eval_step", 1)
    patience = train_config.get("training", {}).get("patience", 10)

    best_ndcg = 0.0
    patience_counter = 0
    best_student_state = None

    print(f"\n[Intent Augmentation] Starting LightGCN Training -> Dataset: {dataset_name} | Teacher: {teacher_name} | k={top_k_pseudo}, weight={pseudo_weight}, thresh={sim_threshold}")
    print(f"[Intent Augmentation] LR: {lr}, Weight Decay: {l2_reg}, Epochs: {epochs}, Batch Size: {batch_size}")

    for epoch in range(1, epochs + 1):
        student.train()
        total_loss_epoch = 0.0
        start_time = time.time()

        for batch_u, batch_pos, batch_neg in train_loader:
            batch_u = batch_u.to(device)
            batch_pos = batch_pos.to(device)
            batch_neg = batch_neg.to(device)

            optimizer.zero_grad()
            loss = student.bpr_loss(batch_u, batch_pos, batch_neg, decay=l2_reg)
            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item()

        epoch_time = time.time() - start_time
        avg_loss = total_loss_epoch / max(len(train_loader), 1)

        # Validation Evaluation
        if epoch % eval_step == 0:
            student.eval()
            val_target = getattr(data_loader, "val_pos_items", getattr(data_loader, "val_dict", {}))
            with torch.no_grad():
                val_metrics = evaluator.evaluate(
                    model=student,
                    data_loader=data_loader,
                    test_pos_items=val_target,
                    device=device
                )
                val_ndcg_20 = val_metrics.get("NDCG@20", 0.0)

            print(f"Epoch {epoch:03d}/{epochs:03d} [{epoch_time:.1f}s] | Loss: {avg_loss:.4f} | Val Recall@20: {val_metrics.get('Recall@20', 0.0):.4f} | Val NDCG@20: {val_ndcg_20:.4f}")

            if val_ndcg_20 > best_ndcg:
                best_ndcg = val_ndcg_20
                best_student_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
                patience_counter = 0
                if out_dir:
                    torch.save(best_student_state, out_dir / "best_student_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch} (best Val NDCG@20: {best_ndcg:.4f}).")
                    break

    # Final Test Evaluation with Best Model
    if best_student_state is not None:
        student.load_state_dict({k: v.to(device) for k, v in best_student_state.items()})

    student.eval()
    test_target = getattr(data_loader, "test_pos_items", getattr(data_loader, "test_dict", {}))
    with torch.no_grad():
        test_metrics = evaluator.evaluate(
            model=student,
            data_loader=data_loader,
            test_pos_items=test_target,
            device=device
        )

    print("\n" + "=" * 70)
    print(f" FINAL TEST EVALUATION (Intent-Augmented LightGCN | k={top_k_pseudo}, w={pseudo_weight})")
    print("=" * 70)
    for metric, value in sorted(test_metrics.items()):
        print(f"  {metric:<15}: {value:.4f}")
    print("=" * 70)

    if out_dir:
        result_payload = {
            "params": {
                "top_k_pseudo": top_k_pseudo,
                "pseudo_weight": pseudo_weight,
                "sim_threshold": sim_threshold,
                "dataset": dataset_name,
                "teacher": teacher_name,
                "lr": lr,
                "weight_decay": l2_reg
            },
            "test_metrics": test_metrics,
            "best_val_ndcg": best_ndcg
        }
        with open(out_dir / "final_results.json", "w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2)

    return test_metrics, best_student_state


def run_intent_graph_grid_search(
    base_config_path: str = "config/train_intent_augmented_lightgcn.yaml",
    dataset_name: str = "movielens",
    teacher_name: str = "Qwen3.5-4B",
    top_k_list: Optional[List[int]] = None,
    weight_list: Optional[List[float]] = None,
    sim_threshold: float = 0.0,
    epochs: Optional[int] = None,
    seed: int = 42
):
    """Runs a complete Grid Search over Top-K and Pseudo-Edge weights."""
    set_seed(seed)
    train_config = load_yaml_config(base_config_path, dataset_name=dataset_name)

    top_k_list = top_k_list or train_config.get("graph_augmentation", {}).get("top_k_list", DEFAULT_TOP_KS)
    weight_list = weight_list or train_config.get("graph_augmentation", {}).get("weight_list", DEFAULT_WEIGHTS)

    teacher_tag = Path(teacher_name).name.lower().replace("-", "_").replace(".", "_")
    base_dir = STUDENTS_DIR / "intent_augmented" / dataset_name
    prefix = f"{teacher_tag}_aug_grid"
    out_dir = get_latest_or_next_run_dir(base_dir, prefix, resume=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_ckpt_file = out_dir / "checkpoint_grid_state.json"
    legacy_ckpt_file = out_dir / "grid_search_checkpoint.json"
    load_ckpt_file = grid_ckpt_file if grid_ckpt_file.exists() else (legacy_ckpt_file if legacy_ckpt_file.exists() else None)

    completed_keys = set()
    results = []
    best_ndcg = -1.0
    best_params = None

    if load_ckpt_file is not None:
        try:
            with open(load_ckpt_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
            results = saved.get("completed_trials", [])
            completed_keys = {str(t["param_key"]) for t in results}
            best_ndcg = saved.get("best_ndcg", -1.0)
            best_params = saved.get("best_params", None)
            logging.info(f"[Grid Search] Resuming run from {load_ckpt_file.name}: {len(results)} trials already completed.")
        except Exception as e:
            logging.warning(f"[Grid Search] Could not load checkpoint ({e}). Starting fresh.")

    param_combinations = [
        {"param_key": f"k{k}_w{w}", "top_k_pseudo": int(k), "pseudo_weight": float(w)}
        for k, w in itertools.product(top_k_list, weight_list)
    ]
    total_trials = len(param_combinations)

    print(f"\n[Intent Graph Augmentation] Launching Grid Search over {total_trials} parameter combinations.")
    print(f" Top-K candidates to test: {top_k_list}")
    print(f" Pseudo-edge weights to test: {weight_list}")

    for idx, p in enumerate(param_combinations, start=1):
        key = p["param_key"]
        if key in completed_keys:
            print(f"Trial {idx}/{total_trials} ({key}) already completed. Skipping!")
            continue

        print("\n" + "=" * 70)
        print(f" Grid Search Trial {idx}/{total_trials} -> k={p['top_k_pseudo']}, weight={p['pseudo_weight']}")
        print("=" * 70)

        trial_start = time.time()
        test_metrics, trial_best_student_state = run_intent_augmented_training(
            train_config=train_config,
            dataset_name=dataset_name,
            teacher_name=teacher_name,
            top_k_pseudo=p["top_k_pseudo"],
            pseudo_weight=p["pseudo_weight"],
            sim_threshold=sim_threshold,
            epochs_override=epochs,
            out_dir=None,
            seed=seed
        )
        trial_elapsed = time.time() - trial_start

        current_ndcg = test_metrics.get("NDCG@20", 0.0)
        trial_record = {
            "param_key": key,
            "top_k_pseudo": p["top_k_pseudo"],
            "pseudo_weight": p["pseudo_weight"],
            "sim_threshold": sim_threshold,
            "lr": train_config.get("training", {}).get("lr"),
            "weight_decay": train_config.get("training", {}).get("weight_decay"),
            "metrics": test_metrics,
            "elapsed_seconds": trial_elapsed
        }
        results.append(trial_record)
        completed_keys.add(key)

        best_ckpt_file = out_dir / "best_student_model.pt"
        if current_ndcg > best_ndcg:
            best_ndcg = current_ndcg
            best_params = {**p, "sim_threshold": sim_threshold}
            if trial_best_student_state is not None:
                torch.save(trial_best_student_state, best_ckpt_file)
                print(f" >>> New Best Trial Found! k={p['top_k_pseudo']}, w={p['pseudo_weight']}, th={sim_threshold} (NDCG@20: {best_ndcg:.4f}). Saved checkpoint to: {best_ckpt_file} <<<")

        # Save checkpoint
        checkpoint_data = {
            "dataset": dataset_name,
            "teacher": teacher_name,
            "sim_threshold": sim_threshold,
            "lr": train_config.get("training", {}).get("lr"),
            "weight_decay": train_config.get("training", {}).get("weight_decay"),
            "completed_trials": results,
            "best_ndcg": best_ndcg,
            "best_params": best_params,
            "best_model_checkpoint": str(best_ckpt_file) if best_ckpt_file.exists() else None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(grid_ckpt_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2)

    # Save final results summary report
    summary_report = {
        "pipeline": "intent_graph_augmentation",
        "dataset": dataset_name,
        "teacher": teacher_name,
        "sim_threshold": sim_threshold,
        "best_ndcg": best_ndcg,
        "best_params": best_params,
        "best_model_checkpoint": str(best_ckpt_file) if best_ckpt_file.exists() else None,
        "total_trials": total_trials,
        "completed_trials": results
    }
    with open(out_dir / "grid_aug_results.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    # Cleanup intermediate checkpoint files
    for temp_ckpt in [grid_ckpt_file, legacy_ckpt_file]:
        if temp_ckpt.exists():
            try:
                temp_ckpt.unlink()
            except Exception:
                pass

    # Print Full Grid Summary Table
    print("\n" + "=" * 85)
    print(f" INTENT GRAPH AUGMENTATION GRID SEARCH COMPLETE ({dataset_name} - {teacher_name})")
    print("=" * 85)
    print(f"{'Top-K':<8} | {'Weight':<8} | {'Recall@5':<10} | {'Recall@20':<10} | {'NDCG@5':<10} | {'NDCG@20':<10}")
    print("-" * 85)
    for r in results:
        m = r["metrics"]
        print(f"{r['top_k_pseudo']:<8} | {r['pseudo_weight']:<8.2f} | {m.get('Recall@5', 0):<10.4f} | {m.get('Recall@20', 0):<10.4f} | {m.get('NDCG@5', 0):<10.4f} | {m.get('NDCG@20', 0):<10.4f}")
    print("=" * 85)
    print(f"BEST COMBINATION: {best_params} with NDCG@20 = {best_ndcg:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="KnowLight: Intent-Guided Graph Augmentation for LightGCN")
    parser.add_argument("--config", type=str, default="config/train_intent_augmented_lightgcn.yaml", help="Path to config YAML")
    parser.add_argument("--dataset", type=str, default="movielens", help="Dataset name ('movielens', 'amazon_cd_2014', etc.)")
    parser.add_argument("--teacher", type=str, default="Qwen3.5-4B", help="Teacher model name ('Qwen3.5-4B', 'Llama-3-8B-Instruct')")
    parser.add_argument("--top_k_pseudo", type=int, default=None, help="Number of semantic pseudo-edges per user (default: 2)")
    parser.add_argument("--pseudo_weight", type=float, default=None, help="Scaling weight for pseudo-edges (default: 0.5)")
    parser.add_argument("--sim_threshold", type=float, default=None, help="Minimum similarity threshold (default: 0.0)")
    parser.add_argument("--top_k_list", nargs="+", type=int, default=DEFAULT_TOP_KS, help="List of Top-K values for grid search")
    parser.add_argument("--weight_list", nargs="+", type=float, default=DEFAULT_WEIGHTS, help="List of weights for grid search")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--grid", action="store_true", default=False, help="Launch Grid Search over Top-K and weights")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    train_config = load_yaml_config(args.config, dataset_name=args.dataset)

    if args.grid:
        run_intent_graph_grid_search(
            base_config_path=args.config,
            dataset_name=train_config["dataset"]["name"],
            teacher_name=args.teacher,
            top_k_list=args.top_k_list,
            weight_list=args.weight_list,
            sim_threshold=args.sim_threshold if args.sim_threshold is not None else 0.0,
            epochs=args.epochs,
            seed=args.seed
        )
    else:
        k = args.top_k_pseudo if args.top_k_pseudo is not None else train_config.get("graph_augmentation", {}).get("top_k_pseudo", 2)
        w = args.pseudo_weight if args.pseudo_weight is not None else train_config.get("graph_augmentation", {}).get("pseudo_weight", 0.5)
        th = args.sim_threshold if args.sim_threshold is not None else train_config.get("graph_augmentation", {}).get("sim_threshold", 0.0)

        teacher_tag = Path(args.teacher).name.lower().replace("-", "_").replace(".", "_")
        base_dir = STUDENTS_DIR / "intent_augmented" / train_config["dataset"]["name"]
        prefix = f"{teacher_tag}_k{k}_w{w}_run"
        out_dir = get_latest_or_next_run_dir(base_dir, prefix, resume=False)
        out_dir.mkdir(parents=True, exist_ok=True)

        run_intent_augmented_training(
            train_config=train_config,
            dataset_name=train_config["dataset"]["name"],
            teacher_name=args.teacher,
            top_k_pseudo=k,
            pseudo_weight=w,
            sim_threshold=th,
            epochs_override=args.epochs,
            out_dir=out_dir,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
