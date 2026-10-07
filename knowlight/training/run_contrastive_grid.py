import os
import sys
import json
import time
import argparse
import logging
import itertools
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure root workspace directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from knowlight.utils.helpers import set_seed, load_yaml_config, get_device
from knowlight.datasets.data_loader import KnowLightDataLoader
from knowlight.datasets import BPRDataset
from knowlight.datasets.batch import InteractionBatch
from knowlight.datasets.interactions import InteractionDataset
from knowlight.student.lightgcn import LightGCN
from knowlight.distillation import (
    ContrastiveDistiller,
    DEDistiller,
    HTDistiller,
)
from knowlight.teachers.state import TeacherState
from knowlight.teachers.llmemb.llmemb_teacher import LLMEmbTeacher
from knowlight.training.evaluator import Evaluator
from knowlight.paths import DISTILLED_STUDENTS_DIR, get_latest_or_next_run_dir

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

DEFAULT_LAMBDA_ITEMS = [0.0, 0.001, 0.01, 0.05, 0.1]
DEFAULT_LAMBDA_USERS = [0.0, 0.001, 0.01, 0.05, 0.1]
DEFAULT_TAUS = [0.05, 0.1, 0.2, 0.5]

DEFAULT_LAMBDA_DE = [0.0001, 0.001, 0.01, 0.1]
DEFAULT_LAMBDA_HTD = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]


def run_contrastive_grid(
    base_config_path: str = "config/train_student_contrastive_distill.yaml",
    dataset_name: str = "movielens",
    teacher_name: str = "Qwen3.5-4B",
    distiller_type: str = "contrastive",
    lambda_item_list: List[float] = None,
    lambda_user_list: List[float] = None,
    tau_list: List[float] = None,
    lambda_de_list: List[float] = None,
    lambda_td_list: List[float] = None,
    epochs: int = None,
    seed: int = 42
):
    """
    Executes a Grid Search over embedding distillation weights:
    - If contrastive: lambda_item, lambda_user, and tau
    - If de: lambda_de with fixed num_experts=20, temp=0.1
    - If htd: lambda_td with fixed alpha=0.5, groups=40, entity_sample_size=256
    """
    set_seed(seed)
    distiller_type = str(distiller_type).lower()

    train_config = load_yaml_config(base_config_path, dataset_name=dataset_name)
    if epochs:
        train_config["training"]["epochs"] = epochs

    lr_val = train_config.get("training", {}).get("lr")
    wd_val = train_config.get("training", {}).get("weight_decay")
    device = get_device(train_config.get("experiment", {}).get("device", "cuda"))
    logging.info(f"[Embedding Grid Search] Running on device: {device} | Distiller: {distiller_type.upper()} | Dataset: '{dataset_name}' | LR: {lr_val} | Weight Decay: {wd_val} | Teacher: '{teacher_name}'")
    print(f"[{distiller_type.upper()} Grid Search] Dataset: '{dataset_name}' | LR: {lr_val} | Weight Decay: {wd_val} | Teacher: '{teacher_name}'")

    # Load dataset once
    data_loader = KnowLightDataLoader(
        dataset_name=dataset_name,
        data_dir=train_config["dataset"]["data_dir"],
        min_uc=train_config["dataset"]["min_uc"],
        min_ic=train_config["dataset"]["min_ic"]
    )
    data_loader.load_data()

    num_users = data_loader.num_users
    num_items = data_loader.num_items

    # Pre-load teacher embeddings once across all trials
    student_cfg = train_config.get("student", {})
    embedding_dim = student_cfg.get("embedding_dim", 64)

    logging.info(f"[Embedding Grid Search] Pre-loading Teacher Embeddings ({teacher_name}) for dataset...")
    llmemb_teacher = LLMEmbTeacher(dataset_name=dataset_name, embedding_dim=embedding_dim, teacher_name=teacher_name)
    llmemb_teacher.load_item_embeddings(num_items=num_items)
    llmemb_teacher.load_user_embeddings(num_users=num_users, train_dict=data_loader.train_dict)

    raw_item_embeds = (llmemb_teacher.intermediate_item_embeddings if llmemb_teacher.intermediate_item_embeddings is not None else llmemb_teacher.item_embeddings).to(device)
    raw_user_embeds = (llmemb_teacher.intermediate_user_embeddings if hasattr(llmemb_teacher, "intermediate_user_embeddings") and llmemb_teacher.intermediate_user_embeddings is not None else llmemb_teacher.user_embeddings).to(device)

    in_dim = raw_item_embeds.shape[1]

    # Setup directories
    base_dir = DISTILLED_STUDENTS_DIR / "lightgcn" / dataset_name
    teacher_tag = Path(teacher_name).name.lower().replace("-", "_").replace(".", "_")
    prefix = f"{teacher_tag}_{distiller_type}_grid"
    out_dir = get_latest_or_next_run_dir(base_dir, prefix, resume=True)
    grid_ckpt_file = out_dir / "checkpoint_grid_state.json"
    ckpt_student_path = out_dir / "checkpoint_best_student.pt"
    ckpt_distiller_path = out_dir / "checkpoint_best_distiller.pt"

    dataset_obj = InteractionDataset.from_train_dict(data_loader.train_dict, num_users, num_items)
    teacher_state = TeacherState(user_embeddings=raw_user_embeds, item_embeddings=raw_item_embeds)

    results = []
    completed_keys = set()
    best_ndcg = -1.0
    best_student_state = None
    best_distiller_state = None
    best_trial_params = None
    search_start_time = time.time()

    if grid_ckpt_file.exists():
        try:
            with open(grid_ckpt_file, "r", encoding="utf-8") as f:
                saved_grid = json.load(f)
            results = saved_grid.get("completed_trials", [])
            completed_keys = {t["param_key"] for t in results}
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

    evaluator = Evaluator(top_k_list=train_config.get("evaluation", {}).get("top_k", [5, 10, 20]))
    bpr_ds = BPRDataset(num_users, num_items, train_dict=data_loader.train_dict, train_matrix=data_loader.train_matrix)

    # Build trial grid based on distiller
    if distiller_type == "de":
        lambda_de_list = lambda_de_list or DEFAULT_LAMBDA_DE
        param_grid = [{"param_key": str(l), "lambda_de": float(l), "num_experts": 20, "temperature": 0.1} for l in lambda_de_list]
    elif distiller_type == "htd":
        lambda_td_list = lambda_td_list or DEFAULT_LAMBDA_HTD
        param_grid = [{"param_key": str(l), "lambda_td": float(l), "alpha": 0.5, "num_groups": 40, "entity_sample_size": 256} for l in lambda_td_list]
    else:
        lambda_item_list = lambda_item_list or DEFAULT_LAMBDA_ITEMS
        lambda_user_list = lambda_user_list or DEFAULT_LAMBDA_USERS
        tau_list = tau_list or DEFAULT_TAUS
        param_grid = [
            {"param_key": f"{l_i}_{l_u}_{t}", "lambda_item": float(l_i), "lambda_user": float(l_u), "tau": float(t)}
            for l_i, l_u, t in itertools.product(lambda_item_list, lambda_user_list, tau_list)
        ]

    total_trials = len(param_grid)
    logging.info(f"[{distiller_type.upper()} Grid Search] Launching Grid Search over {total_trials} parameter combinations.")

    for idx, params in enumerate(param_grid, start=1):
        key = params["param_key"]
        if key in completed_keys:
            print(f"\n[{distiller_type.upper()} Grid Search] Trial {idx}/{total_trials} ({key}) already completed. Skipping!")
            continue

        # Reset random seed for every trial to ensure identical weight initialization
        # and identical negative sampling sequences across all embedding distillation configurations
        set_seed(seed)

        print("\n" + "=" * 80)
        print(f" {distiller_type.upper()} Grid Search Trial {idx}/{total_trials} -> Params: {params} (Seed: {seed})")
        print("=" * 80)

        trial_start = time.time()

        # Re-initialize fresh Student LightGCN for each trial
        student = LightGCN(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            n_layers=student_cfg.get("n_layers", 3),
            keep_prob=student_cfg.get("keep_prob", 0.6)
        ).to(device)
        student.set_graph(data_loader.train_matrix, device=device)

        # Distiller Initialization
        if distiller_type == "de":
            distiller = DEDistiller(
                teacher_dim=in_dim,
                student_dim=embedding_dim,
                num_experts=20,
                lambda_de=params["lambda_de"],
                temperature=0.1
            ).to(device)
        elif distiller_type == "htd":
            distiller = HTDistiller(
                teacher_dim=in_dim,
                student_dim=embedding_dim,
                lambda_td=params["lambda_td"],
                alpha=0.5,
                num_groups=40,
                topology_mode="group_pe",
                initial_tau=1.0,
                min_tau=1e-10,
                decay_epochs=100,
                entity_sample_size=256
            ).to(device)

        else:
            distiller = ContrastiveDistiller(
                in_dim=in_dim,
                hidden_dim=512,
                out_dim=embedding_dim,
                lambda_item=params["lambda_item"],
                lambda_user=params["lambda_user"],
                tau=params["tau"]
            ).to(device)

        if hasattr(distiller, "on_train_start"):
            distiller.on_train_start(teacher_state=teacher_state, dataset=dataset_obj)

        lr = float(train_config.get("training", {}).get("lr", 0.001))
        l2_reg = float(train_config.get("training", {}).get("weight_decay", 0.0001))

        opt_params = list(student.parameters()) + list(distiller.parameters())
        optimizer = torch.optim.Adam(opt_params, lr=lr, weight_decay=0.0)

        train_loader = DataLoader(
            bpr_ds,
            batch_size=train_config.get("training", {}).get("batch_size", 1024),
            shuffle=True
        )

        n_epochs = train_config.get("training", {}).get("epochs", 20)
        eval_step = train_config.get("training", {}).get("eval_step", 1)
        trial_patience = train_config.get("training", {}).get("patience", 10)
        trial_patience_counter = 0
        trial_best_ndcg = 0.0
        trial_best_student_state = None
        trial_best_distiller_state = None

        for epoch in range(1, n_epochs + 1):
            student.train()
            if hasattr(distiller, "train"):
                distiller.train()
            if hasattr(distiller, "on_epoch_start"):
                distiller.on_epoch_start()

            for batch_u, batch_pos, batch_neg in train_loader:
                batch_u = batch_u.to(device)
                batch_pos = batch_pos.to(device)
                batch_neg = batch_neg.to(device)

                optimizer.zero_grad()

                bpr_loss = student.bpr_loss(batch_u, batch_pos, batch_neg, decay=l2_reg)

                if distiller_type in ["de", "htd"]:
                    batch_obj = InteractionBatch(users=batch_u, pos_items=batch_pos, neg_items=batch_neg)
                    l_dist_tensor = distiller.compute_loss(student, batch_obj, None)
                    loss = bpr_loss + l_dist_tensor
                else:
                    gcn_user_embeds = student.get_user_embedding(batch_u)
                    gcn_item_embeds = student.get_item_embedding(batch_pos)
                    batch_llmemb_user = raw_user_embeds[batch_u]
                    batch_llmemb_item = raw_item_embeds[batch_pos]
                    loss_dict = distiller(
                        bpr_loss=bpr_loss,
                        gcn_item_embeds=gcn_item_embeds,
                        llmemb_item_embeds=batch_llmemb_item,
                        gcn_user_embeds=gcn_user_embeds,
                        llmemb_user_embeds=batch_llmemb_user
                    )
                    loss = loss_dict["total_loss"]

                loss.backward()
                optimizer.step()

            if epoch % eval_step == 0:
                val_metrics = evaluator.evaluate(student, data_loader, data_loader.val_pos_items, device)
                val_ndcg = val_metrics.get("NDCG@20", 0.0)
                if val_ndcg > trial_best_ndcg:
                    trial_best_ndcg = val_ndcg
                    trial_patience_counter = 0
                    trial_best_student_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
                    trial_best_distiller_state = {k: v.cpu().clone() for k, v in distiller.state_dict().items()}
                else:
                    trial_patience_counter += eval_step
                    if trial_patience_counter >= trial_patience:
                        print(f" [{distiller_type.upper()} Grid Search] Early stopping triggered at epoch {epoch} (best val NDCG@20: {trial_best_ndcg:.4f})")
                        break

        if trial_best_student_state is not None:
            student.load_state_dict(trial_best_student_state)
        if trial_best_distiller_state is not None:
            distiller.load_state_dict(trial_best_distiller_state)

        # Test evaluation after training
        test_metrics = evaluator.evaluate(student, data_loader, data_loader.test_pos_items, device)
        trial_elapsed = time.time() - trial_start
        current_ndcg = test_metrics.get("NDCG@20", 0.0)

        trial_record = {
            "trial_id": idx,
            "param_key": key,
            "params": params,
            "lr": lr,
            "weight_decay": l2_reg,
            "metrics": {k: float(v) for k, v in test_metrics.items()},
            "elapsed_seconds": round(trial_elapsed, 2)
        }
        results.append(trial_record)
        completed_keys.add(key)

        if current_ndcg > best_ndcg:
            best_ndcg = current_ndcg
            best_trial_params = params
            best_student_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            best_distiller_state = {k: v.cpu().clone() for k, v in distiller.state_dict().items()}
            torch.save(best_student_state, ckpt_student_path)
            torch.save(best_distiller_state, ckpt_distiller_path)
            print(f" >>> New Best Trial Found! Params={params} (NDCG@20: {best_ndcg:.4f}) <<<")

        grid_save_data = {
            "completed_trials": results,
            "best_ndcg": best_ndcg,
            "best_trial_params": best_trial_params
        }
        with open(grid_ckpt_file, "w", encoding="utf-8") as f:
            json.dump(grid_save_data, f, indent=4)

    total_search_time = time.time() - search_start_time
    report_file = out_dir / f"grid_{distiller_type}_results.json"

    summary_output = {
        "pipeline": f"embedding_distillation_{distiller_type}",
        "distiller": distiller_type,
        "dataset": dataset_name,
        "teacher": teacher_name,
        "lr": lr_val,
        "weight_decay": wd_val,
        "total_trials": total_trials,
        "total_elapsed_seconds": round(total_search_time, 2),
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

        distiller_save_path = out_dir / f"best_{distiller_type}_distiller.pt"
        torch.save(best_distiller_state, distiller_save_path)
        print(f"[KnowLight Grid Search] Saved best distiller weights (Params={best_trial_params}) to: {distiller_save_path}")

    # Cleanup intermediate checkpoints
    for temp_p in [grid_ckpt_file, ckpt_student_path, ckpt_distiller_path]:
        if temp_p.exists():
            try:
                temp_p.unlink()
            except Exception:
                pass

    print("\n" + "=" * 80)
    print(f"  KnowLight {distiller_type.upper()} Embedding Distillation Grid Search Final Summary ({dataset_name.upper()})")
    print("=" * 80)
    print(f"{'Params':<35} | {'NDCG@20':<12} | {'Recall@20':<12} | {'HR@20':<12}")
    print("-" * 75)
    for r in results:
        m = r["metrics"]
        print(f"{r['param_key']:<35} | {m.get('NDCG@20', 0.0):<12.4f} | {m.get('Recall@20', 0.0):<12.4f} | {m.get('HR@20', 0.0):<12.4f}")
    print("=" * 80)
    print(f" Best Params overall : {best_trial_params} (NDCG@20: {best_ndcg:.4f})")
    print(f" Results saved to    : {report_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Pipeline 2 Embedding Distillation Grid Search (Contrastive, DE, HTD)")
    parser.add_argument("--config", type=str, default="config/train_student_contrastive_distill.yaml", help="Path to base config file")
    parser.add_argument("--dataset", type=str, default="movielens", help="Dataset name")
    parser.add_argument("--teacher", type=str, default="Qwen3.5-4B", help="Teacher model name")
    parser.add_argument("--distiller", type=str, default="contrastive", choices=["contrastive", "de", "htd"], help="Distillation method for grid search")
    parser.add_argument("--lambda_item", nargs="+", type=float, default=None, help="Lambda item values for contrastive grid")
    parser.add_argument("--lambda_user", nargs="+", type=float, default=None, help="Lambda user values for contrastive grid")
    parser.add_argument("--tau", nargs="+", type=float, default=None, help="Tau temperature values for contrastive grid")
    parser.add_argument("--lambda_de", nargs="+", type=float, default=None, help="Lambda DE values to test in grid search")
    parser.add_argument("--lambda_td", nargs="+", type=float, default=None, help="Lambda TD values to test in grid search")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run_contrastive_grid(
        base_config_path=args.config,
        dataset_name=args.dataset,
        teacher_name=args.teacher,
        distiller_type=args.distiller,
        lambda_item_list=args.lambda_item,
        lambda_user_list=args.lambda_user,
        tau_list=args.tau,
        lambda_de_list=args.lambda_de,
        lambda_td_list=args.lambda_td,
        epochs=args.epochs,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
