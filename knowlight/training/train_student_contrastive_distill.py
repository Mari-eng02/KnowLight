from pathlib import Path
import argparse
import os
import sys
import time
import json
import torch
from torch.utils.data import DataLoader

# Ensure root workspace directory is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from knowlight.utils.helpers import set_seed, load_yaml_config, get_device
from knowlight.datasets.data_loader import KnowLightDataLoader
from knowlight.datasets import BPRDataset
from knowlight.datasets.batch import InteractionBatch
from knowlight.datasets.interactions import InteractionDataset
from knowlight.student.lightgcn import LightGCN
from knowlight.teachers.state import TeacherState
from knowlight.distillation import (
    ContrastiveDistiller,
    DEDistiller,
    HTDistiller,
)
from knowlight.teachers.llmemb.llmemb_teacher import LLMEmbTeacher
from knowlight.training.run_contrastive_grid import run_contrastive_grid
from knowlight.training.evaluator import Evaluator
from knowlight.paths import DISTILLED_STUDENTS_DIR, get_next_run_dir, get_latest_or_next_run_dir


def train_contrastive_pipeline(args):
    """
    Executes Pipeline 2: Teacher Embedding Distillation (Contrastive, DE, HTD).
    """
    distiller_type = getattr(args, "distiller", "contrastive").lower()

    if getattr(args, "grid", False):
        print(f"[Pipeline 2] Launching {distiller_type.upper()} Embedding Distillation Grid Search using config: {args.config}...")
        run_contrastive_grid(
            base_config_path=args.config,
            dataset_name=args.dataset,
            teacher_name=getattr(args, "teacher", "Qwen3.5-4B"),
            distiller_type=distiller_type,
            lambda_item_list=args.lambda_item,
            lambda_user_list=args.lambda_user,
            tau_list=args.tau,
            lambda_de_list=args.lambda_de if isinstance(args.lambda_de, list) else ([args.lambda_de] if args.lambda_de is not None else None),
            lambda_td_list=args.lambda_td if isinstance(args.lambda_td, list) else ([args.lambda_td] if args.lambda_td is not None else None),
            epochs=args.epochs,
            seed=args.seed
        )
        return

    train_config = load_yaml_config(args.config, dataset_name=args.dataset)

    # Overrides
    if args.epochs:
        train_config["training"]["epochs"] = args.epochs
    train_config["experiment"]["seed"] = args.seed

    set_seed(args.seed)

    dataset_name = train_config["dataset"]["name"]
    lr_val = train_config.get("training", {}).get("lr")
    wd_val = train_config.get("training", {}).get("weight_decay")
    device = get_device(train_config.get("experiment", {}).get("device", "cuda"))
    teacher_name = getattr(args, "teacher", "Qwen3.5-4B")
    print(f"[Pipeline 2] Distiller: {distiller_type.upper()} | Teacher: {teacher_name} | Dataset: '{dataset_name}' | LR: {lr_val} | Weight Decay: {wd_val} | Device: {device}")

    dataset_name = train_config["dataset"]["name"]
    data_loader = KnowLightDataLoader(
        dataset_name=dataset_name,
        data_dir=train_config["dataset"]["data_dir"],
        min_uc=train_config["dataset"]["min_uc"],
        min_ic=train_config["dataset"]["min_ic"]
    )
    data_loader.load_data()

    num_users = data_loader.num_users
    num_items = data_loader.num_items

    # 1. Initialize Student LightGCN
    student_cfg = train_config.get("student", {})
    embedding_dim = student_cfg.get("embedding_dim", 64)
    student = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        n_layers=student_cfg.get("n_layers", 3),
        keep_prob=student_cfg.get("keep_prob", 0.6)
    ).to(device)
    student.set_graph(data_loader.train_matrix, device=device)

    # 2. Load Teacher Item & User Embeddings
    print(f"[Pipeline 2] Loading Teacher Embeddings ({teacher_name})...")
    llmemb_teacher = LLMEmbTeacher(dataset_name=dataset_name, embedding_dim=embedding_dim, teacher_name=teacher_name)
    llmemb_teacher.load_item_embeddings(num_items=num_items)
    llmemb_teacher.load_user_embeddings(num_users=num_users, train_dict=data_loader.train_dict)

    raw_item_embeds = (llmemb_teacher.intermediate_item_embeddings if llmemb_teacher.intermediate_item_embeddings is not None else llmemb_teacher.item_embeddings).to(device)
    raw_user_embeds = (llmemb_teacher.intermediate_user_embeddings if hasattr(llmemb_teacher, "intermediate_user_embeddings") and llmemb_teacher.intermediate_user_embeddings is not None else llmemb_teacher.user_embeddings).to(device)

    in_dim = raw_item_embeds.shape[1]

    # Build TeacherState & InteractionDataset objects
    teacher_state = TeacherState(user_embeddings=raw_user_embeds, item_embeddings=raw_item_embeds)
    dataset_obj = InteractionDataset.from_train_dict(data_loader.train_dict, num_users, num_items)

    # 3. Initialize Selected Distiller (Embedding Distillation: Contrastive, DE, HTD)
    dist_cfg = train_config.get("distillation", {})

    if distiller_type == "de":
        raw_l_de = getattr(args, "lambda_de", dist_cfg.get("lambda_de", 0.01))
        lambda_de = float(raw_l_de[0] if isinstance(raw_l_de, list) else raw_l_de)
        distiller = DEDistiller(
            teacher_dim=in_dim,
            student_dim=embedding_dim,
            num_experts=20,
            lambda_de=lambda_de,
            temperature=0.1
        ).to(device)
        print(f" [Distiller DE] Initialized with lambda_de={lambda_de}, num_experts=20, temp=0.1")

    elif distiller_type == "htd":
        raw_l_td = getattr(args, "lambda_td", dist_cfg.get("lambda_td", 0.01))
        lambda_td = float(raw_l_td[0] if isinstance(raw_l_td, list) else raw_l_td)
        distiller = HTDistiller(
            teacher_dim=in_dim,
            student_dim=embedding_dim,
            lambda_td=lambda_td,
            alpha=0.5,
            num_groups=40,
            topology_mode="group_pe",
            initial_tau=1.0,
            min_tau=1e-10,
            decay_epochs=100,
            entity_sample_size=256
        ).to(device)
        print(f" [Distiller HTD] Initialized with teacher_dim={in_dim}, student_dim={embedding_dim}, lambda_td={lambda_td}, alpha=0.5, num_groups=40, entity_sample_size=256")


    else:
        # Default: ContrastiveDistiller
        distiller_type = "contrastive"
        lambda_item = float(dist_cfg.get("lambda_item", 0.1))
        lambda_user = float(dist_cfg.get("lambda_user", 0.1))
        tau = float(dist_cfg.get("tau", 0.2))
        distiller = ContrastiveDistiller(
            in_dim=in_dim,
            hidden_dim=512,
            out_dim=embedding_dim,
            lambda_item=lambda_item,
            lambda_user=lambda_user,
            tau=tau
        ).to(device)
        print(f" [Distiller Contrastive] Lambda Item: {lambda_item}, Lambda User: {lambda_user}, Temperature Tau: {tau}")

    # Trigger distiller initialization with TeacherState
    if hasattr(distiller, "on_train_start"):
        distiller.on_train_start(teacher_state=teacher_state, dataset=dataset_obj)

    # Optimizer
    lr = float(train_config.get("training", {}).get("lr", 0.001))
    l2_reg = float(train_config.get("training", {}).get("weight_decay", 0.0001))

    dist_params = list(distiller.parameters()) if hasattr(distiller, "parameters") else []
    params = list(student.parameters()) + dist_params
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0.0)

    evaluator = Evaluator(top_k_list=train_config.get("evaluation", {}).get("top_k", [5, 10, 20]))

    # Data Loader
    bpr_ds = BPRDataset(num_users, num_items, train_dict=data_loader.train_dict, train_matrix=data_loader.train_matrix)
    train_loader = DataLoader(
        bpr_ds,
        batch_size=train_config.get("training", {}).get("batch_size", 1024),
        shuffle=True
    )

    epochs = train_config.get("training", {}).get("epochs", 20)
    eval_step = train_config.get("training", {}).get("eval_step", 1)
    patience = train_config.get("training", {}).get("patience", 20)

    best_ndcg = 0.0
    patience_counter = 0
    best_student_state = None
    best_distiller_state = None

    teacher_name = getattr(args, "teacher", "Qwen3.5-4B")
    base_dir = DISTILLED_STUDENTS_DIR / "lightgcn" / dataset_name
    teacher_tag = Path(teacher_name).name.lower().replace("-", "_").replace(".", "_")
    prefix = f"{teacher_tag}_{distiller_type}_run"
    out_dir = get_latest_or_next_run_dir(base_dir, prefix, resume=True)
    temp_ckpt_path = out_dir / "temp_checkpoint.pt"

    start_epoch = 1
    if temp_ckpt_path.exists():
        try:
            ckpt = torch.load(temp_ckpt_path, map_location=device, weights_only=False)
            start_epoch = ckpt.get("epoch", 0) + 1
            student.load_state_dict(ckpt["student_state_dict"])
            if "distiller_state_dict" in ckpt and ckpt["distiller_state_dict"] and hasattr(distiller, "load_state_dict"):
                distiller.load_state_dict(ckpt["distiller_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            best_ndcg = ckpt.get("best_ndcg", 0.0)
            patience_counter = ckpt.get("patience_counter", 0)
            print(f"[Pipeline 2] Resuming training from epoch {start_epoch}/{epochs} (best val NDCG@20: {best_ndcg:.4f}).")
        except Exception as e:
            print(f"[Pipeline 2] Could not load epoch temp checkpoint ({e}). Starting fresh from epoch 1.")
            start_epoch = 1

    for epoch in range(start_epoch, epochs + 1):
        student.train()
        if hasattr(distiller, "train"):
            distiller.train()
        if hasattr(distiller, "on_epoch_start"):
            distiller.on_epoch_start()

        total_loss_epoch = 0.0
        bpr_loss_epoch = 0.0
        dist_loss_epoch = 0.0
        start_time = time.time()

        for batch_u, batch_pos, batch_neg in train_loader:
            batch_u = batch_u.to(device)
            batch_pos = batch_pos.to(device)
            batch_neg = batch_neg.to(device)

            optimizer.zero_grad()

            bpr_loss = student.bpr_loss(batch_u, batch_pos, batch_neg, decay=l2_reg)

            batch_obj = InteractionBatch(users=batch_u, pos_items=batch_pos, neg_items=batch_neg)
            aux_batch = distiller.build_aux_batch(batch_obj, device=device) if hasattr(distiller, "build_aux_batch") else None

            if distiller_type == "contrastive":
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
                l_dist = loss_dict.get("l_item", 0.0) + loss_dict.get("l_user", 0.0)
            else:
                l_dist_tensor = distiller.compute_loss(student, batch_obj, aux_batch)
                loss = bpr_loss + l_dist_tensor
                l_dist = l_dist_tensor.item()

            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item()
            bpr_loss_epoch += bpr_loss.item()
            dist_loss_epoch += l_dist

        elapsed = time.time() - start_time
        num_batches = len(train_loader)
        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - Total Loss: {total_loss_epoch / num_batches:.4f} (BPR: {bpr_loss_epoch / num_batches:.4f}, Distill: {dist_loss_epoch / num_batches:.4f})")

        # Evaluation & Early Stopping
        if epoch % eval_step == 0:
            val_metrics = evaluator.evaluate(student, data_loader, data_loader.val_pos_items, device)
            val_ndcg = val_metrics.get("NDCG@20", 0.0)
            val_recall = val_metrics.get("Recall@20", 0.0)
            print(f" Validation Metrics -> Recall@20: {val_recall:.4f}, NDCG@20: {val_ndcg:.4f}, HR@20: {val_metrics.get('HR@20', 0.0):.4f}")

            if val_ndcg > best_ndcg:
                best_ndcg = val_ndcg
                patience_counter = 0
                best_student_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
                best_distiller_state = {k: v.cpu().clone() for k, v in distiller.state_dict().items()} if hasattr(distiller, "state_dict") else None
            else:
                patience_counter += eval_step
                if patience_counter >= patience:
                    print(f"[Pipeline 2] Early stopping triggered at epoch {epoch} (best val NDCG@20: {best_ndcg:.4f})")
                    break

        try:
            torch.save({
                "epoch": epoch,
                "student_state_dict": student.state_dict(),
                "distiller_state_dict": distiller.state_dict() if hasattr(distiller, "state_dict") else None,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_ndcg": best_ndcg,
                "patience_counter": patience_counter
            }, temp_ckpt_path)
        except Exception:
            pass

    if best_student_state is not None:
        student.load_state_dict(best_student_state)

    # Final Test Evaluation
    print(f"\n--- Final Test Evaluation (Pipeline 2: {distiller_type.upper()} Distillation) ---")
    test_metrics = evaluator.evaluate(student, data_loader, data_loader.test_pos_items, device)
    for k, v in test_metrics.items():
        print(f" {k}: {v:.4f}")

    # Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = out_dir / f"best_student_lightgcn_{distiller_type}.pt"
    torch.save(student.state_dict(), model_save_path)
    if best_distiller_state is not None:
        torch.save(best_distiller_state, out_dir / "best_distiller.pt")

    if temp_ckpt_path.exists():
        try:
            temp_ckpt_path.unlink()
        except Exception:
            pass

    report_info = {
        "pipeline": f"embedding_distillation_{distiller_type}",
        "distiller": distiller_type,
        "dataset": dataset_name,
        "teacher": teacher_name,
        "lr": lr,
        "weight_decay": l2_reg,
        "metrics": {k: float(v) for k, v in test_metrics.items()}
    }

    report_file = out_dir / f"{distiller_type}_run_results.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_info, f, indent=4)
    print(f"[Pipeline 2] Saved results summary to: {report_file}")
    print(f"[Pipeline 2] Saved model weights to: {model_save_path}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline 2: Teacher + LightGCN Embedding Distillation Training (Contrastive, DE, HTD)")
    parser.add_argument("--config", type=str, default="config/train_student_contrastive_distill.yaml", help="Path to contrastive train config YAML")
    parser.add_argument("--dataset", type=str, default="movielens", help="Dataset name ('movielens', 'amazon_beauty', etc.)")
    parser.add_argument("--teacher", type=str, default="Qwen3.5-4B", help="Teacher model name ('Qwen3.5-4B', 'Llama-3-8B-Instruct', etc.)")
    parser.add_argument("--distiller", type=str, default="contrastive", choices=["contrastive", "de", "htd"], help="Distillation method to use: 'contrastive' (default), 'de', 'htd'")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--grid", action="store_true", default=False, help="Enable Distillation Grid Search over lambda values")
    parser.add_argument("--lambda_item", nargs="+", type=float, default=None, help="Lambda item values to test in contrastive grid search")
    parser.add_argument("--lambda_user", nargs="+", type=float, default=None, help="Lambda user values to test in contrastive grid search")
    parser.add_argument("--tau", nargs="+", type=float, default=None, help="Tau temperature values to test in contrastive grid search")
    parser.add_argument("--lambda_de", nargs="+", type=float, default=None, help="Distillation weight(s) for DE (grid default: [0.0001, 0.001, 0.01, 0.1])")
    parser.add_argument("--lambda_td", nargs="+", type=float, default=None, help="Distillation weight(s) for HTD (grid default: [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0])")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    train_contrastive_pipeline(args)


if __name__ == "__main__":
    main()
