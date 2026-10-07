import os
import time
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader

from knowlight.prompts.prompt_builder import PromptBuilder
from knowlight.teachers.llm.llm_teacher import LLMTeacher
from knowlight.student.lightgcn import LightGCN
from knowlight.distillation import (
    KnowLightDistiller,
    RRDDistiller,
    UnKDDistiller,
)
from knowlight.samplers import RRDSampler, TeacherTopKProvider
from knowlight.teachers.state import TeacherState
from knowlight.datasets.batch import InteractionBatch
from knowlight.datasets.interactions import InteractionDataset
from knowlight.training.evaluator import Evaluator
from knowlight.datasets import BPRDataset


class Trainer:
    """
    Main Trainer for KnowLight Framework.
    Manages Teacher feature caching, LightGCN student training, Multi-level Distillation, and Evaluation.
    """

    def __init__(
        self,
        train_config: dict,
        llm_config: dict,
        data_loader,
        device: torch.device,
        teacher_name: str = "Qwen3.5-4B",
        preloaded_scores: Optional[Dict[int, List[Tuple[int, float]]]] = None,
        prompt_builder: Optional[PromptBuilder] = None
    ):
        self.train_config = train_config
        self.llm_config = llm_config
        self.data_loader = data_loader
        self.device = torch.device(device)
        self.teacher_name = teacher_name

        self.num_users = data_loader.num_users
        self.num_items = data_loader.num_items

        # Check if any distillation loss is active
        dist_cfg = train_config.get("distillation", {})
        self.is_distillation_active = dist_cfg.get("active", False)
        self.distiller_type = str(dist_cfg.get("type", "score")).lower()

        if self.is_distillation_active:
            if prompt_builder is not None:
                self.prompt_builder = prompt_builder
                self.pyg_graph = getattr(prompt_builder, "graph", None)
            else:
                dataset_name = train_config.get("dataset", {}).get("name", "movielens")
                from knowlight.paths import GRAPHS_DIR
                graph_path = GRAPHS_DIR / f"{dataset_name}_tag_graph.pt"
                if not graph_path.exists():
                    graph_path = Path("graphs") / "output" / f"{dataset_name}_tag_graph.pt"

                if graph_path.exists():
                    self.pyg_graph = torch.load(graph_path, weights_only=False)
                    object.__setattr__(self.pyg_graph, "_store", self.pyg_graph.__dict__)
                    self.prompt_builder = PromptBuilder(self.pyg_graph)
                else:
                    self.prompt_builder = None
            self.llm_teacher = None
        else:
            self.prompt_builder = None
            self.llm_teacher = None

        # Student Model (LightGCN)
        student_cfg = train_config.get("student", {})
        embedding_dim = student_cfg.get("embedding_dim", 64)
        self.student = LightGCN(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=embedding_dim,
            n_layers=student_cfg.get("n_layers", 3),
            keep_prob=student_cfg.get("keep_prob", 0.6)
        ).to(device)
        self.student.set_graph(data_loader.train_matrix, device=device)

        # Distiller Initialization
        if self.is_distillation_active:
            if self.distiller_type == "rrd":
                lambda_rrd = float(dist_cfg.get("lambda_rrd", 0.1))
                int_sz = int(dist_cfg.get("interesting_size", 40))
                unint_sz = int(dist_cfg.get("uninteresting_size", 50))
                temp_rrd = float(dist_cfg.get("temperature_rrd", 1.0))
                top_k_req = int(dist_cfg.get("teacher_topk", 500))
                top_k_prov = min(top_k_req, max(50, self.num_items // 4))
                rrd_sampler = RRDSampler(
                    interesting_size=int_sz,
                    uninteresting_size=unint_sz,
                    temperature=temp_rrd,
                    topk_provider=TeacherTopKProvider(top_k=top_k_prov)
                )
                self.distiller = RRDDistiller(sampler=rrd_sampler, lambda_rrd=lambda_rrd).to(device)
                print(f"[Trainer] Distiller RRD active (lambda={lambda_rrd}, int={int_sz}, unint={unint_sz}, temp={temp_rrd}, top_k={top_k_prov})")

            elif self.distiller_type == "unkd":
                lambda_unkd = float(dist_cfg.get("lambda_unkd", 0.1))
                sample_num = int(dist_cfg.get("sample_num", 30))
                group_count = int(dist_cfg.get("group_count", 4))
                pop_lambda = float(dist_cfg.get("popularity_lambda", 1.0))
                rank_top_k = int(dist_cfg.get("rank_top_k", 1000))
                rank_temp = float(dist_cfg.get("rank_temperature", 20.0))
                self.distiller = UnKDDistiller(
                    lambda_unkd=lambda_unkd,
                    sample_num=sample_num,
                    group_count=group_count,
                    popularity_lambda=pop_lambda,
                    rank_top_k=rank_top_k,
                    rank_temperature=rank_temp
                ).to(device)
                print(f"[Trainer] Distiller UnKD active (lambda={lambda_unkd}, samples={sample_num}, groups={group_count})")

            else:
                self.distiller_type = "score"
                self.distiller = KnowLightDistiller(
                    alpha=float(dist_cfg.get("alpha", 1.0)),
                    beta=float(dist_cfg.get("beta", 1.0))
                ).to(device)
                print(f"[Trainer] Distiller Score/KnowLight active (alpha={dist_cfg.get('alpha', 1.0)}, beta={dist_cfg.get('beta', 1.0)})")
        else:
            self.distiller = None

        # Optimizer
        lr = float(train_config.get("training", {}).get("lr", 0.001))
        self.l2_reg = float(train_config.get("training", {}).get("weight_decay", 0.0001))
        ds_name = train_config.get("dataset", {}).get("name", "unknown")
        print(f"[Trainer] Initializing optimizer for dataset '{ds_name}': Adam(lr={lr}), Weight Decay (L2 reg)={self.l2_reg}")

        if self.is_distillation_active and self.distiller is not None:
            dist_params = list(self.distiller.parameters()) if hasattr(self.distiller, "parameters") else []
            params = list(self.student.parameters()) + dist_params
        else:
            params = list(self.student.parameters())

        self.optimizer = torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=0.0
        )

        self.evaluator = Evaluator(top_k_list=train_config.get("evaluation", {}).get("top_k", [5, 10, 20]))

        # Cached teacher knowledge
        self.cached_user_semantic_embs = None
        self.cached_pseudo_edges = None
        self.cached_user_evaluated_items = preloaded_scores or {}

    def preextract_teacher_knowledge(self):
        """Pre-extract and cache LLM semantic probability scores."""
        if not self.is_distillation_active or len(self.cached_user_evaluated_items) > 0:
            return

        dataset_name = self.train_config.get("dataset", {}).get("name", "movielens")
        from knowlight.paths import CACHE_DIR
        
        teacher_dir = CACHE_DIR / self.teacher_name / dataset_name
        teacher_norm_dir = CACHE_DIR / self.teacher_name.lower().replace("-", "_").replace(".", "_") / dataset_name
        
        possible_paths = [
            teacher_dir / "scores" / f"{dataset_name}_llm_scores_cache.pt",
            teacher_norm_dir / "scores" / f"{dataset_name}_llm_scores_cache.pt",
            teacher_dir / f"{dataset_name}_llm_scores_cache.pt",
            teacher_norm_dir / f"{dataset_name}_llm_scores_cache.pt",
            teacher_dir / "scores" / f"{dataset_name}_scores.pt",
            teacher_norm_dir / "scores" / f"{dataset_name}_scores.pt",
        ]

        raw_cache_path = None
        for p in possible_paths:
            if p.exists():
                raw_cache_path = p
                break

        if raw_cache_path is None or not raw_cache_path.exists():
            print(f"[Trainer] No teacher score cache found for '{self.teacher_name}' ({dataset_name}).")
            print(f"[Trainer] Triggering on-the-fly score extraction using extract_scores...")
            try:
                from knowlight.teachers.llm.extract_scores import extract_dataset_llm_scores
                raw_cache_path = extract_dataset_llm_scores(
                    dataset_name=dataset_name,
                    teacher_name=self.teacher_name,
                    cache_dir=CACHE_DIR
                )
            except Exception as e:
                print(f"[Trainer] Warning: On-the-fly score extraction failed ({e}). Distillation will fall back to BPR loss.")

        self.cached_user_evaluated_items = {}

        if raw_cache_path is not None and raw_cache_path.exists():
            print(f"[Trainer] Loading pre-extracted teacher interaction scores from cache: {raw_cache_path}")
            if raw_cache_path.suffix == ".pkl":
                import pickle
                with open(raw_cache_path, "rb") as f:
                    raw_dict = pickle.load(f)
            else:
                raw_dict = torch.load(raw_cache_path, map_location=self.device, weights_only=False)
            
            # Ensure prompt builder / mappings are loaded
            if self.prompt_builder is None:
                from knowlight.paths import GRAPHS_DIR
                graph_path = GRAPHS_DIR / f"{dataset_name}_tag_graph.pt"
                if graph_path.exists():
                    self.pyg_graph = torch.load(graph_path, weights_only=False)
                    object.__setattr__(self.pyg_graph, "_store", self.pyg_graph.__dict__)
                    self.prompt_builder = PromptBuilder(self.pyg_graph)

            # Map string IDs to integer indices
            if self.prompt_builder is not None and len(raw_dict) > 0:
                user_mapping = self.prompt_builder.user_mapping
                item_mapping = self.prompt_builder.item_mapping
                
                loaded_count = 0
                for u_str, items_dict in raw_dict.items():
                    u_idx = user_mapping.get(str(u_str))
                    if u_idx is not None and u_idx < self.num_users:
                        self.cached_user_evaluated_items[u_idx] = []
                        for i_str, score in items_dict.items():
                            i_idx = item_mapping.get(str(i_str))
                            if i_idx is not None:
                                # Convert PyG bipartite node index (num_users + item_idx) to student item index
                                if i_idx >= self.num_users:
                                    i_idx -= self.num_users
                                if 0 <= i_idx < self.num_items:
                                    self.cached_user_evaluated_items[u_idx].append((i_idx, float(score)))
                        loaded_count += 1
                        
                print(f"[Trainer] Successfully aligned teacher scores for {loaded_count}/{self.num_users} users.")
            else:
                print("[Trainer] Warning: PromptBuilder not initialized. Cannot align scores.")
        else:
            print(f"[Trainer] Warning: No teacher score cache available for '{self.teacher_name}' ({dataset_name}). Distillation will only use BPR loss.")

        print("[Trainer] Teacher feature preparation complete.")

    def train(self, out_dir: Optional[Path] = None):
        """Main training loop."""
        if self.is_distillation_active:
            self.preextract_teacher_knowledge()

        dataset_obj = InteractionDataset.from_train_dict(self.data_loader.train_dict, self.num_users, self.num_items)
        if self.is_distillation_active and self.distiller is not None:
            # Build teacher_topk dict from cached_user_evaluated_items if present for RRD
            if self.distiller_type == "rrd" and len(self.cached_user_evaluated_items) > 0:
                topk_by_user = {}
                for u in range(self.num_users):
                    items_list = self.cached_user_evaluated_items.get(u, [])
                    if items_list:
                        sorted_items = [idx for idx, _ in sorted(items_list, key=lambda x: x[1], reverse=True)]
                        topk_by_user[u] = sorted_items
                if topk_by_user and hasattr(self.distiller, "sampler"):
                    self.distiller.sampler.teacher_topk = topk_by_user

            if self.distiller_type == "unkd" and len(self.cached_user_evaluated_items) > 0:
                user_score_dict = {}
                for u in range(self.num_users):
                    items_list = self.cached_user_evaluated_items.get(u, [])
                    if items_list:
                        user_score_dict[u] = {int(i): float(s) for i, s in items_list}
                if user_score_dict:
                    self.distiller.teacher_scores = user_score_dict

            teacher_state = TeacherState(num_users=self.num_users, num_items=self.num_items)
            if hasattr(self.distiller, "on_train_start"):
                self.distiller.on_train_start(teacher_state=teacher_state, dataset=dataset_obj)

        bpr_ds = BPRDataset(self.num_users, self.num_items, train_dict=self.data_loader.train_dict, train_matrix=self.data_loader.train_matrix)
        train_loader = DataLoader(
            bpr_ds,
            batch_size=self.train_config.get("training", {}).get("batch_size", 1024),
            shuffle=True
        )

        epochs = self.train_config.get("training", {}).get("epochs", 20)
        eval_step = self.train_config.get("training", {}).get("eval_step", 1)
        best_ndcg = 0.0
        patience_counter = 0
        best_student_state = None
        start_epoch = 1

        temp_ckpt_path = (out_dir / "temp_checkpoint.pt") if out_dir else None
        if temp_ckpt_path and temp_ckpt_path.exists():
            try:
                ckpt = torch.load(temp_ckpt_path, map_location=self.device, weights_only=False)
                start_epoch = ckpt.get("epoch", 0) + 1
                self.student.load_state_dict(ckpt["student_state_dict"])
                if self.is_distillation_active and "distiller_state_dict" in ckpt and ckpt["distiller_state_dict"] and self.distiller is not None and hasattr(self.distiller, "load_state_dict"):
                    self.distiller.load_state_dict(ckpt["distiller_state_dict"])
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                best_ndcg = ckpt.get("best_ndcg", 0.0)
                patience_counter = ckpt.get("patience_counter", 0)
                print(f"[Trainer] Resuming training from epoch {start_epoch}/{epochs} (best val NDCG@20: {best_ndcg:.4f}).")
            except Exception as e:
                print(f"[Trainer] Could not load epoch temp checkpoint ({e}). Starting fresh from epoch 1.")
                start_epoch = 1

        for epoch in range(start_epoch, epochs + 1):
            self.student.train()
            if self.is_distillation_active and self.distiller is not None:
                if hasattr(self.distiller, "train"):
                    self.distiller.train()
                if hasattr(self.distiller, "on_epoch_start"):
                    self.distiller.on_epoch_start()

            total_loss_epoch = 0.0
            bpr_loss_epoch = 0.0
            ld_loss_epoch = 0.0
            start_time = time.time()

            for batch_u, batch_pos, batch_neg in train_loader:
                batch_u = batch_u.to(self.device)
                batch_pos = batch_pos.to(self.device)
                batch_neg = batch_neg.to(self.device)

                self.optimizer.zero_grad()

                # 1. Base Student BPR Loss
                bpr_loss = self.student.bpr_loss(batch_u, batch_pos, batch_neg, decay=self.l2_reg)

                batch_obj = InteractionBatch(users=batch_u, pos_items=batch_pos, neg_items=batch_neg)
                aux_batch = self.distiller.build_aux_batch(batch_obj, device=self.device) if (self.is_distillation_active and self.distiller is not None and hasattr(self.distiller, "build_aux_batch")) else None

                # Distillation logic
                if self.is_distillation_active and self.distiller is not None:
                    if self.distiller_type in ["score", "knowlight"]:
                        if len(self.cached_user_evaluated_items) > 0:
                            distill_u = []
                            distill_i = []
                            distill_llm_scores = []
                            for u in batch_u.tolist():
                                if u in self.cached_user_evaluated_items:
                                    for i_idx, score in self.cached_user_evaluated_items[u]:
                                        if i_idx < self.num_items:
                                            distill_u.append(u)
                                            distill_i.append(i_idx)
                                            distill_llm_scores.append(score)

                            if len(distill_u) > 0:
                                dU_tensor = torch.tensor(distill_u, device=self.device, dtype=torch.long)
                                dI_tensor = torch.tensor(distill_i, device=self.device, dtype=torch.long)
                                llm_scores_tensor = torch.tensor(distill_llm_scores, device=self.device, dtype=torch.float32)

                                eu = self.student.get_user_embedding(dU_tensor)
                                ei = self.student.get_item_embedding(dI_tensor)
                                gcn_scores = torch.sigmoid(torch.sum(eu * ei, dim=1))

                                loss_dict = self.distiller(
                                    bpr_loss=bpr_loss,
                                    llm_scores=llm_scores_tensor,
                                    gcn_scores=gcn_scores
                                )
                                loss = loss_dict["total_loss"]
                            else:
                                loss = bpr_loss
                                loss_dict = {"l_bpr": bpr_loss.item(), "l_d": 0.0, "total_loss": loss}
                        else:
                            loss = bpr_loss
                            loss_dict = {"l_bpr": bpr_loss.item(), "l_d": 0.0, "total_loss": loss}
                    else:
                        # RRD, UnKD, DE, HTD
                        l_dist_tensor = self.distiller.compute_loss(self.student, batch_obj, aux_batch)
                        loss = bpr_loss + l_dist_tensor
                        loss_dict = {"l_bpr": bpr_loss.item(), "l_d": l_dist_tensor.item(), "total_loss": loss}
                else:
                    loss = bpr_loss
                    loss_dict = {"l_bpr": bpr_loss.item(), "l_d": 0.0, "total_loss": loss}


                # ==========================================
                # DEBUG: Gradient Norm Analysis
                # First epoch, first batch only
                # ==========================================
                if epoch == 1 and total_loss_epoch == 0 and self.is_distillation_active:

                    l_d = loss_dict["total_loss"] - bpr_loss

                    student_params = [
                        p for p in self.student.parameters()
                        if p.requires_grad
                    ]

                    bpr_grads = torch.autograd.grad(
                        bpr_loss,
                        student_params,
                        retain_graph=True,
                        allow_unused=True
                    )

                    dist_grads = torch.autograd.grad(
                        l_d,
                        student_params,
                        retain_graph=True,
                        allow_unused=True
                    )

                    bpr_norm = 0.0
                    for g in bpr_grads:
                        if g is not None:
                            bpr_norm += g.norm(2).item() ** 2
                    bpr_norm = bpr_norm ** 0.5

                    dist_norm = 0.0
                    for g in dist_grads:
                        if g is not None:
                            dist_norm += g.norm(2).item() ** 2
                    dist_norm = dist_norm ** 0.5

                    print("\n========== GRADIENT ANALYSIS ==========")
                    print(f"BPR Gradient Norm: {bpr_norm:.6f}")
                    print(f"Distillation Gradient Norm: {dist_norm:.6f}")

                    if bpr_norm > 0:
                        print(f"Ratio Dist/BPR: {dist_norm / bpr_norm:.2f}")
                    print("=======================================\n")


                loss.backward()
                self.optimizer.step()

                total_loss_epoch += loss.item()
                bpr_loss_epoch += loss_dict.get("l_bpr", 0.0)
                ld_loss_epoch += loss_dict.get("l_d", 0.0)

            elapsed = time.time() - start_time
            num_batches = len(train_loader)
            print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - Total Loss: {total_loss_epoch / num_batches:.4f} (BPR: {bpr_loss_epoch / num_batches:.4f}, LD: {ld_loss_epoch / num_batches:.4f})")

            # Evaluation & Early Stopping
            if epoch % eval_step == 0:
                val_metrics = self.evaluator.evaluate(self.student, self.data_loader, self.data_loader.val_pos_items, self.device)
                val_ndcg = val_metrics.get("NDCG@20", 0.0)
                val_recall = val_metrics.get("Recall@20", 0.0)
                print(f" Validation Metrics -> Recall@20: {val_recall:.4f}, NDCG@20: {val_ndcg:.4f}, HR@20: {val_metrics.get('HR@20', 0.0):.4f}")

                patience = self.train_config.get("training", {}).get("patience", 20)
                if val_ndcg > best_ndcg:
                    best_ndcg = val_ndcg
                    patience_counter = 0
                    best_student_state = {k: v.cpu().clone() for k, v in self.student.state_dict().items()}
                else:
                    patience_counter += eval_step
                if patience_counter >= patience:
                    print(f"[Trainer] Early stopping triggered at epoch {epoch} (best val NDCG@20: {best_ndcg:.4f})")
                    break

            if temp_ckpt_path:
                try:
                    torch.save({
                        "epoch": epoch,
                        "student_state_dict": self.student.state_dict(),
                        "distiller_state_dict": self.distiller.state_dict() if hasattr(self, "distiller") else None,
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_ndcg": best_ndcg,
                        "patience_counter": patience_counter
                    }, temp_ckpt_path)
                except Exception as e:
                    pass

        if best_student_state is not None:
            self.student.load_state_dict(best_student_state)

        # Final Test Evaluation
        print("\n--- Final Test Evaluation ---")
        test_metrics = self.evaluator.evaluate(self.student, self.data_loader, self.data_loader.test_pos_items, self.device)
        for k, v in test_metrics.items():
            print(f" {k}: {v:.4f}")

        if temp_ckpt_path and temp_ckpt_path.exists():
            try:
                temp_ckpt_path.unlink()
            except Exception:
                pass

        return test_metrics
