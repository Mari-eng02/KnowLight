import os
import pickle
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np
import torch

from knowlight.paths import dataset_directory, CACHE_DIR
from knowlight.teachers.llmemb.rat_adapter import apply_rat_transformation, RATAdapter

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class LLMEmbTeacher:
    """
    Teacher Loader for Pipeline 2 (LLMEmb Contrastive Alignment).
    Loads pre-extracted/adapted item embeddings e_i^LLMEmb (from cache/llmemb/{dataset}/)
    and computes/caches user barycenter embeddings e_u^LLMEmb.
    """

    def __init__(
        self,
        dataset_name: str,
        embedding_dim: int = 64,
        llmemb_dir: Optional[Path] = None,
        teacher_name: Optional[str] = None
    ):
        self.dataset_name = dataset_name
        self.embedding_dim = embedding_dim
        self.teacher_name = teacher_name
        self.dataset_dir = dataset_directory(dataset_name)

        if llmemb_dir is None:
            teacher_str = (teacher_name or "").lower()
            possible_dirs = []

            # 1. Direct match with exact teacher_name if provided
            if teacher_name:
                possible_dirs.append(CACHE_DIR / teacher_name / dataset_name / "embeddings")
                possible_dirs.append(CACHE_DIR / teacher_name / dataset_name)

            # 2. Specific teacher keyword resolution
            if "qwen" in teacher_str:
                possible_dirs.append(CACHE_DIR / "Qwen3.5-4B" / dataset_name / "embeddings")
                possible_dirs.append(CACHE_DIR / "Qwen3.5-4B" / dataset_name)
                possible_dirs.append(CACHE_DIR / "qwen" / dataset_name / "embeddings")
                possible_dirs.append(CACHE_DIR / "qwen" / dataset_name)
            elif "llama" in teacher_str:
                possible_dirs.append(CACHE_DIR / "llama" / dataset_name / "embeddings")
                possible_dirs.append(CACHE_DIR / "llama" / dataset_name)
            elif "llmemb" in teacher_str:
                possible_dirs.append(CACHE_DIR / "llmemb" / dataset_name / "embeddings")
                possible_dirs.append(CACHE_DIR / "llmemb" / dataset_name)

            # 3. Fallbacks
            possible_dirs.extend([
                CACHE_DIR / "llmemb" / dataset_name,
                CACHE_DIR / "Qwen3.5-4B" / dataset_name / "embeddings",
                CACHE_DIR / "Qwen3.5-4B" / dataset_name,
                self.dataset_dir / "llmemb"
            ])

            selected_dir = None
            for p in possible_dirs:
                if p.exists() and (list(p.glob("*.npy")) or list(p.glob("*.pt")) or list(p.glob("*.pkl"))):
                    selected_dir = p
                    break

            self.llmemb_dir = selected_dir or (CACHE_DIR / (teacher_name or "llmemb") / dataset_name)
        else:
            self.llmemb_dir = Path(llmemb_dir)

        self.intermediate_item_embeddings: Optional[torch.Tensor] = None
        self.item_embeddings: Optional[torch.Tensor] = None
        self.user_embeddings: Optional[torch.Tensor] = None
        self.adapter: Optional[RATAdapter] = None

    def load_item_embeddings(self, num_items: int) -> torch.Tensor:
        """
        Loads intermediate item embeddings \bar{e}_i^LLMEmb from file (.npy, .pt, or .pkl).
        If features are of dimension > embedding_dim (e.g. 1536 or 4096), instantiates trainable RATAdapter MLP.
        """
        pkl_path = self.llmemb_dir / "item_emb_llmemb.pkl"
        pt_path = self.llmemb_dir / "item_emb_llmemb.pt"
        npy_path = self.llmemb_dir / "item_emb_llmemb.npy"
        raw_qwen_path = self.llmemb_dir / "raw_item_emb_qwen.npy"
        qwen_npy_path = self.llmemb_dir / "item_emb_qwen.npy"

        # Check for matching files
        if raw_qwen_path.exists():
            logging.info(f"Found raw Qwen embeddings at {raw_qwen_path}. Applying Level 1 PCA reduction...")
            raw_data = np.load(raw_qwen_path)
            from knowlight.teachers.llmemb.rat_adapter import apply_pca_reduction
            embeds = apply_pca_reduction(raw_data, pca_dim=1536)
        elif qwen_npy_path.exists():
            logging.info(f"Loading Qwen item embeddings from {qwen_npy_path}")
            data = np.load(qwen_npy_path)
            embeds = torch.tensor(data, dtype=torch.float32)
        elif npy_path.exists():
            logging.info(f"Loading LLMEmb item embeddings from {npy_path}")
            data = np.load(npy_path)
            embeds = torch.tensor(data, dtype=torch.float32)
        elif pt_path.exists():
            logging.info(f"Loading LLMEmb item embeddings from {pt_path}")
            embeds = torch.load(pt_path, weights_only=True).float()
        elif pkl_path.exists():
            logging.info(f"Loading LLMEmb item embeddings from {pkl_path}")
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            embeds = torch.tensor(data, dtype=torch.float32)
        else:
            teacher_str = (self.teacher_name or "").lower()
            if "qwen" in teacher_str:
                teacher_tag = "qwen"
            elif "llama" in teacher_str:
                teacher_tag = "llama"
            else:
                teacher_tag = "llmemb"

            # Check for item text in handled/item_str_llmemb.jsonline
            from knowlight.paths import RAW_DATA_DIR
            jsonline_file = RAW_DATA_DIR / self.dataset_name / "handled" / "item_str_llmemb.jsonline"
            if not jsonline_file.exists():
                jsonline_file = RAW_DATA_DIR / self.dataset_name / "item_str_llmemb.jsonline"

            self.llmemb_dir.mkdir(parents=True, exist_ok=True)
            save_item_path = self.llmemb_dir / f"item_emb_{teacher_tag}.npy"

            if jsonline_file.exists():
                logging.info(f"Extracting item embeddings on-the-fly using teacher '{self.teacher_name}' for dataset '{self.dataset_name}'...")
                try:
                    import json
                    from knowlight.teachers.llm.extract_user_intents import LLMIntentExtractor
                    model_path = self.teacher_name or "models/Qwen3.5-4B"
                    extractor = LLMIntentExtractor(model_name_or_path=model_path)
                    items_all = []
                    with open(jsonline_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                items_all.append(json.loads(line))
                    item_texts = [it.get("input") or it.get("text") or it.get("prompt") or "" for it in items_all]
                    embs_np = extractor.extract_text_embeddings(item_texts, batch_size=32)
                    np.save(save_item_path, embs_np)
                    logging.info(f"Saved extracted {self.teacher_name} item embeddings ({embs_np.shape}) to {save_item_path}")
                    embeds = torch.tensor(embs_np, dtype=torch.float32)
                except Exception as e:
                    logging.warning(f"Teacher extraction failed ({e}). Initializing random fallback embeddings...")
                    embeds = torch.randn(num_items, self.embedding_dim) * 0.1
                    np.save(save_item_path, embeds.numpy())
            else:
                # Fallback to any available .npy file in the directory
                npy_files = list(self.llmemb_dir.glob("*.npy"))
                if npy_files:
                    target_file = npy_files[0]
                    logging.info(f"Loading item embeddings from {target_file}")
                    data = np.load(target_file)
                    embeds = torch.tensor(data, dtype=torch.float32)
                else:
                    logging.warning(
                        f"No pre-extracted item embeddings or item text found in {self.llmemb_dir}. "
                        f"Generating initialized embeddings ({num_items}, {self.embedding_dim}) and caching to disk."
                    )
                    embeds = torch.randn(num_items, self.embedding_dim) * 0.1
                    np.save(save_item_path, embeds.numpy())
                    logging.info(f"Saved initialized item embeddings to: {save_item_path}")

        if embeds.shape[0] != num_items:
            logging.warning(
                f"Item count mismatch: file has {embeds.shape[0]}, graph expects {num_items}. "
                f"Adjusting shape..."
            )
            if embeds.shape[0] < num_items:
                padding = torch.zeros(num_items - embeds.shape[0], embeds.shape[1])
                embeds = torch.cat([embeds, padding], dim=0)
            else:
                embeds = embeds[:num_items]

        # Check if intermediate dimension requiring RATAdapter projection
        if embeds.shape[1] > self.embedding_dim:
            in_dim = embeds.shape[1]
            hidden_dim = min(in_dim, 512)
            logging.info(f"Initializing Trainable RATAdapter MLP: in_dim={in_dim} -> hidden_dim={hidden_dim} -> out_dim={self.embedding_dim}")
            self.adapter = RATAdapter(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=self.embedding_dim)
            self.intermediate_item_embeddings = embeds
            self.item_embeddings = self.adapter(embeds)
        else:
            self.intermediate_item_embeddings = None
            self.item_embeddings = embeds
            self.adapter = None

        return self.item_embeddings

    def get_adapted_item_embeddings(self) -> torch.Tensor:
        """
        Returns adapted item embeddings e_i^LLMEmb of shape (num_items, embedding_dim).
        If RATAdapter is present, computes forward pass through trainable MLP.
        """
        if self.adapter is not None and self.intermediate_item_embeddings is not None:
            return self.adapter(self.intermediate_item_embeddings)
        return self.item_embeddings

    def get_adapted_user_embeddings(self) -> torch.Tensor:
        """
        Returns adapted user embeddings e_u^LLMEmb of shape (num_users, embedding_dim).
        If RATAdapter is present, computes forward pass through trainable MLP on intermediate user embeddings.
        """
        if self.adapter is not None and hasattr(self, "intermediate_user_embeddings") and self.intermediate_user_embeddings is not None:
            return self.adapter(self.intermediate_user_embeddings)
        return self.user_embeddings

    def compute_user_barycenter_embeddings(
        self,
        num_users: int,
        train_dict: Dict[int, Any],
        use_intermediate: bool = False
    ) -> torch.Tensor:
        """
        Computes user embeddings e_u^LLMEmb as the vectorized average/barycenter of interacted item embeddings:
        e_u^LLMEmb = 1 / |I(u)| * sum_{i in I(u)} e_i^LLMEmb
        Uses PyTorch sparse matrix multiplication to maintain full autograd gradient tracking through RATAdapter.
        """
        if use_intermediate and self.intermediate_item_embeddings is not None:
            item_embeds = self.intermediate_item_embeddings
        else:
            item_embeds = self.get_adapted_item_embeddings()

        if item_embeds is None:
            raise ValueError("Item embeddings must be loaded before computing user barycenters.")

        device = item_embeds.device
        num_items = item_embeds.shape[0]

        if not hasattr(self, "_cached_sparse_adj") or self._cached_sparse_adj is None or self._cached_sparse_adj.device != device:
            indices = []
            for u, items in train_dict.items():
                for i in items:
                    if u < num_users and i < num_items:
                        indices.append([u, i])

            if not indices:
                return torch.zeros(num_users, item_embeds.shape[1], device=device, dtype=torch.float32)

            indices_tensor = torch.tensor(indices, dtype=torch.long, device=device).t()
            
            user_degrees = torch.zeros(num_users, device=device, dtype=torch.float32)
            for u, items in train_dict.items():
                if u < num_users:
                    user_degrees[u] = max(len(items), 1)

            deg_inv = 1.0 / user_degrees
            values = deg_inv[indices_tensor[0]]

            self._cached_sparse_adj = torch.sparse_coo_tensor(
                indices_tensor, values, size=(num_users, num_items), device=device
            )

        user_embeds = torch.sparse.mm(self._cached_sparse_adj, item_embeds)
        if use_intermediate:
            self.intermediate_user_embeddings = user_embeds
        else:
            self.user_embeddings = user_embeds
        return user_embeds

    def load_user_embeddings(
        self,
        num_users: int,
        train_dict: Dict[int, Any],
        force_recompute: bool = False
    ) -> torch.Tensor:
        """
        Loads precomputed user barycenter embeddings from cache if present,
        otherwise computes them via sparse matrix multiplication and saves to cache.
        """
        teacher_str = (self.teacher_name or "").lower()
        if "recgpt" in teacher_str or (self.llmemb_dir / "item_emb_recgpt.npy").exists():
            teacher_tag = "recgpt"
        elif "qwen" in teacher_str or (self.llmemb_dir / "item_emb_qwen.npy").exists():
            teacher_tag = "qwen"
        else:
            teacher_tag = "llmemb"

        teacher_label = self.teacher_name or teacher_tag.upper()

        # Check teacher-specific file first, fallback to user_emb_llmemb.npy
        user_cache_path = self.llmemb_dir / f"user_emb_{teacher_tag}.npy"
        if not user_cache_path.exists() and (self.llmemb_dir / "user_emb_llmemb.npy").exists():
            user_cache_path = self.llmemb_dir / "user_emb_llmemb.npy"

        expected_dim = self.intermediate_item_embeddings.shape[1] if self.intermediate_item_embeddings is not None else self.embedding_dim

        if not force_recompute and user_cache_path.exists():
            logging.info(f"Loading precomputed {teacher_label} user embeddings from {user_cache_path}")
            data = np.load(user_cache_path)
            embeds = torch.tensor(data, dtype=torch.float32)
            if embeds.shape[0] == num_users and embeds.shape[1] == expected_dim:
                if embeds.shape[1] > self.embedding_dim:
                    self.intermediate_user_embeddings = embeds
                    self.user_embeddings = self.get_adapted_user_embeddings()
                else:
                    self.user_embeddings = embeds
            else:
                logging.warning(

                    f"Cached user embeddings shape mismatch: cached {tuple(embeds.shape)} vs expected ({num_users}, {expected_dim}). "
                    f"Recomputing user barycenter embeddings..."
                )
                embeds = self.compute_user_barycenter_embeddings(num_users, train_dict, use_intermediate=True)
                np.save(user_cache_path, embeds.detach().cpu().numpy())
                logging.info(f"Saved updated {teacher_label} user barycenter embeddings of shape {embeds.shape} to {user_cache_path}")
                if embeds.shape[1] > self.embedding_dim:
                    self.intermediate_user_embeddings = embeds
                    self.user_embeddings = self.get_adapted_user_embeddings()
                else:
                    self.user_embeddings = embeds
        else:
            logging.info(f"Precomputing and caching {teacher_label} user barycenter embeddings...")
            embeds = self.compute_user_barycenter_embeddings(num_users, train_dict, use_intermediate=True)
            save_path = self.llmemb_dir / f"user_emb_{teacher_tag}.npy"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, embeds.detach().cpu().numpy())
            logging.info(f"Saved {teacher_label} user barycenter embeddings of shape {embeds.shape} to {save_path}")
            if embeds.shape[1] > self.embedding_dim:
                self.intermediate_user_embeddings = embeds
                self.user_embeddings = self.get_adapted_user_embeddings()
            else:
                self.user_embeddings = embeds

        return self.user_embeddings


