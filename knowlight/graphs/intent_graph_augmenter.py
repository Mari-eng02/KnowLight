import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from knowlight.paths import CACHE_DIR, RAW_DATA_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class IntentGraphAugmenter:
    """
    Constructs an augmented user-item interaction matrix R_tilde = R_train + W_pseudo
    by projecting LLM-generated User Intent descriptions against Item metadata representations.
    
    The augmented graph is used strictly during LightGCN message passing (training)
    to enrich collaborative representations without any data leakage on validation/test sets.
    """

    def __init__(
        self,
        dataset_name: str = "movielens",
        teacher_name: str = "Qwen3.5-4B",
        top_k_pseudo: int = 2,
        pseudo_weight: float = 0.5,
        sim_threshold: float = 0.0,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.dataset_name = dataset_name
        self.teacher_name = teacher_name
        self.top_k_pseudo = top_k_pseudo
        self.pseudo_weight = pseudo_weight
        self.sim_threshold = sim_threshold
        self.device = device

        self.teacher_tag = Path(teacher_name).name.replace("/", "_")
        self.cache_dir = CACHE_DIR / self.teacher_tag / dataset_name
        self.intents_dir = self.cache_dir / "intents"
        self.aug_graph_dir = self._resolve_augmented_graphs_dir()

    def _get_existing_augmented_dirs(self) -> List[Tuple[int, Path]]:
        """Scans cache_dir for all augmented_graphs* subdirectories sorted by index ascending."""
        pattern = re.compile(r"^augmented_graphs(?:_?(\d+))?$")
        existing = []
        if self.cache_dir.exists():
            for d in self.cache_dir.iterdir():
                if d.is_dir():
                    m = pattern.match(d.name)
                    if m:
                        idx = int(m.group(1)) if m.group(1) else 1
                        existing.append((idx, d))
        existing.sort(key=lambda x: (x[0], x[1].name))
        return existing

    def _resolve_augmented_graphs_dir(self, total_expected_trials: int = 25) -> Path:
        """
        Resolves the active target directory for saving new augmented graphs:
        - If the latest existing directory is complete (>= 25 files), allocates the next index.
        - If the latest directory is still in progress, continues using it.
        """
        existing = self._get_existing_augmented_dirs()
        if not existing:
            target = self.cache_dir / "augmented_graphs"
            target.mkdir(parents=True, exist_ok=True)
            return target

        latest_idx, latest_dir = existing[-1]
        num_npz = len(list(latest_dir.glob("*.npz")))
        if num_npz >= total_expected_trials:
            max_idx = max(x[0] for x in existing)
            next_idx = max(max_idx, latest_idx) + 1
            target = self.cache_dir / f"augmented_graphs_{next_idx}"
            target.mkdir(parents=True, exist_ok=True)
            logging.info(f"[Graph Augmenter] Previous directory '{latest_dir.name}' is complete ({num_npz} files). Active output directory: '{target.name}'.")
            return target
        else:
            logging.info(f"[Graph Augmenter] Active output directory: '{latest_dir.name}' ({num_npz}/{total_expected_trials} files).")
            return latest_dir

    def find_cached_graph(self, cache_key: str) -> Optional[Path]:
        """
        Searches for an existing cached graph file strictly within the active augmented_graphs directory.
        Does NOT look into older/previous directories so that the active directory creates and maintains
        its own complete set of augmented graphs without reusing graphs from past runs.
        """
        active_file = self.aug_graph_dir / cache_key
        if active_file.exists():
            return active_file
        return None

    def load_user_intent_embeddings(self, num_users: int) -> torch.Tensor:
        """Loads pre-extracted User Intent text embeddings (shape: [num_users, dim])."""
        npy_path = self.intents_dir / "user_intent_text_emb.npy"
        if not npy_path.exists():
            # Fallback path checks
            alt_path = CACHE_DIR / self.teacher_name / self.dataset_name / "intents" / "user_intent_text_emb.npy"
            if alt_path.exists():
                npy_path = alt_path
            else:
                raise FileNotFoundError(
                    f"User Intent embeddings not found at {npy_path}. "
                    f"Please run knowlight/teachers/llm/extract_user_intents.py first."
                )

        logging.info(f"[Graph Augmenter] Loading User Intent embeddings from: {npy_path}")
        raw_arr = np.load(npy_path)
        
        # Slice or pad to exact num_users
        if raw_arr.shape[0] < num_users:
            pad = np.zeros((num_users - raw_arr.shape[0], raw_arr.shape[1]), dtype=np.float32)
            raw_arr = np.vstack([raw_arr, pad])
        elif raw_arr.shape[0] > num_users:
            raw_arr = raw_arr[:num_users]

        return torch.tensor(raw_arr, dtype=torch.float32)

    def load_or_extract_item_embeddings(self, num_items: int, target_dim: int) -> torch.Tensor:
        """
        Loads pre-extracted item text embeddings. If not found, extracts and caches item embeddings
        using the EXACT same Teacher LLM to ensure perfect semantic space alignment.
        """
        item_emb_path = self.cache_dir / "item_embeddings" / "item_text_emb.npy"
        if item_emb_path.exists():
            arr = np.load(item_emb_path)
            if arr.shape[1] == target_dim:
                logging.info(f"[Graph Augmenter] Loaded matching item embeddings ({arr.shape}) from {item_emb_path}")
                tensor_items = torch.tensor(arr[:num_items], dtype=torch.float32)
                return tensor_items

        # Extract item embeddings directly using the configured Teacher LLM
        item_texts = self._load_item_metadata_texts(num_items)
        logging.info(f"[Graph Augmenter] Extracting native item embeddings for {num_items} items using Teacher '{self.teacher_name}' (dim={target_dim})...")
        try:
            from knowlight.teachers.llm.extract_user_intents import LLMIntentExtractor
            extractor = LLMIntentExtractor(model_name_or_path=self.teacher_name, device=self.device)
            item_embs = extractor.extract_text_embeddings(item_texts, batch_size=32)
            
            save_dir = self.cache_dir / "item_embeddings"
            save_dir.mkdir(parents=True, exist_ok=True)
            np.save(item_emb_path, item_embs)
            logging.info(f"[Graph Augmenter] Extracted and cached native item embeddings ({item_embs.shape}) to: {item_emb_path}")
            return torch.tensor(item_embs[:num_items], dtype=torch.float32)
        except Exception as e:
            logging.warning(f"[Graph Augmenter] Teacher item extraction failed ({e}). Checking legacy embeddings.")

        # Fallback: only if native extraction fails
        llmemb_item_path = CACHE_DIR / "llmemb" / self.dataset_name / "item_emb_llmemb.npy"
        if llmemb_item_path.exists():
            arr = np.load(llmemb_item_path)
            if arr.shape[1] == target_dim:
                logging.info(f"[Graph Augmenter] Loaded fallback matching LLMEmb embeddings {arr.shape}")
                return torch.tensor(arr[:num_items], dtype=torch.float32)
            
            logging.warning(f"[Graph Augmenter] Warning: Fallback projecting cross-model item embeddings ({arr.shape[1]} -> {target_dim}).")
            proj_matrix_path = self.cache_dir / f"item_proj_{arr.shape[1]}_{target_dim}.pt"
            if proj_matrix_path.exists():
                proj = torch.load(proj_matrix_path, weights_only=True)
            else:
                torch.manual_seed(42)
                proj = torch.randn(arr.shape[1], target_dim) / np.sqrt(arr.shape[1])
                torch.save(proj, proj_matrix_path)
            
            projected = torch.matmul(torch.tensor(arr[:num_items], dtype=torch.float32), proj)
            return projected

        torch.manual_seed(42)
        return torch.randn(num_items, target_dim, dtype=torch.float32)

    def _load_item_metadata_texts(self, num_items: int) -> List[str]:
        """Loads item descriptive strings for all catalog items."""
        raw_meta_file = RAW_DATA_DIR / self.dataset_name / "item_metadata.json"
        handled_json = RAW_DATA_DIR / self.dataset_name / "handled" / "item_text.json"
        jsonline_file = RAW_DATA_DIR / self.dataset_name / "handled" / "item_str_llmemb.jsonline"
        
        texts = [f"Item {i}" for i in range(num_items)]

        # 1. Check handled item_text.json (Amazon / Yelp processed mappings)
        if handled_json.exists():
            try:
                with open(handled_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.items():
                    idx = int(k) - 1
                    if 0 <= idx < num_items:
                        texts[idx] = str(v)
                return texts
            except Exception:
                pass

        # 2. Check item_str_llmemb.jsonline
        if jsonline_file.exists():
            try:
                with open(jsonline_file, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line.strip())
                        idx = int(data.get("item_id", 0)) - 1
                        if 0 <= idx < num_items:
                            texts[idx] = data.get("input", f"Item {idx}")
                return texts
            except Exception:
                pass

        # 3. Check raw item_metadata.json (MovieLens / Yelp raw mappings)
        if raw_meta_file.exists():
            try:
                with open(raw_meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                for i in range(num_items):
                    str_i = str(i + 1)
                    if str_i in meta:
                        texts[i] = str(meta[str_i])
                return texts
            except Exception:
                pass

        return texts

    def build_augmented_interaction_matrix(
        self,
        train_interaction_matrix: sp.csr_matrix,
        train_dict: Dict[int, List[int]],
        top_k_pseudo: Optional[int] = None,
        pseudo_weight: Optional[float] = None,
        sim_threshold: Optional[float] = None,
        use_cache: bool = True
    ) -> sp.csr_matrix:
        """
        Builds the augmented interaction matrix R_tilde = R_train + W_pseudo.
        
        Args:
            train_interaction_matrix: Original training interaction CSR matrix R (values = 1.0).
            train_dict: Dict mapping user_id -> list of seen training item_ids.
            top_k_pseudo: Number of top semantic candidate items to attach per user.
            pseudo_weight: Scaling factor alpha for pseudo-edge weights (w_ui = alpha * Sim(u, i)).
            sim_threshold: Minimum cosine similarity threshold to attach a pseudo-edge.
            use_cache: If True, saves and reloads cached augmented sparse matrices.
        """
        k = self.top_k_pseudo if top_k_pseudo is None else int(top_k_pseudo)
        alpha = self.pseudo_weight if pseudo_weight is None else float(pseudo_weight)
        thresh = self.sim_threshold if sim_threshold is None else float(sim_threshold)

        cache_key = f"aug_r_k{k}_w{alpha}_th{thresh}.npz"
        cache_file = self.aug_graph_dir / cache_key

        if use_cache:
            found_file = self.find_cached_graph(cache_key)
            if found_file and found_file.exists():
                try:
                    augmented_r = sp.load_npz(found_file)
                    logging.info(f"[Graph Augmenter] Retrieved cached augmented graph from: {found_file} (nnz={augmented_r.nnz})")
                    return augmented_r
                except Exception as e:
                    logging.warning(f"[Graph Augmenter] Failed loading graph cache ({e}). Rebuilding.")
                    cache_file = self.aug_graph_dir / cache_key

        num_users, num_items = train_interaction_matrix.shape
        logging.info(f"[Graph Augmenter] Constructing Augmented Interaction Matrix for {num_users} users, {num_items} items (k={k}, alpha={alpha}, thresh={thresh})...")

        # 1. Load Embeddings
        user_intent_embs = self.load_user_intent_embeddings(num_users=num_users)  # (U, D)
        item_embs = self.load_or_extract_item_embeddings(num_items=num_items, target_dim=user_intent_embs.shape[1])  # (I, D)

        # 2. Normalize for Cosine Similarity
        user_intent_norm = F.normalize(user_intent_embs, p=2, dim=-1)
        item_norm = F.normalize(item_embs, p=2, dim=-1)

        # 3. Compute Similarity Block by Block to conserve memory
        batch_size = 512
        pseudo_rows: List[int] = []
        pseudo_cols: List[int] = []
        pseudo_data: List[float] = []

        total_pseudo_edges = 0
        for u_start in range(0, num_users, batch_size):
            u_end = min(u_start + batch_size, num_users)
            u_batch = user_intent_norm[u_start:u_end]  # (B, D)
            sim_matrix = torch.matmul(u_batch, item_norm.T)  # (B, I)

            for local_u, global_u in enumerate(range(u_start, u_end)):
                seen = train_dict.get(global_u, [])
                if seen:
                    sim_matrix[local_u, list(seen)] = -1e9  # Mask seen train items

                # Zero out empty intent vectors (if any)
                if user_intent_embs[global_u].abs().sum() == 0:
                    continue

                # Top-K candidate extraction
                top_scores, top_items = torch.topk(sim_matrix[local_u], k=min(k, num_items))
                for score, item in zip(top_scores.tolist(), top_items.tolist()):
                    if score >= thresh and score > -1e8:
                        weight = float(alpha * max(score, 0.01))
                        pseudo_rows.append(global_u)
                        pseudo_cols.append(item)
                        pseudo_data.append(weight)
                        total_pseudo_edges += 1

        logging.info(f"[Graph Augmenter] Injected {total_pseudo_edges} semantic pseudo-edges across {num_users} users (avg {total_pseudo_edges / max(num_users, 1):.2f} edges/user).")

        # 4. Merge Original Train Matrix + Pseudo-edges
        # Convert train matrix to COO
        train_coo = train_interaction_matrix.tocoo()
        all_rows = np.concatenate([train_coo.row, np.array(pseudo_rows, dtype=np.int64)])
        all_cols = np.concatenate([train_coo.col, np.array(pseudo_cols, dtype=np.int64)])
        all_data = np.concatenate([train_coo.data.astype(np.float32), np.array(pseudo_data, dtype=np.float32)])

        augmented_r = sp.csr_matrix(
            (all_data, (all_rows, all_cols)),
            shape=(num_users, num_items),
            dtype=np.float32
        )

        if use_cache:
            try:
                sp.save_npz(cache_file, augmented_r)
                logging.info(f"[Graph Augmenter] Saved augmented graph to cache: {cache_file}")
            except Exception as e:
                logging.warning(f"[Graph Augmenter] Failed saving graph cache ({e}).")

        return augmented_r
