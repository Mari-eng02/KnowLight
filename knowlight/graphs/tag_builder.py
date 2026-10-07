import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import torch

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from knowlight.paths import dataset_directory, dataset_filepath
from knowlight.datasets.metadata_loader import load_dataset_item_metadata

# Try importing torch_geometric Data, or use a PyG-compatible Data container
try:
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

    class Data:
        """Lightweight PyG Data container fallback when torch_geometric is not installed."""
        def __init__(self, edge_index=None, **kwargs):
            self.edge_index = edge_index
            for k, v in kwargs.items():
                setattr(self, k, v)

        def __repr__(self):
            info = []
            if self.edge_index is not None:
                info.append(f"edge_index={list(self.edge_index.shape)}")
            return f"Data({', '.join(info)})"


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class TAGGraphBuilder:
    """
    Text-Attributed Graph (TAG) Builder for Recommender System Datasets.
    Builds a PyG Bipartite Graph (User-Item) enriched exclusively with raw textual metadata.
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_directory(dataset_name, create_if_not_exists=True)

    def load_interactions(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Reads train, val, test TSV split files."""
        splits = {}
        for split in ["train", "val", "test", "processed"]:
            filepath = dataset_filepath(self.dataset_name, split_name=split, exists=True)
            if filepath.exists():
                df = pd.read_csv(
                    filepath,
                    sep="\t",
                    names=["user_id", "item_id", "rating", "timestamp"],
                    dtype={"user_id": str, "item_id": str},
                    header=None
                )
                splits[split] = df

        if "train" in splits:
            all_df = pd.concat([splits.get(s) for s in ["train", "val", "test"] if s in splits], ignore_index=True)
        elif "processed" in splits:
            all_df = splits["processed"]
        else:
            raise FileNotFoundError(f"No interaction TSV files found in {self.dataset_dir}")

        return all_df, splits

    def load_metadata(self) -> Dict[str, str]:
        """Loads item textual metadata from item_metadata.json."""
        meta = load_dataset_item_metadata(self.dataset_name, self.dataset_dir)
        logging.info(f"Loaded {len(meta)} item metadata entries for dataset '{self.dataset_name}'")
        return meta

    def build_graph(self) -> Data:
        """
        Builds the PyG Text-Attributed Graph (TAG) object with raw text attributes.
        - User nodes: 0 ... N_u - 1
        - Item nodes: N_u ... N_u + N_i - 1
        - edge_index: Bipartite User <-> Item connections [2, 2 * num_edges]
        - raw_text / node_text: List of raw textual strings for every node in the graph
        - item_text_dict: Map of raw item_id -> raw textual metadata
        - user_text_dict: Map of raw user_id -> raw user text description
        """
        logging.info(f"Building PyG TAG Graph (Raw Text) for '{self.dataset_name}'...")
        all_df, splits = self.load_interactions()
        item_metadata = self.load_metadata()

        unique_users = sorted(all_df["user_id"].unique())
        unique_items = sorted(all_df["item_id"].unique())

        num_users = len(unique_users)
        num_items = len(unique_items)
        num_nodes = num_users + num_items

        user_mapping = {uid: idx for idx, uid in enumerate(unique_users)}
        item_mapping = {iid: idx + num_users for idx, iid in enumerate(unique_items)}

        logging.info(f"Graph stats: {num_users} Users, {num_items} Items ({num_nodes} total nodes)")

        # Construct Bipartite edge_index (Undirected: User -> Item and Item -> User)
        src_nodes = [user_mapping[u] for u in all_df["user_id"]]
        dst_nodes = [item_mapping[i] for i in all_df["item_id"]]

        edge_src = torch.tensor(src_nodes + dst_nodes, dtype=torch.long)
        edge_dst = torch.tensor(dst_nodes + src_nodes, dtype=torch.long)
        edge_index = torch.stack([edge_src, edge_dst], dim=0)

        # Construct Split Masks if train/val/test are present
        train_edge_mask = None
        val_edge_mask = None
        test_edge_mask = None

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        if "train" in splits and "val" in splits and "test" in splits:
            num_tr = len(splits["train"])
            num_va = len(splits["val"])
            num_te = len(splits["test"])

            tr_list = [True] * num_tr + [False] * (num_va + num_te)
            va_list = [False] * num_tr + [True] * num_va + [False] * num_te
            te_list = [False] * (num_tr + num_va) + [True] * num_te

            # Duplicate for undirected edges
            train_edge_mask = torch.tensor(tr_list + tr_list, dtype=torch.bool)
            val_edge_mask = torch.tensor(va_list + va_list, dtype=torch.bool)
            test_edge_mask = torch.tensor(te_list + te_list, dtype=torch.bool)

            # Node-level masks
            tr_u_nodes = [user_mapping[u] for u in splits["train"]["user_id"] if u in user_mapping]
            tr_i_nodes = [item_mapping[i] for i in splits["train"]["item_id"] if i in item_mapping]
            train_mask[tr_u_nodes + tr_i_nodes] = True

            va_u_nodes = [user_mapping[u] for u in splits["val"]["user_id"] if u in user_mapping]
            va_i_nodes = [item_mapping[i] for i in splits["val"]["item_id"] if i in item_mapping]
            val_mask[va_u_nodes + va_i_nodes] = True

            te_u_nodes = [user_mapping[u] for u in splits["test"]["user_id"] if u in user_mapping]
            te_i_nodes = [item_mapping[i] for i in splits["test"]["item_id"] if i in item_mapping]
            test_mask[te_u_nodes + te_i_nodes] = True
        else:
            train_mask[:] = True

        # Construct Raw Node Text (TAG textual attribute)
        node_text: List[str] = []
        user_text_dict: Dict[str, str] = {}
        item_text_dict: Dict[str, str] = {}

        # 1. Item nodes raw text (metadata from item_metadata.json)
        missing_meta = 0
        for iid in unique_items:
            text = item_metadata.get(str(iid), item_metadata.get(iid, ""))
            if not text:
                text = f"Item {iid} in dataset {self.dataset_name}"
                missing_meta += 1
            item_text_dict[str(iid)] = text

        if missing_meta > 0:
            logging.info(f"{missing_meta}/{num_items} items had missing metadata text (default text assigned).")

        # Group training interactions by user to build rich user interaction profiles
        user_interactions_df = splits.get("train", all_df)
        user_history_map = user_interactions_df.groupby("user_id")["item_id"].apply(list).to_dict()

        # 2. User nodes raw text (constructed from interaction history)
        max_items_in_profile = 10
        for uid in unique_users:
            history_item_ids = user_history_map.get(uid, [])
            if history_item_ids:
                # Extract titles/text of top interacted items
                history_texts = [item_text_dict.get(str(iid), f"Item {iid}") for iid in history_item_ids[:max_items_in_profile]]
                u_text = f"User {uid} interacted with: {'; '.join(history_texts)}"
            else:
                u_text = f"User {uid} in dataset {self.dataset_name}"

            user_text_dict[str(uid)] = u_text

        # Append in node order: Users 0..N_u-1 then Items N_u..N_u+N_i-1
        for uid in unique_users:
            node_text.append(user_text_dict[str(uid)])
        for iid in unique_items:
            node_text.append(item_text_dict[str(iid)])

        # Create PyG Data object storing raw text attributes
        graph = Data(
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_users=num_users,
            num_items=num_items,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            node_text=node_text,
            raw_text=node_text,
            item_text_dict=item_text_dict,
            user_text_dict=user_text_dict,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            train_edge_mask=train_edge_mask,
            val_edge_mask=val_edge_mask,
            test_edge_mask=test_edge_mask,
            dataset_name=self.dataset_name
        )

        return graph

    def save_graph(self, graph: Data, output_path: Optional[Path] = None) -> Path:
        """Saves PyG Data graph object to file (.pt)."""
        if output_path is None:
            output_dir = root_dir / "graphs" / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{self.dataset_name}_tag_graph.pt"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph, output_path)
        logging.info(f"Successfully saved PyG TAG Graph to: {output_path}")

        # Save summary metadata JSON for quick inspection
        info_file = output_path.with_suffix(".json")
        info_data = {
            "dataset_name": self.dataset_name,
            "num_nodes": int(graph.num_nodes),
            "num_users": int(graph.num_users),
            "num_items": int(graph.num_items),
            "num_edges": int(graph.edge_index.shape[1]),
            "has_raw_text": bool(hasattr(graph, "raw_text") and graph.raw_text),
            "pyg_native": HAS_PYG
        }
        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(info_data, f, indent=2)

        return output_path
