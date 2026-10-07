import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

import torch

# Ensure root workspace directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from knowlight.graphs.multihop_subgraph_builder import MultiHopSubgraphBuilder
from knowlight.prompts.prompt_templates import PromptTemplates
from knowlight.paths import AMAZONBEAUTY, AMAZONCD, AMAZONTOYSANDGAMES, MOVIELENS, YELP, GRAPHS_DIR, PROMPTS_DIR

SUPPORTED_DATASETS = [AMAZONBEAUTY, AMAZONCD, AMAZONTOYSANDGAMES, MOVIELENS, YELP]

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class PromptBuilder:
    """
    Builds structured multi-hop graph Ranking Prompts for LLM Instruction Tuning.
    """

    def __init__(self, pyg_graph: Any):
        self.graph = pyg_graph
        
        # Always use train_edge_mask as context to prevent any data leakage
        edge_mask = None
        if hasattr(pyg_graph, "train_edge_mask") and pyg_graph.train_edge_mask is not None:
            edge_mask = pyg_graph.train_edge_mask

        self.subgraph_builder = MultiHopSubgraphBuilder(pyg_graph, edge_mask=edge_mask)
        self.user_mapping = getattr(pyg_graph, "user_mapping", {})
        self.item_mapping = getattr(pyg_graph, "item_mapping", {})
        self.user_text_dict = getattr(pyg_graph, "user_text_dict", {})
        self.item_text_dict = getattr(pyg_graph, "item_text_dict", {})
        
        # Build list of items that are present in the training set
        self.train_item_keys = []
        for i_key, i_idx in self.item_mapping.items():
            if i_idx in self.subgraph_builder.adj_list and len(self.subgraph_builder.adj_list[i_idx]) > 0:
                self.train_item_keys.append(i_key)
                
        if not self.train_item_keys:
            self.train_item_keys = list(self.item_mapping.keys())

    def build_prompt(
        self,
        user_raw_id: str,
        candidate_item_id: str,
        max_hops: int = 2,
        max_neighbors: int = 10,
        exclude_item_raw_id: Optional[str] = None
    ) -> str:
        """
        Builds the context Prompt for a target user and a single candidate item via BFS.
        """
        ctx = self.subgraph_builder.get_user_subgraph_context_by_raw_id(
            str(user_raw_id),
            max_hops=max_hops,
            max_neighbors_per_hop=max_neighbors,
            exclude_item_raw_id=exclude_item_raw_id
        )

        n1_list = ctx["hop_attributes"].get(1, [])
        n2_list = ctx["hop_attributes"].get(2, [])

        n1_str = "; ".join(n1_list[:max_neighbors]) if n1_list else "None"
        n2_str = "; ".join(n2_list[:max_neighbors]) if n2_list else "None"

        # Format candidate item with metadata
        desc = self.item_text_dict.get(str(candidate_item_id), f"Item {candidate_item_id}")
        candidate_str = f"[{candidate_item_id}: {desc}]"

        return PromptTemplates.build_prompt(
            user_id=str(user_raw_id),
            neighbors_1=n1_str,
            neighbors_2=n2_str,
            candidate_item=candidate_str
        )

    def build_user_intent_prompt(
        self,
        user_raw_id: str,
        max_hops: int = 2,
        max_neighbors: int = 10
    ) -> str:
        """
        Builds a pure User Context & 2-Hop Graph Neighborhood prompt (without candidate item) for User Intent CoT reasoning.
        """
        ctx = self.subgraph_builder.get_user_subgraph_context_by_raw_id(
            str(user_raw_id),
            max_hops=max_hops,
            max_neighbors_per_hop=max_neighbors
        )

        n1_list = ctx["hop_attributes"].get(1, [])
        n2_list = ctx["hop_attributes"].get(2, [])

        n1_str = "; ".join(n1_list[:max_neighbors]) if n1_list else "None"
        n2_str = "; ".join(n2_list[:max_neighbors]) if n2_list else "None"

        return (
            f"User ID: {user_raw_id}\n"
            f"Direct Interacted Items (1-Hop): {n1_str}\n"
            f"Graph Neighborhood & Similar UsersInteractions (2-Hop): {n2_str}\n"
        )

def generate_dataset_prompts(
    dataset_name: str,
    max_hops: int = 2,
    max_neighbors: int = 10,
    num_candidates: int = 5,
    graph_dir: Path = GRAPHS_DIR,
    output_dir: Path = PROMPTS_DIR
) -> Path:
    """
    Generates and saves the .jsonl Prompt dataset for LLM Instruction Tuning.
    """
    graph_path = graph_dir / f"{dataset_name}_tag_graph.pt"
    if not graph_path.exists():
        raise FileNotFoundError(f"PyG graph not found at {graph_path}. Run `python graphs/generate_tag_graphs.py --dataset {dataset_name}` first.")

    logging.info(f"Loading PyG TAG graph for '{dataset_name}' from {graph_path}...")
    pyg_graph = torch.load(graph_path, weights_only=False)
    object.__setattr__(pyg_graph, "_store", pyg_graph.__dict__)

    builder = PromptBuilder(pyg_graph)
    user_keys = list(builder.user_mapping.keys())
    item_keys = list(builder.item_mapping.keys())

    inv_item_mapping = {idx: raw_id for raw_id, idx in builder.item_mapping.items()}

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{dataset_name}_train_prompts.jsonl"

    # Load ground-truth positive items (train.tsv) if available
    split_pos_map = {}
    from knowlight.paths import RAW_DATA_DIR
    split_file = RAW_DATA_DIR / dataset_name / "train.tsv"
    if split_file.exists():
        logging.info(f"Loading ground-truth target items for training from {split_file}...")
        import pandas as pd
        df = pd.read_csv(split_file, sep="\t", header=None)
        for _, row in df.iterrows():
                split_pos_map[str(row[0])] = str(row[1])

    logging.info(f"Building Prompts for {len(user_keys)} users in '{dataset_name}'...")

    prompt_count = 0
    import random
    with open(out_file, "w", encoding="utf-8") as f:
        for u_idx, u_id in enumerate(user_keys):
            u_node = builder.user_mapping[u_id]
            # Context neighbors (masked to avoid leakage, now always train_edge_mask)
            neighbor_nodes = builder.subgraph_builder.adj_list.get(u_node, [])
            user_interacted_items = [inv_item_mapping[n] for n in neighbor_nodes if n in inv_item_mapping]
            user_interacted_set = set(user_interacted_items)

            if str(u_id) in split_pos_map:
                pos_item = split_pos_map[str(u_id)]
            elif user_interacted_items:
                pos_item = user_interacted_items[0]
            else:
                pos_item = builder.train_item_keys[u_idx % len(builder.train_item_keys)]
                user_interacted_set.add(pos_item)

            # Randomly sample negative items
            available_negs = list(set(builder.train_item_keys) - user_interacted_set - {pos_item})
            if len(available_negs) >= (num_candidates - 1):
                neg_items = random.sample(available_negs, num_candidates - 1)
            else:
                neg_items = available_negs

            cand_items = [(pos_item, "1")] + [(neg, "0") for neg in neg_items]

            for cand_item, target_score in cand_items:
                prompt_str = builder.build_prompt(
                    user_raw_id=u_id,
                    candidate_item_id=cand_item,
                    max_hops=max_hops,
                    max_neighbors=max_neighbors,
                    exclude_item_raw_id=pos_item
                )

                sample_record = {
                    "user_id": u_id,
                    "item_id": cand_item,
                    "prompt": prompt_str,
                    "response": target_score
                }

                f.write(json.dumps(sample_record, ensure_ascii=False) + "\n")
                prompt_count += 1

    logging.info(f"Successfully generated and saved {prompt_count} prompts to: {out_file}")
    return out_file


def generate_user_intent_prompts(
    dataset_name: str,
    max_hops: int = 2,
    max_neighbors: int = 10,
    graph_dir: Path = GRAPHS_DIR,
    output_dir: Path = PROMPTS_DIR
) -> Path:
    """
    Generates and saves data/prompts/{dataset_name}_user_intent_prompts.jsonl containing 2-hop graph context for pure User Intent extraction.
    """
    graph_path = graph_dir / f"{dataset_name}_tag_graph.pt"
    if not graph_path.exists():
        raise FileNotFoundError(f"PyG graph not found at {graph_path}. Run `python graphs/generate_tag_graphs.py --dataset {dataset_name}` first.")

    logging.info(f"Loading PyG TAG graph for '{dataset_name}' from {graph_path}...")
    pyg_graph = torch.load(graph_path, weights_only=False)
    object.__setattr__(pyg_graph, "_store", pyg_graph.__dict__)

    builder = PromptBuilder(pyg_graph)
    user_keys = list(builder.user_mapping.keys())

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{dataset_name}_user_intent_prompts.jsonl"

    logging.info(f"Building User Intent Prompts for {len(user_keys)} users in '{dataset_name}'...")
    count = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for u_id in user_keys:
            prompt_str = builder.build_user_intent_prompt(
                user_raw_id=u_id,
                max_hops=max_hops,
                max_neighbors=max_neighbors
            )
            sample_record = {
                "user_id": u_id,
                "prompt": prompt_str
            }
            f.write(json.dumps(sample_record, ensure_ascii=False) + "\n")
            count += 1

    logging.info(f"Successfully generated and saved {count} User Intent prompts to: {out_file}")
    return out_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate .jsonl Ranking & User Intent Prompt Datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=SUPPORTED_DATASETS + ["all"],
        help="Dataset name ('amazon_beauty', 'amazon_cd_2014', 'amazon_toys_and_games', 'movielens', 'yelp', or 'all')"
    )
    parser.add_argument("--hops", type=int, default=2, help="Maximum BFS exploration hops (default: 2)")
    parser.add_argument("--max_neighbors", type=int, default=10, help="Maximum neighbors per hop (default: 10)")
    parser.add_argument("--num_candidates", type=int, default=5, help="Number of candidate items in ranking prompt (default: 5)")
    parser.add_argument("--user_intent", action="store_true", help="Generate User Intent Prompts without candidate items")

    args = parser.parse_args()

    target_datasets = SUPPORTED_DATASETS if args.dataset == "all" else [args.dataset]

    print("=" * 70)
    print("  KnowLight Prompt Dataset Generator ")
    print(f"  Target Datasets: {target_datasets}")
    print(f"  Mode           : {'User Intent Prompts' if args.user_intent else 'Ranking Prompts'}")
    print(f"  Max Hops       : {args.hops}, Max Neighbors per Hop: {args.max_neighbors}")
    print("=" * 70)

    for ds in target_datasets:
        print(f"\n Processing dataset: '{ds}'...")
        try:
            if args.user_intent:
                out_p = generate_user_intent_prompts(
                    dataset_name=ds,
                    max_hops=args.hops,
                    max_neighbors=args.max_neighbors
                )
            else:
                out_p = generate_dataset_prompts(
                    dataset_name=ds,
                    max_hops=args.hops,
                    max_neighbors=args.max_neighbors,
                    num_candidates=args.num_candidates
                )
            print(f" Dataset '{ds}' successfully processed -> {out_p}")
        except FileNotFoundError as e:
            print(f" [Skipped] {e}")
        except Exception as e:
            print(f" [Error] Failed to process dataset '{ds}': {e}")

    print("\n" + "=" * 70)
    print(" Finished generating prompts.")
    print("=" * 70)


if __name__ == "__main__":
    main()
