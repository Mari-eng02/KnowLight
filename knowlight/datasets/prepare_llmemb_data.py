import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
from tqdm import tqdm

# Ensure root workspace directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from knowlight.paths import dataset_directory, RAW_DATA_DIR

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def convert_knowlight_to_llmemb_format(dataset_name: str) -> Path:
    """
    Converts KnowLight preprocessed TSV datasets and item_metadata.json into 
    the exact format expected by LLMEmb (id_map.json, inter_seq.txt, inter.txt, item_str_llmemb.jsonline).
    
    Inputs (from data/raw/{dataset_name}):
    - train.tsv, val.tsv, test.tsv (or dataset.tsv)
    - item_metadata.json
    
    Outputs (to data/raw/{dataset_name}/handled/ and LLMEmb-master/data/{dataset_name}/handled/):
    - id_map.json: User and Item 1-indexed ID mappings
    - inter_seq.txt: User interaction sequence per line (user_id item1 item2 ...)
    - inter.txt: Pairwise user item interactions (user_id item_id)
    - item_str_llmemb.jsonline: JSONL prompts for LLM text embedding extraction
    """
    raw_dir = RAW_DATA_DIR / dataset_name
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found at: {raw_dir}")

    output_dir = raw_dir / "handled"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Processing dataset '{dataset_name}' for LLMEmb format...")

    # 1. Load TSV split files
    split_dfs = []
    for split_name in ["train", "val", "test", "dataset"]:
        tsv_file = raw_dir / f"{split_name}.tsv"
        if tsv_file.exists():
            df = pd.read_csv(
                tsv_file,
                sep="\t",
                header=None,
                dtype={0: str, 1: str}
            )
            # Standardize columns to [user, item, rating, timestamp]
            if len(df.columns) == 2:
                df.columns = ["user", "item"]
                df["rating"] = 1.0
                df["timestamp"] = 0
            elif len(df.columns) == 3:
                df.columns = ["user", "item", "rating"]
                df["timestamp"] = 0
            else:
                df.columns = ["user", "item", "rating", "timestamp"][:len(df.columns)]
            split_dfs.append(df)

    if not split_dfs:
        raise FileNotFoundError(f"No interaction TSV files found in {raw_dir}")

    full_df = pd.concat(split_dfs, ignore_index=True)

    # 2. Build 1-indexed Mappings (id_map.json)
    unique_users = sorted(full_df["user"].astype(str).unique())
    unique_items = sorted(full_df["item"].astype(str).unique())

    user2id = {u_raw: idx + 1 for idx, u_raw in enumerate(unique_users)}
    item2id = {i_raw: idx + 1 for idx, i_raw in enumerate(unique_items)}
    id2user = {idx + 1: u_raw for idx, u_raw in enumerate(unique_users)}
    id2item = {idx + 1: i_raw for idx, i_raw in enumerate(unique_items)}

    id_map = {
        "user2id": user2id,
        "item2id": item2id,
        "id2user": id2user,
        "id2item": id2item
    }

    id_map_file = output_dir / "id_map.json"
    with open(id_map_file, "w", encoding="utf-8") as f:
        json.dump(id_map, f, indent=4)
    logging.info(f"Saved ID mappings ({len(user2id)} users, {len(item2id)} items) to {id_map_file}")

    # 3. Generate inter_seq.txt & inter.txt
    user_interactions = {}
    for _, row in full_df.iterrows():
        u_raw = str(row["user"])
        i_raw = str(row["item"])
        ts = int(row["timestamp"]) if "timestamp" in row else 0

        u_id = user2id[u_raw]
        i_id = item2id[i_raw]

        if u_id not in user_interactions:
            user_interactions[u_id] = []
        user_interactions[u_id].append((i_id, ts))

    # Sort interactions per user by timestamp
    inter_seq_file = output_dir / "inter_seq.txt"
    inter_file = output_dir / "inter.txt"

    with open(inter_seq_file, "w", encoding="utf-8") as f_seq, open(inter_file, "w", encoding="utf-8") as f_inter:
        for u_id in sorted(user_interactions.keys()):
            # Sort items by timestamp
            item_list = [i for i, _ in sorted(user_interactions[u_id], key=lambda x: x[1])]
            
            # Remove duplicate consecutive interactions while preserving order
            seen_items = set()
            unique_seq = []
            for item in item_list:
                if item not in seen_items:
                    seen_items.add(item)
                    unique_seq.append(item)

            if unique_seq:
                # Write to inter_seq.txt: user_id item1 item2 item3 ...
                seq_str = f"{u_id} " + " ".join(map(str, unique_seq)) + "\n"
                f_seq.write(seq_str)

                # Write to inter.txt: user_id item_id per line
                for i_id in unique_seq:
                    f_inter.write(f"{u_id} {i_id}\n")

    logging.info(f"Saved sequential interactions to {inter_seq_file}")
    logging.info(f"Saved pairwise interactions to {inter_file}")

    # 4. Generate LLM Text Prompts (item_str_llmemb.jsonline & item_text.json)
    meta_file = raw_dir / "item_metadata.json"
    item_metadata = {}
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            item_metadata = json.load(f)

    jsonl_data = []
    item_text_dict = {}

    for i_raw, i_id in item2id.items():
        desc = item_metadata.get(str(i_raw), item_metadata.get(i_raw, f"Item {i_raw}"))
        
        # Build LLMEmb prompt structure
        prompt_text = f"The item has the following attributes: {desc}"
        item_text_dict[str(i_id)] = prompt_text

        jsonl_record = {
            "input": prompt_text,
            "target": "",
            "item": i_raw,
            "item_id": i_id
        }
        jsonl_data.append(jsonl_record)

    jsonl_file = output_dir / "item_str_llmemb.jsonline"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for record in jsonl_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    item_text_file = output_dir / "item_text.json"
    with open(item_text_file, "w", encoding="utf-8") as f:
        json.dump(item_text_dict, f, indent=4)

    logging.info(f"Saved LLM item text prompts for {len(jsonl_data)} items to {jsonl_file}")

    # 5. Mirror files to LLMEmb-master/data/{dataset_name}/handled/ if available
    llmemb_master_handled = root_dir / "LLMEmb-master" / "data" / dataset_name / "handled"
    if (root_dir / "LLMEmb-master").exists():
        llmemb_master_handled.mkdir(parents=True, exist_ok=True)
        import shutil
        for f_name in ["id_map.json", "inter_seq.txt", "inter.txt", "item_str_llmemb.jsonline", "item_text.json"]:
            src = output_dir / f_name
            dst = llmemb_master_handled / f_name
            shutil.copy2(src, dst)
        logging.info(f"Mirrored handled files to LLMEmb-master path: {llmemb_master_handled}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Convert KnowLight TSV datasets to LLMEmb data format")
    parser.add_argument(
        "--dataset",
        type=str,
        default="movielens",
        help="Dataset name ('movielens', 'amazon_beauty', 'amazon_toys_and_games', 'amazon_cd_2014', 'yelp', or 'all')"
    )
    args = parser.parse_args()

    target_datasets = ["movielens", "amazon_beauty", "amazon_toys_and_games", "amazon_cd_2014", "yelp"] if args.dataset == "all" else [args.dataset]

    print("=" * 70)
    print(" KnowLight to LLMEmb Data Converter ")
    print(f" Target Datasets: {target_datasets}")
    print("=" * 70)

    for ds in target_datasets:
        print(f"\nConverting dataset '{ds}'...")
        try:
            out_p = convert_knowlight_to_llmemb_format(ds)
            print(f" Dataset '{ds}' successfully converted -> {out_p}")
        except FileNotFoundError as e:
            print(f" [Skipped] {e}")
        except Exception as e:
            print(f" [Error] Failed to convert dataset '{ds}': {e}")

    print("\n" + "=" * 70)
    print(" Data conversion completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
