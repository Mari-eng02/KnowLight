import os
import sys
import pandas as pd
import numpy as np

# Add root workspace directory to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from datarec.datasets import Yelp
from knowlight.paths import YELP, dataset_directory, dataset_filepath
from knowlight.datasets.metadata_loader import save_item_metadata_from_datarec

dataset_name = YELP

print("[Yelp Processor] Loading raw Yelp dataset...")
raw_dataset = Yelp()
dataset = raw_dataset.prepare_and_load()

print("[Yelp Processor] Converting to pandas DataFrame...")
raw_data = dataset.to_rawdata()
if hasattr(raw_data, "dataframe"):
    df = raw_data.dataframe.copy()
elif hasattr(raw_data, "data"):
    df = pd.DataFrame(raw_data.data).copy()
else:
    df = pd.DataFrame(list(raw_data)).copy()

# Convert timestamp to integer UNIX epoch for uniformity with other datasets
print("[Yelp Processor] Converting timestamp to integer UNIX epoch...")
df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10**9

# 1. Binarize (keep positive ratings >= 3 and convert to implicit feedback 1)
print("[Yelp Processor] Binarizing ratings (threshold >= 3)...")
df = df[df["rating"] >= 3].copy()
df["rating"] = 1

# 2. Optimized Vectorized Iterative K-Core (cores=[10, 10])
print("[Yelp Processor] Running fast vectorized Iterative K-Core (cores=[10, 10])...")
user_core = 10
item_core = 10

iteration = 0
while True:
    initial_len = len(df)
    iteration += 1
    
    # Filter users with < 10 interactions
    user_counts = df["user_id"].value_counts()
    valid_users = user_counts[user_counts >= user_core].index
    df = df[df["user_id"].isin(valid_users)]
    
    # Filter items with < 10 interactions
    item_counts = df["item_id"].value_counts()
    valid_items = item_counts[item_counts >= item_core].index
    df = df[df["item_id"].isin(valid_items)]
    
    current_len = len(df)
    print(f"  Iteration {iteration}: {initial_len} -> {current_len} interactions")
    if current_len == initial_len:
        break

print(f"[Yelp Processor] K-Core complete. Final interactions: {len(df)}, Users: {df['user_id'].nunique()}, Items: {df['item_id'].nunique()}")

# 3. Optimized Temporal Leave-One-Last Splitting
print("[Yelp Processor] Performing temporal Leave-One-Last split...")
df = df.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)

# Group by user and rank interactions by timestamp descending
df["rank"] = df.groupby("user_id")["timestamp"].rank(method="first", ascending=False)

test_df = df[df["rank"] == 1].drop(columns=["rank"])
val_df = df[df["rank"] == 2].drop(columns=["rank"])
train_df = df[df["rank"] > 2].drop(columns=["rank"])
processed_df = df.drop(columns=["rank"])

print(f"[Yelp Processor] Split complete - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# 4. Save Tabular TSV Files
dataset_dir = dataset_directory(dataset_name=dataset_name, create_if_not_exists=True)
print(f"[Yelp Processor] Saving TSV files to: {dataset_dir}")

def save_tsv(dataframe, filepath):
    dataframe.to_csv(filepath, sep="\t", header=False, index=False, columns=["user_id", "item_id", "rating", "timestamp"])
    print(f"  Saved {os.path.basename(filepath)} ({len(dataframe)} rows)")

save_tsv(processed_df, dataset_filepath(dataset_name, "processed", exists=False))
save_tsv(train_df, dataset_filepath(dataset_name, "train", exists=False))
save_tsv(val_df, dataset_filepath(dataset_name, "val", exists=False))
save_tsv(test_df, dataset_filepath(dataset_name, "test", exists=False))

# 5. Save Item Metadata JSON
print("[Yelp Processor] Extracting item metadata JSON...")
save_item_metadata_from_datarec(dataset_name, raw_dataset, dataset_dir)
print("[Yelp Processor] All done!")

