import os
from pathlib import Path

# Project root directory (KnowLight repository root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Centralized Data Directory Structure under root data/
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
GRAPHS_DIR = DATA_DIR / "graphs"
PROMPTS_DIR = DATA_DIR / "prompts"
CACHE_DIR = DATA_DIR / "cache"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
STUDENTS_DIR = CHECKPOINTS_DIR / "students"
DISTILLED_STUDENTS_DIR = CHECKPOINTS_DIR / "distilled_students"
TEACHERS_CHECKPOINTS_DIR = CHECKPOINTS_DIR / "teachers"

MODELS_DIR = PROJECT_ROOT / "models"

# Ensure essential data subdirectories exist
for d in [DATA_DIR, RAW_DATA_DIR, GRAPHS_DIR, PROMPTS_DIR, CACHE_DIR, CHECKPOINTS_DIR, STUDENTS_DIR, DISTILLED_STUDENTS_DIR, TEACHERS_CHECKPOINTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset Name Constants
AMAZONBEAUTY = "amazon_beauty"
AMAZONCD = "amazon_cd_2014"
AMAZONTOYSANDGAMES = "amazon_toys_and_games"
MOVIELENS = "movielens"
YELP = "yelp"


def dataset_directory(dataset_name: str, create_if_not_exists: bool = True) -> Path:
    """Returns Path to the raw data directory for a given dataset."""
    base_dir = RAW_DATA_DIR / dataset_name
    if create_if_not_exists:
        base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def dataset_filepath(dataset_name: str, split_name: str = "processed", exists: bool = False) -> Path:
    """Returns Path to a specific split file (tsv/csv) for a given dataset."""
    target_dir = dataset_directory(dataset_name, create_if_not_exists=True)
    
    if split_name in ("processed", "dataset"):
        filename = "dataset.tsv"
    elif split_name in ("val", "validation"):
        filename = "val.tsv"
    else:
        filename = f"{split_name}.tsv"
        
    filepath = target_dir / filename
    if exists and not filepath.exists():
        # Try .csv extension fallback
        csv_filepath = target_dir / f"{filename.stem}.csv"
        if csv_filepath.exists():
            return csv_filepath
    return filepath


def get_next_run_dir(base_dir: Path, prefix: str) -> Path:
    """
    Scans base_dir for subdirectories matching f"{prefix}_<num>" 
    and creates/returns a new Path f"{prefix}_{max_num + 1}".
    If no matching directories exist, creates and returns base_dir / f"{prefix}_1".
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    existing_nums = []
    prefix_pattern = f"{prefix}_"
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.startswith(prefix_pattern):
            num_str = item.name[len(prefix_pattern):]
            if num_str.isdigit():
                existing_nums.append(int(num_str))
    
    next_num = max(existing_nums) + 1 if existing_nums else 1
    run_dir = base_dir / f"{prefix}_{next_num}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_latest_or_next_run_dir(base_dir: Path, prefix: str, resume: bool = True) -> Path:
    """
    Scans base_dir for subdirectories matching f"{prefix}_<num>".
    If resume=True, checks if the latest directory contains an unfinished checkpoint 
    (e.g., temp_checkpoint.pt or checkpoint_grid_state.json) and returns it.
    Otherwise, creates and returns a new Path f"{prefix}_{max_num + 1}".
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    existing_nums = []
    prefix_pattern = f"{prefix}_"
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.startswith(prefix_pattern):
            num_str = item.name[len(prefix_pattern):]
            if num_str.isdigit():
                existing_nums.append(int(num_str))

    if existing_nums and resume:
        max_num = max(existing_nums)
        latest_dir = base_dir / f"{prefix}_{max_num}"
        has_grid_checkpoint = (latest_dir / "checkpoint_grid_state.json").exists() or (latest_dir / "grid_search_checkpoint.json").exists()
        has_epoch_checkpoint = (latest_dir / "temp_checkpoint.pt").exists()
        has_finished_report = any(latest_dir.glob("*results.json"))

        if (has_grid_checkpoint or has_epoch_checkpoint) and not has_finished_report:
            return latest_dir

    return get_next_run_dir(base_dir, prefix)

