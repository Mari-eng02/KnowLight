import os
import gzip
import json
from pathlib import Path
import pandas as pd


def load_dataset_item_metadata(dataset_name: str, dataset_dir: Path | str | None = None) -> dict:
    """
    Loads item metadata (title, categories, description, genres, brand) for a given dataset.
    Checks for item_metadata.json saved by dataset download scripts first.
    """
    if dataset_dir:
        meta_json_file = Path(dataset_dir) / "item_metadata.json"
        if meta_json_file.exists():
            try:
                with open(meta_json_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

    dataset_name = dataset_name.lower()
    if "movielens" in dataset_name:
        return _load_movielens_metadata(dataset_dir)
    elif "yelp" in dataset_name:
        return _load_yelp_metadata(dataset_dir)
    elif "amazon" in dataset_name:
        return _load_amazon_metadata(dataset_name, dataset_dir)

    return {}


def save_item_metadata_from_datarec(dataset_name: str, raw_dataset, dataset_dir: Path | str) -> Path | None:
    """
    Extracts secondary item metadata from raw DataRec dataset resources or local files
    and saves item_metadata.json directly into dataset_dir.
    """
    dataset_dir = Path(dataset_dir)
    meta_path = dataset_dir / "item_metadata.json"
    meta_dict = {}

    # 1. MovieLens
    if "movielens" in dataset_name.lower():
        try:
            if hasattr(raw_dataset, "resources") and "movies" in raw_dataset.resources:
                movies_res = raw_dataset.resources["movies"]
                movies_res.prepare()
                m_path = str(movies_res.path() if callable(movies_res.path) else movies_res.path)
                if os.path.exists(m_path):
                    df = pd.read_csv(m_path, sep="::", names=["item_id", "title", "genres"], engine="python", encoding="latin-1")
                    for _, row in df.iterrows():
                        meta_dict[str(row["item_id"])] = f"Title: {row['title']} | Genres: {row['genres']}"
        except Exception as e:
            print(f"[Metadata Notice] MovieLens extraction notice: {e}")

    # 2. Yelp
    elif "yelp" in dataset_name.lower():
        try:
            if hasattr(raw_dataset, "resources") and "business" in raw_dataset.resources:
                b_res = raw_dataset.resources["business"]
                b_res.prepare()
                b_path = str(b_res.path() if callable(b_res.path) else b_res.path)
                if os.path.exists(b_path):
                    with open(b_path, "r", encoding="utf-8") as f:
                        for line in f:
                            data = json.loads(line)
                            b_id = str(data.get("business_id"))
                            name = data.get("name", "")
                            categories = data.get("categories", "")
                            stars = data.get("stars", "")
                            meta_dict[b_id] = f"Business Name: {name} | Categories: {categories} | Rating: {stars} stars"
        except Exception as e:
            print(f"[Metadata Notice] Yelp extraction notice: {e}")

    # 3. Amazon Datasets
    elif "amazon" in dataset_name.lower():
        meta_dict = _load_amazon_metadata(dataset_name, dataset_dir)

    if meta_dict:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)
        print(f"[Metadata Loader] Saved item_metadata.json to {meta_path} ({len(meta_dict)} items).")
        return meta_path

    return None


def _load_movielens_metadata(dataset_dir: Path | str | None = None) -> dict:
    """Loads MovieLens item metadata (MovieID -> Title, Genres)."""
    meta_map = {}
    try:
        from datarec.datasets import Movielens
        ml = Movielens(version="1m")
        ml.prepare()
        if "movies" in ml.resources:
            movies_res = ml.resources["movies"]
            movies_res.prepare()
            m_path = str(movies_res.path() if callable(movies_res.path) else movies_res.path)
            if os.path.exists(m_path):
                df = pd.read_csv(m_path, sep="::", names=["item_id", "title", "genres"], engine="python", encoding="latin-1")
                for _, row in df.iterrows():
                    meta_map[str(row["item_id"])] = f"Title: {row['title']} | Genres: {row['genres']}"
                return meta_map
    except Exception:
        pass

    if dataset_dir:
        movies_path = Path(dataset_dir) / "movies.dat"
        if movies_path.exists():
            df = pd.read_csv(movies_path, sep="::", names=["item_id", "title", "genres"], engine="python", encoding="latin-1")
            for _, row in df.iterrows():
                meta_map[str(row["item_id"])] = f"Title: {row['title']} | Genres: {row['genres']}"

    return meta_map


def _load_yelp_metadata(dataset_dir: Path | str | None = None) -> dict:
    """Loads Yelp business metadata (business_id -> Name, Categories, Stars, Attributes)."""
    meta_map = {}
    json_paths = []
    
    if dataset_dir:
        json_paths.append(Path(dataset_dir) / "yelp_academic_dataset_business.json")
        json_paths.append(Path(dataset_dir) / "business.json")

    for p in json_paths:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    b_id = str(data.get("business_id"))
                    name = data.get("name", "")
                    categories = data.get("categories", "")
                    stars = data.get("stars", "")
                    meta_map[b_id] = f"Business Name: {name} | Categories: {categories} | Rating: {stars} stars"
            break

    return meta_map


def _load_amazon_metadata(dataset_name: str, dataset_dir: Path | str | None = None) -> dict:
    """Loads Amazon product metadata (ASIN -> Title, Categories, Brand, Description)."""
    import urllib.request

    meta_map = {}
    dataset_dir = Path(dataset_dir) if dataset_dir else None
    possible_paths = []

    if dataset_dir:
        possible_paths.extend([
            dataset_dir / "meta.jsonl.gz",
            dataset_dir / "meta.json.gz",
            dataset_dir / "meta.json",
            dataset_dir / "metadata.json"
        ])

    existing_path = None
    for p in possible_paths:
        if p.exists():
            existing_path = p
            break

    # If no local metadata file exists, download automatically from official UCSD / SNAP repositories
    if not existing_path and dataset_dir:
        dataset_name_lower = dataset_name.lower()
        target_gz = dataset_dir / "meta.jsonl.gz"
        url = None

        if "beauty" in dataset_name_lower:
            url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Beauty_and_Personal_Care.jsonl.gz"
        elif "toys" in dataset_name_lower:
            url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Toys_and_Games.jsonl.gz"
        elif "2014" in dataset_name_lower:
            url = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_CDs_and_Vinyl.json.gz"
        elif "cd" in dataset_name_lower:
            url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_CDs_and_Vinyl.jsonl.gz"

        if url:
            print(f"[Metadata Loader] Downloading raw Amazon metadata from {url}...")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as resp, open(target_gz, "wb") as out_file:
                    while chunk := resp.read(1024 * 1024):
                        out_file.write(chunk)
                print(f"[Metadata Loader] Downloaded metadata to {target_gz}")
                existing_path = target_gz
            except Exception as e:
                print(f"[Metadata Loader Error] Failed to download metadata from {url}: {e}")

    if existing_path and existing_path.exists():
        open_fn = gzip.open if str(existing_path).endswith(".gz") else open
        import ast
        with open_fn(existing_path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    try:
                        data = json.loads(line)
                    except Exception:
                        data = ast.literal_eval(line)

                    asin = str(data.get("parent_asin") or data.get("asin"))
                    title = data.get("title", "")
                    cats = data.get("categories", "")
                    if isinstance(cats, list):
                        cats = " > ".join([c for sub in cats for c in (sub if isinstance(sub, list) else [sub])])
                    brand = data.get("store") or data.get("brand", "")

                    text_pieces = [f"Title: {title}"]
                    if cats:
                        text_pieces.append(f"Categories: {cats}")
                    if brand:
                        text_pieces.append(f"Brand: {brand}")

                    meta_map[asin] = " | ".join(text_pieces)
                except Exception:
                    continue

    return meta_map
