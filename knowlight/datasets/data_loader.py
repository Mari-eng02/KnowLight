from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import os
import numpy as np
import scipy.sparse as sp

from knowlight.datasets.interactions import InteractionDataset
from knowlight.paths import dataset_directory, dataset_filepath
from knowlight.datasets.metadata_loader import load_dataset_item_metadata


USER_COLUMNS = ("user", "userId", "user_id", "uid", 0)
ITEM_COLUMNS = ("item", "itemId", "item_id", "iid", 1)
RATING_COLUMNS = ("rating", "ratings", "score", 2)
TIMESTAMP_COLUMNS = ("timestamp", "time", "ts", 3)
SPLIT_ORDER = ("train", "val", "test")


@dataclass(frozen=True)
class LoadedSplit:
    name: str
    path: Path
    frame: pd.DataFrame
    backend: str


@dataclass(frozen=True)
class EncodedDataset:
    frames: dict[str, pd.DataFrame]
    num_users: int
    num_items: int
    backend: str


def datarec_available() -> bool:
    return _datarec_reader() is not None


def load_split_frame(
    dataset_name: str,
    split_name: str,
    *,
    repo_root: Path | str = ".",
    use_datarec: bool = True,
) -> LoadedSplit:
    """Load one dataset split through DataRec, falling back to pandas if unavailable."""
    path = _split_path(dataset_name=dataset_name, split_name=split_name, repo_root=repo_root)
    columns = _column_names(dataset_name)
    return load_frame_from_path(path, dataset_name=dataset_name, split_name=split_name, columns=columns, use_datarec=use_datarec)


def load_frame_from_path(
    path: Path | str,
    *,
    dataset_name: str,
    split_name: str,
    columns: list[str] | None = None,
    use_datarec: bool = True,
) -> LoadedSplit:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{split_name} split not found at {path}")
    columns = columns or ["userId", "itemId", "rating"]

    reader = _datarec_reader() if use_datarec else None
    if reader is not None:
        data = reader(
            str(path),
            sep="\t",
            user_col=columns[0],
            item_col=columns[1],
            rating_col=columns[2] if len(columns) > 2 else None,
            timestamp_col=columns[3] if len(columns) > 3 else None,
            header=None,
            cols=columns,
            dataset_name=dataset_name,
            version_name=split_name,
        )
        return LoadedSplit(
            name=split_name,
            path=path,
            frame=_datarec_to_frame(data),
            backend="datarec",
        )

    frame = pd.read_csv(path, sep="\t", header=None, names=columns)
    print(frame.columns)
    return LoadedSplit(name=split_name, path=path, frame=frame, backend="pandas")


def load_train_dataset(
    dataset_name: str,
    teacher_num_users: int,
    teacher_num_items: int,
    user_mapping: dict[int, int] | dict[str, int] | None = None,
    item_mapping: dict[int, int] | dict[str, int] | None = None,
    id_space: str | None = None,
) -> tuple[InteractionDataset, int]:
    # Build train structures (train_dict, train_sequences, interactions) preserving timestamps
    train_dict = {}
    train_sequences = {}
    train_timestamps: dict[int, list[int]] = {}
    interactions = []
    dropped = 0

    def _process_frame(frame: pd.DataFrame):
        nonlocal dropped
        user_col = _pick_column(frame.columns, USER_COLUMNS)
        item_col = _pick_column(frame.columns, ITEM_COLUMNS)
        ts_col = None
        try:
            ts_col = _pick_column(frame.columns, TIMESTAMP_COLUMNS)
        except ValueError:
            ts_col = None

        # If timestamp column exists, sort by it to get chronological order
        if ts_col is not None:
            try:
                frame_sorted = frame.sort_values(by=[user_col, ts_col])
            except Exception:
                frame_sorted = frame
        else:
            frame_sorted = frame

        user_encoder: dict[Any, int] = {}
        item_encoder: dict[Any, int] = {}

        if ts_col is not None and ts_col in frame_sorted.columns:
            it = frame_sorted[[user_col, item_col, ts_col]].itertuples(index=False, name=None)
        else:
            it = frame_sorted[[user_col, item_col]].itertuples(index=False, name=None)
        for row in it:
            user = _resolve_or_encode(row[0], user_mapping, user_encoder, prefer_numeric=teacher_num_users is not None)
            item = _resolve_or_encode(row[1], item_mapping, item_encoder, prefer_numeric=teacher_num_items is not None)
            if user is None or item is None:
                dropped += 1
                continue
            if user < 0 or item < 0:
                dropped += 1
                continue
            if teacher_num_users is not None and user >= teacher_num_users:
                dropped += 1
                continue
            if teacher_num_items is not None and item >= teacher_num_items:
                dropped += 1
                continue
            train_dict.setdefault(user, set()).add(item)
            train_sequences.setdefault(user, []).append(item)
            if ts_col is not None and len(row) > 2:
                try:
                    ts = int(row[2])
                except Exception:
                    ts = row[2]
                train_timestamps.setdefault(user, []).append(ts)
            interactions.append((user, item))

    # Handle different id_space / encoding strategies similar to load_ground_truth_split
    if _normalize_id_space(id_space) == "dataset_integer":
        loaded = load_split_frame(dataset_name, "train", use_datarec=False)
        _process_frame(loaded.frame)
    elif user_mapping is None and item_mapping is None:
        encoded = load_encoded_dataset(dataset_name)
        # encoded frames contain already encoded user/item columns
        frame = encoded.frames.get("train")
        if frame is None:
            raise ValueError("Unknown split: train")
        _process_frame(frame)
    else:
        loaded = load_split_frame(dataset_name, "train")
        _process_frame(loaded.frame)

    dataset = InteractionDataset.from_train_dict(
        train_dict=train_dict,
        num_users=teacher_num_users,
        num_items=teacher_num_items,
        train_sequences=train_sequences,
    )
    # Attach timestamps aligned with train_sequences if available
    if train_timestamps:
        dataset.train_timestamps = train_timestamps
    # replace interactions with chronological interactions (preserve original attribute for compatibility)
    dataset.interactions = interactions
    return dataset, dropped


def load_eval_split(
    dataset_name: str,
    split_name: str,
    teacher_num_users: int,
    teacher_num_items: int,
    user_mapping: dict[int, int] | dict[str, int] | None = None,
    item_mapping: dict[int, int] | dict[str, int] | None = None,
    id_space: str | None = None,
) -> tuple[dict[int, set[int]], int]:
    return load_ground_truth_split(
        dataset_name=dataset_name,
        split_name=split_name,
        num_users=teacher_num_users,
        num_items=teacher_num_items,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        id_space=id_space,
    )


def load_interaction_dataset(
    dataset_name: str,
    user_mapping: dict[int, int] | dict[str, int] | None = None,
    item_mapping: dict[int, int] | dict[str, int] | None = None,
    num_users: int | None = None,
    num_items: int | None = None,
    id_space: str | None = None,
) -> InteractionDataset:
    if _normalize_id_space(id_space) == "dataset_integer":
        split, _ = _frame_to_split(
            load_split_frame(dataset_name, "train", use_datarec=False).frame,
            user_mapping=None,
            item_mapping=None,
            num_users=num_users,
            num_items=num_items,
        )
        if num_users is None:
            num_users = max(split.keys(), default=-1) + 1
        if num_items is None:
            num_items = max((item for items in split.values() for item in items), default=-1) + 1
    elif user_mapping is None and item_mapping is None:
        encoded = load_encoded_dataset(dataset_name)
        if num_users is None:
            num_users = encoded.num_users
        if num_items is None:
            num_items = encoded.num_items
        split, _ = _encoded_frame_to_split(encoded.frames["train"], num_users=num_users, num_items=num_items)
    else:
        split, _ = _frame_to_split(
            load_split_frame(dataset_name, "train").frame,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            num_users=num_users,
            num_items=num_items,
        )
        if num_users is None:
            num_users = max(split.keys(), default=-1) + 1
        if num_items is None:
            num_items = max((item for items in split.values() for item in items), default=-1) + 1
    return InteractionDataset.from_train_dict(split, num_users=num_users, num_items=num_items)


def resolve_teacher_dataset_mappings(
    metadata: dict[str, Any] | None,
    *,
    dataset_name: str,
) -> tuple[dict[int, int] | dict[str, int] | None, dict[int, int] | dict[str, int] | None, str]:
    metadata = metadata or {}
    user_mapping = metadata.get("public_to_local_user_id")
    item_mapping = metadata.get("public_to_local_item_id")
    if user_mapping or item_mapping:
        return user_mapping, item_mapping, "teacher_metadata"

    id_space = _normalize_id_space(metadata.get("id_space"))
    if id_space == "dataset_integer":
        return None, None, "dataset_integer"
    if id_space in {"internal_integer", "framework_integer", "local_integer"}:
        raise ValueError(
            f"External teacher artifact declares id_space={id_space!r}, so its user/item IDs are framework-local "
            "indices rather than dataset-original IDs. Evaluation/training against raw dataset splits requires "
            "public_to_local_user_id and public_to_local_item_id metadata, or a prediction export converted back "
            "to the original user/item IDs."
        )

    framework = str(metadata.get("framework") or metadata.get("source") or "").strip().lower()
    source_framework = str(metadata.get("import_source_framework") or "").strip().lower()
    if framework == "external" or source_framework == "external":
        raise ValueError(
            "External teacher artifacts must define either public_to_local_user_id/public_to_local_item_id "
            "metadata or id_space='dataset_integer'. Refusing to evaluate with an implicit DataRec re-encoding."
        )

    if not user_mapping and not item_mapping:
        return None, None, "datarec"


def load_ground_truth_split(
    dataset_name: str,
    split_name: str,
    num_users: int | None = None,
    num_items: int | None = None,
    user_mapping: dict[int, int] | dict[str, int] | None = None,
    item_mapping: dict[int, int] | dict[str, int] | None = None,
    id_space: str | None = None,
) -> tuple[dict[int, set[int]], int]:
    if _normalize_id_space(id_space) == "dataset_integer":
        loaded = load_split_frame(dataset_name, split_name, use_datarec=False)
        return _frame_to_split(
            loaded.frame,
            user_mapping=None,
            item_mapping=None,
            num_users=num_users,
            num_items=num_items,
        )

    if user_mapping is None and item_mapping is None:
        encoded = load_encoded_dataset(dataset_name)
        normalized_split = "val" if split_name == "validation" else split_name
        if normalized_split not in encoded.frames:
            raise ValueError(f"Unknown split: {split_name}")
        return _encoded_frame_to_split(
            encoded.frames[normalized_split],
            num_users=num_users,
            num_items=num_items,
        )

    loaded = load_split_frame(dataset_name, split_name)
    return _frame_to_split(
        loaded.frame,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        num_users=num_users,
        num_items=num_items,
    )


def load_encoded_dataset(dataset_name: str, *, repo_root: Path | str = ".") -> EncodedDataset:
    """Load all splits and apply one shared DataRec user/item encoding."""
    return _load_encoded_dataset_cached(dataset_name, str(Path(repo_root)))


@lru_cache(maxsize=None)
def _load_encoded_dataset_cached(dataset_name: str, repo_root: str) -> EncodedDataset:
    loaded_splits = {
        split: load_split_frame(dataset_name, split, repo_root=repo_root, use_datarec=True)
        for split in SPLIT_ORDER
    }
    frames = {split: loaded.frame.copy() for split, loaded in loaded_splits.items()}
    encoded_frames, num_users, num_items = _encode_split_frames_with_datarec(frames)
    backends = {loaded.backend for loaded in loaded_splits.values()}
    backend = "datarec" if backends == {"datarec"} else "+".join(sorted(backends))
    return EncodedDataset(frames=encoded_frames, num_users=num_users, num_items=num_items, backend=backend)


def _encode_split_frames_with_datarec(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], int, int]:
    encoder_cls = _datarec_incremental_encoder() or _FallbackIncrementalEncoder
    user_encoder = encoder_cls()
    item_encoder = encoder_cls()
    encoded: dict[str, pd.DataFrame] = {}

    for split in SPLIT_ORDER:
        frame = frames[split].copy()
        user_col = _pick_column(frame.columns, USER_COLUMNS)
        item_col = _pick_column(frame.columns, ITEM_COLUMNS)
        frame[user_col] = user_encoder.encode_many(frame[user_col].tolist())
        frame[item_col] = item_encoder.encode_many(frame[item_col].tolist())
        frame[user_col] = frame[user_col].astype("int64")
        frame[item_col] = frame[item_col].astype("int64")
        encoded[split] = frame

    return encoded, len(user_encoder), len(item_encoder)


def _encoded_frame_to_split(
    frame: pd.DataFrame,
    *,
    num_users: int | None,
    num_items: int | None,
) -> tuple[dict[int, set[int]], int]:
    user_col = _pick_column(frame.columns, USER_COLUMNS)
    item_col = _pick_column(frame.columns, ITEM_COLUMNS)
    split: dict[int, set[int]] = {}
    dropped = 0

    for user_raw, item_raw in frame[[user_col, item_col]].itertuples(index=False, name=None):
        try:
            user = int(user_raw)
            item = int(item_raw)
        except (TypeError, ValueError):
            dropped += 1
            continue
        if user < 0 or item < 0:
            dropped += 1
            continue
        if num_users is not None and user >= num_users:
            dropped += 1
            continue
        if num_items is not None and item >= num_items:
            dropped += 1
            continue
        split.setdefault(user, set()).add(item)
    return split, dropped


class _FallbackIncrementalEncoder:
    def __init__(self, offset: int = 0):
        self.offset = offset
        self._forward: dict[Any, int] = {}
        self._reverse: list[Any] = []

    def __len__(self) -> int:
        return len(self._forward)

    def encode_one(self, key: Any) -> int:
        if key in self._forward:
            return self._forward[key]
        idx = self.offset + len(self._forward)
        self._forward[key] = idx
        self._reverse.append(key)
        return idx

    def encode_many(self, values: Iterable[Any]) -> list[int]:
        return [self.encode_one(value) for value in values]


def resolve_local_id(raw_value: Any, mapping: dict[int, int] | dict[str, int] | None) -> int | None:
    candidates: list[Any] = [raw_value]
    raw_string = str(raw_value).strip()
    candidates.append(raw_string)
    try:
        parsed_int = int(raw_string)
    except (TypeError, ValueError):
        parsed_int = None
    if parsed_int is not None:
        candidates.extend([parsed_int, str(parsed_int)])

    if mapping is not None:
        mapped_value = None
        for candidate in candidates:
            if candidate in mapping:
                mapped_value = mapping[candidate]
                break
    else:
        mapped_value = parsed_int if parsed_int is not None else raw_string

    if mapped_value is None:
        return None
    try:
        return int(mapped_value)
    except (TypeError, ValueError):
        return None


def _frame_to_split(
    frame: pd.DataFrame,
    *,
    user_mapping: dict[int, int] | dict[str, int] | None,
    item_mapping: dict[int, int] | dict[str, int] | None,
    num_users: int | None,
    num_items: int | None,
) -> tuple[dict[int, set[int]], int]:
    user_col = _pick_column(frame.columns, USER_COLUMNS)
    item_col = _pick_column(frame.columns, ITEM_COLUMNS)
    split: dict[int, set[int]] = {}
    dropped = 0
    user_encoder: dict[Any, int] = {}
    item_encoder: dict[Any, int] = {}

    for row in frame[[user_col, item_col]].itertuples(index=False, name=None):
        user = _resolve_or_encode(row[0], user_mapping, user_encoder, prefer_numeric=num_users is not None)
        item = _resolve_or_encode(row[1], item_mapping, item_encoder, prefer_numeric=num_items is not None)
        if user is None or item is None:
            dropped += 1
            continue
        if user < 0 or item < 0:
            dropped += 1
            continue
        if num_users is not None and user >= num_users:
            dropped += 1
            continue
        if num_items is not None and item >= num_items:
            dropped += 1
            continue
        split.setdefault(user, set()).add(item)
    return split, dropped


def _resolve_or_encode(
    raw_value: Any,
    mapping: dict[int, int] | dict[str, int] | None,
    encoder: dict[Any, int],
    prefer_numeric: bool,
) -> int | None:
    resolved = resolve_local_id(raw_value, mapping)
    if resolved is not None:
        return resolved
    if mapping is not None or prefer_numeric:
        return resolved

    key = str(raw_value).strip()
    if key == "":
        return None
    if key not in encoder:
        encoder[key] = len(encoder)
    return encoder[key]


def _normalize_id_space(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _looks_like_identity_mapping(mapping: Any) -> bool:
    if not isinstance(mapping, dict) or not mapping:
        return False
    checked = 0
    for key, value in mapping.items():
        try:
            if int(str(key).strip()) != int(value):
                return False
        except (TypeError, ValueError):
            return False
        checked += 1
        if checked >= 1000:
            break
    return True


def _dataset_has_sparse_numeric_ids(dataset_name: str) -> bool:
    for split_name in SPLIT_ORDER:
        try:
            frame = load_split_frame(dataset_name, split_name).frame
        except Exception:
            continue
        for candidates in (USER_COLUMNS, ITEM_COLUMNS):
            column = _pick_column(frame.columns, candidates)
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.isna().any() or values.empty:
                continue
            max_id = int(values.max())
            unique_count = int(values.nunique())
            if max_id + 1 > unique_count:
                return True
    return False


def _infer_numeric_shape_from_splits(dataset_name: str) -> tuple[int | None, int | None]:
    try:
        encoded = load_encoded_dataset(dataset_name)
    except Exception:
        return None, None
    return encoded.num_users, encoded.num_items


def _split_path(dataset_name: str, split_name: str, repo_root: Path | str) -> Path:
    normalized_split = "val" if split_name in ("val", "validation") else split_name
    path = dataset_filepath(dataset_name, normalized_split, exists=False)
    if not path.exists():
        # Fallback search under repo_root / data / dataset_name
        fallback = Path(repo_root) / "data" / dataset_name / f"{normalized_split}.tsv"
        if fallback.exists():
            return fallback
        fallback_csv = Path(repo_root) / "data" / dataset_name / f"{normalized_split}.csv"
        if fallback_csv.exists():
            return fallback_csv
    return path


def _column_names(dataset_name: str) -> list[str]:
    return ["userId", "itemId", "rating", "timestamp"]


def _datarec_reader():
    try:
        from datarec.io import read_transactions_tabular

        return read_transactions_tabular
    except Exception:
        pass
    try:
        from datarec.io.readers.transactions.tabular import read_transactions_tabular

        return read_transactions_tabular
    except Exception:
        return None


def _datarec_incremental_encoder():
    try:
        from datarec.data.utils import IncrementalEncoder

        return IncrementalEncoder
    except Exception:
        return None


def _datarec_to_frame(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()

    for attr_path in (
        ("data",),
        ("df",),
        ("dataframe",),
        ("transactions",),
        ("interactions",),
        ("rawdata", "transactions"),
        ("rawdata", "interactions"),
        ("rawdata", "data"),
        ("rawdata", "df"),
        ("raw_data", "transactions"),
        ("raw_data", "interactions"),
        ("raw_data", "data"),
        ("raw_data", "df"),
    ):
        value = _get_nested_attr(data, attr_path)
        if isinstance(value, pd.DataFrame):
            return value.copy()
        if isinstance(value, (list, tuple, dict)):
            return pd.DataFrame(value)

    for method in ("to_pandas", "to_dataframe", "as_dataframe"):
        candidate = getattr(data, method, None)
        if callable(candidate):
            value = candidate()
            if isinstance(value, pd.DataFrame):
                return value.copy()
            if isinstance(value, (list, tuple, dict)):
                return pd.DataFrame(value)

    raise TypeError(
        "Unable to extract a pandas DataFrame from DataRec output. "
        f"Unsupported object type: {type(data)!r}"
    )


def _get_nested_attr(value: Any, attr_path: Iterable[str]) -> Any:
    current = value
    for attr in attr_path:
        current = getattr(current, attr, None)
        if current is None:
            return None
    return current


def _pick_column(columns: Iterable[Any], candidates: Iterable[Any]) -> Any:
    cols = list(columns)
    for candidate in candidates:
        if candidate in cols:
            return candidate
    for candidate in candidates:
        if isinstance(candidate, int) and 0 <= candidate < len(cols):
            return cols[candidate]
    raise ValueError(f"Unable to find any of {list(candidates)} in columns {cols}")


class KnowLightDataLoader:
    """
    Unified Data Loader for KnowLight Framework.
    Loads preprocessed DataRec dataset splits (train.tsv, val.tsv, test.tsv).
    """

    def __init__(self, dataset_name: str = "amazon_beauty", data_dir: str = "./data_files", min_uc: int = 5, min_ic: int = 5):
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir
        self.min_uc = min_uc
        self.min_ic = min_ic

        self.interaction_dataset = None
        self.num_users = 0
        self.num_items = 0
        self.train_matrix = None
        self.train_dict = {}
        self.test_pos_items = {}
        self.val_pos_items = {}
        self.user_sequences = {}
        self.item_attributes = {}
        self.user_attributes = {}

    def load_data(self):
        """Loads dataset using DataRec split reader."""
        train_path = dataset_filepath(self.dataset_name, "train")
        if train_path.exists():
            print(f"[DataRec Loader] Loading DataRec dataset '{self.dataset_name}' from {train_path.parent}")
            self._load_from_datarec_splits()
            return

        csv_path = os.path.join(self.data_dir, f"{self.dataset_name}.csv")
        if os.path.exists(csv_path):
            print(f"[DataRec Loader] Loading raw CSV from {csv_path}...")
            df = pd.read_csv(csv_path)
            self._process_raw_df(df)
            return

        raise FileNotFoundError(
            f"[DataRec Loader Error] Dataset files for '{self.dataset_name}' not found under '{train_path.parent}'. "
            f"Please run the dataset preparation script first (e.g. 'python data/{self.dataset_name}.py')."
        )

    def _load_from_datarec_splits(self):
        """Loads train, validation, and test splits processed by DataRec."""
        try:
            encoded = load_encoded_dataset(self.dataset_name)
            train_df = encoded.frames.get("train")
            val_df = encoded.frames.get("val")
            test_df = encoded.frames.get("test")
            self.num_users = encoded.num_users
            self.num_items = encoded.num_items

        except Exception:
            train_loaded = load_split_frame(self.dataset_name, "train")
            val_loaded = load_split_frame(self.dataset_name, "val")
            test_loaded = load_split_frame(self.dataset_name, "test")

            train_df = train_loaded.frame
            val_df = val_loaded.frame
            test_df = test_loaded.frame

            user_set = set(train_df.iloc[:, 0]).union(val_df.iloc[:, 0]).union(test_df.iloc[:, 0])
            item_set = set(train_df.iloc[:, 1]).union(val_df.iloc[:, 1]).union(test_df.iloc[:, 1])

            user_map = {u: idx for idx, u in enumerate(sorted(user_set))}
            item_map = {i: idx for idx, i in enumerate(sorted(item_set))}

            self.num_users = len(user_map)
            self.num_items = len(item_map)

            train_df = train_df.copy()
            val_df = val_df.copy()
            test_df = test_df.copy()

            train_df.iloc[:, 0] = train_df.iloc[:, 0].map(user_map)
            train_df.iloc[:, 1] = train_df.iloc[:, 1].map(item_map)
            val_df.iloc[:, 0] = val_df.iloc[:, 0].map(user_map)
            val_df.iloc[:, 1] = val_df.iloc[:, 1].map(item_map)
            test_df.iloc[:, 0] = test_df.iloc[:, 0].map(user_map)
            test_df.iloc[:, 1] = test_df.iloc[:, 1].map(item_map)

        u_col, i_col = train_df.columns[0], train_df.columns[1]

        # Build train_dict
        self.train_dict = {}
        for u, i in train_df[[u_col, i_col]].values:
            self.train_dict.setdefault(int(u), set()).add(int(i))

        # Build InteractionDataset
        self.interaction_dataset = InteractionDataset.from_train_dict(
            train_dict=self.train_dict,
            num_users=self.num_users,
            num_items=self.num_items
        )

        # Build CSR matrix
        rows, cols = [], []
        for u, items in self.train_dict.items():
            for i in items:
                rows.append(u)
                cols.append(i)
        data = np.ones(len(rows), dtype=np.float32)
        self.train_matrix = sp.csr_matrix((data, (rows, cols)), shape=(self.num_users, self.num_items))

        # Build positive items for val and test
        self.val_pos_items = {}
        for u, i in val_df[[val_df.columns[0], val_df.columns[1]]].values:
            self.val_pos_items.setdefault(int(u), []).append(int(i))

        self.test_pos_items = {}
        for u, i in test_df[[test_df.columns[0], test_df.columns[1]]].values:
            self.test_pos_items.setdefault(int(u), []).append(int(i))

        # Build user interaction sequences
        grouped_train = train_df.groupby(u_col)[i_col].apply(list)
        self.user_sequences = {}
        for u_id, items in grouped_train.items():
            seq = items[-50:] if len(items) >= 50 else [0] * (50 - len(items)) + items
            self.user_sequences[int(u_id)] = seq

        # Load metadata textual features
        metadata_map = load_dataset_item_metadata(self.dataset_name, dataset_filepath(self.dataset_name, "train").parent)
        all_items = set(train_df.iloc[:, 1]).union(val_df.iloc[:, 1]).union(test_df.iloc[:, 1])
        for i_idx, raw_id in enumerate(sorted(all_items)):
            key = str(raw_id)
            if key in metadata_map:
                self.item_attributes[i_idx] = metadata_map[key]
            elif raw_id in metadata_map:
                self.item_attributes[i_idx] = metadata_map[raw_id]
            else:
                self.item_attributes[i_idx] = f"Item {raw_id} (Category: Recommendation Entity)"

        for u_idx in range(self.num_users):
            self.user_attributes[u_idx] = f"User {u_idx} Profile"

        total = len(train_df) + len(val_df) + len(test_df)
        print(f"[DataRec Loader] Loaded '{self.dataset_name}': {self.num_users} users, {self.num_items} items, {total} interactions.")

    def _process_raw_df(self, df: pd.DataFrame):
        u_col, i_col = df.columns[0], df.columns[1]
        user_mapping = {u: idx for idx, u in enumerate(df[u_col].unique())}
        item_mapping = {i: idx for idx, i in enumerate(df[i_col].unique())}

        self.num_users = len(user_mapping)
        self.num_items = len(item_mapping)

        self.train_dict = {}
        grouped = df.groupby(u_col)[i_col].apply(list)
        for u_raw, items in grouped.items():
            u_idx = user_mapping[u_raw]
            item_indices = [item_mapping[i] for i in items]
            if len(item_indices) >= 3:
                self.train_dict[u_idx] = set(item_indices[:-2])
                self.val_pos_items[u_idx] = [item_indices[-2]]
                self.test_pos_items[u_idx] = [item_indices[-1]]
            else:
                self.train_dict[u_idx] = set(item_indices[:-1])
                self.val_pos_items[u_idx] = [item_indices[-1]]
                self.test_pos_items[u_idx] = [item_indices[-1]]

        self.interaction_dataset = InteractionDataset.from_train_dict(self.train_dict, self.num_users, self.num_items)
        rows, cols = [], []
        for u, items in self.train_dict.items():
            for i in items:
                rows.append(u)
                cols.append(i)
        data = np.ones(len(rows), dtype=np.float32)
        self.train_matrix = sp.csr_matrix((data, (rows, cols)), shape=(self.num_users, self.num_items))

        for i_raw, i_idx in item_mapping.items():
            self.item_attributes[i_idx] = f"Item {i_raw}"
        for u_raw, u_idx in user_mapping.items():
            self.user_attributes[u_idx] = f"User {u_raw}"

