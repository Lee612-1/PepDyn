from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import lmdb
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import knn_graph
from tqdm import tqdm

AMINO_ACIDS = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]
RESIDUE_TO_INDEX = {name: idx for idx, name in enumerate(AMINO_ACIDS)}
ELEMENTS = ["C", "N", "O", "S", "P", "H", "HALOGEN", "OTHER"]
ELEMENT_TO_INDEX = {name: idx for idx, name in enumerate(ELEMENTS)}


def infer_element(atom_name: str) -> str:
    token = "".join(ch for ch in atom_name if ch.isalpha())
    token = token.capitalize()
    if not token:
        return "OTHER"
    if token.startswith(("Cl", "Br", "F", "I")):
        return "HALOGEN"
    first = token[0]
    return first if first in {"C", "N", "O", "S", "P", "H"} else "OTHER"


def one_hot(indices: np.ndarray, size: int) -> np.ndarray:
    eye = np.eye(size, dtype=np.float32)
    return eye[indices]


def prune_carbon_bound_hydrogens(
    coords: np.ndarray,
    atom_names: Sequence[str],
    residue_ids: Sequence[int],
    distance_cutoff: float = 1.25,
) -> np.ndarray:
    elements = [infer_element(name) for name in atom_names]
    keep_mask = np.ones(len(atom_names), dtype=bool)

    for i, element in enumerate(elements):
        if element != "H":
            continue

        same_residue = np.asarray(residue_ids) == residue_ids[i]
        same_residue[i] = False
        candidate_indices = np.where(same_residue)[0]
        if candidate_indices.size == 0:
            continue

        deltas = coords[candidate_indices] - coords[i]
        distances = np.linalg.norm(deltas, axis=1)
        nearest_local = int(np.argmin(distances))
        nearest_index = int(candidate_indices[nearest_local])
        nearest_distance = float(distances[nearest_local])

        if nearest_distance <= distance_cutoff and elements[nearest_index] == "C":
            keep_mask[i] = False

    return keep_mask


@dataclass
class NormalizationStats:
    x_mean: np.ndarray
    x_std: np.ndarray
    target_mean: float
    target_std: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "x_mean": self.x_mean.tolist(),
            "x_std": self.x_std.tolist(),
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "NormalizationStats":
        return cls(
            x_mean=np.asarray(payload["x_mean"], dtype=np.float32),
            x_std=np.asarray(payload["x_std"], dtype=np.float32),
            target_mean=float(payload["target_mean"]),
            target_std=float(payload["target_std"]),
        )


class PepDynLMDB:
    _instances: Dict[str, "PepDynLMDB"] = {}

    def __new__(cls, lmdb_path: str | Path):
        resolved = str(Path(lmdb_path).resolve())
        if resolved in cls._instances:
            return cls._instances[resolved]
        instance = super().__new__(cls)
        cls._instances[resolved] = instance
        return instance

    def __init__(self, lmdb_path: str | Path):
        self.lmdb_path = Path(lmdb_path)
        if hasattr(self, "env"):
            return
        self.env = lmdb.open(
            str(self.lmdb_path),
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin() as txn:
            self.keys: List[str] = pickle.loads(txn.get(b"__keys__"))

    def get(self, sample_id: str) -> dict:
        with self.env.begin() as txn:
            raw = txn.get(sample_id.encode())
        if raw is None:
            raise KeyError(sample_id)
        return pickle.loads(raw)

    def sample_ids(self) -> List[str]:
        return list(self.keys)


def summarize_sample(sample: dict) -> Dict[str, float]:
    metadata = sample["metadata"]
    mmgbsa_series = np.asarray(sample["frame_features"]["mmgbsa"], dtype=np.float32)
    return {
        "sample_id": metadata["sample_id"],
        "pdbid": metadata["pdbid"],
        "n_atoms": int(metadata["n_atoms"]),
        "n_frames": int(metadata["n_frames"]),
        "mmgbsa_frame0": float(mmgbsa_series[0]),
        "mmgbsa_mean": float(mmgbsa_series.mean()),
    }


def read_key_file(path: str | Path) -> List[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def make_split_table_from_key_files(
    lmdb_dataset: PepDynLMDB,
    train_keys_path: str | Path,
    test_keys_path: str | Path,
    val_ratio_from_train: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    train_keys_all = read_key_file(train_keys_path)
    test_keys = read_key_file(test_keys_path)

    rows = [summarize_sample(lmdb_dataset.get(sample_id)) for sample_id in tqdm(train_keys_all)]
    train_df_full = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)

    if val_ratio_from_train > 0.0:
        n_bins = min(8, max(2, len(train_df_full) // 200))
        train_df_full["strata"] = pd.qcut(
            train_df_full["mmgbsa_mean"], q=n_bins, labels=False, duplicates="drop"
        )
        strata_counts = Counter(train_df_full["strata"].tolist())
        stratify = train_df_full["strata"] if min(strata_counts.values()) >= 2 else None
        train_df, val_df = train_test_split(
            train_df_full,
            test_size=val_ratio_from_train,
            random_state=seed,
            stratify=stratify,
        )
        train_df = train_df.drop(columns=["strata"], errors="ignore")
        val_df = val_df.drop(columns=["strata"], errors="ignore")
    else:
        train_df = train_df_full.drop(columns=["strata"], errors="ignore")
        val_df = train_df_full.iloc[0:0].copy()

    test_rows = [summarize_sample(lmdb_dataset.get(sample_id)) for sample_id in test_keys]
    test_df = pd.DataFrame(test_rows).sort_values("sample_id").reset_index(drop=True)

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    return pd.concat([train_df, val_df, test_df], ignore_index=True).sort_values(["split", "sample_id"]).reset_index(
        drop=True
    )


def make_split_table(
    lmdb_dataset: PepDynLMDB,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train/val/test ratios must sum to 1.")

    rows = [summarize_sample(lmdb_dataset.get(sample_id)) for sample_id in lmdb_dataset.sample_ids()]
    df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)

    n_bins = min(4, max(2, len(df) // 8))
    df["strata"] = pd.qcut(df["mmgbsa_mean"], q=n_bins, labels=False, duplicates="drop")
    strata_counts = Counter(df["strata"].tolist())
    stratify_full = df["strata"] if min(strata_counts.values()) >= 2 else None

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=stratify_full,
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    temp_counts = Counter(temp_df["strata"].tolist())
    stratify_temp = temp_df["strata"] if temp_counts and min(temp_counts.values()) >= 2 else None
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=seed,
        stratify=stratify_temp,
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    return (
        pd.concat([train_df, val_df, test_df], ignore_index=True)
        .sort_values("sample_id")
        .drop(columns=["strata"])
        .reset_index(drop=True)
    )


def build_atom_graph(
    sample: dict,
    frame_idx: int,
    k_neighbors: int = 12,
    target_key: str = "rmsf",
    prune_c_hydrogens: bool = False,
) -> Data:
    metadata = sample["metadata"]
    coords = np.asarray(sample["coords"], dtype=np.float32)
    frame_coords = coords[frame_idx]
    atom_names = metadata["atom_names"]
    residue_names = metadata["residue_names"]
    residue_ids = np.asarray(metadata["residue_ids"])
    peptide_start = int(metadata["peptide_start_index"])
    keep_mask = np.ones(frame_coords.shape[0], dtype=bool)
    if prune_c_hydrogens:
        keep_mask = prune_carbon_bound_hydrogens(frame_coords, atom_names, residue_ids)

    frame_coords = frame_coords[keep_mask]
    centered_coords = frame_coords - frame_coords.mean(axis=0, keepdims=True)
    atom_names = [name for idx, name in enumerate(atom_names) if keep_mask[idx]]
    residue_names = [name for idx, name in enumerate(residue_names) if keep_mask[idx]]
    residue_ids = residue_ids[keep_mask]
    n_atoms = frame_coords.shape[0]

    element_indices = np.asarray(
        [ELEMENT_TO_INDEX[infer_element(name)] for name in atom_names], dtype=np.int64
    )
    residue_indices = np.asarray(
        [RESIDUE_TO_INDEX.get(name, len(AMINO_ACIDS)) for name in residue_names], dtype=np.int64
    )

    element_oh = one_hot(element_indices, len(ELEMENTS))
    residue_oh = one_hot(residue_indices, len(AMINO_ACIDS) + 1)
    original_indices = np.arange(len(keep_mask))[keep_mask]
    is_peptide = (original_indices >= peptide_start).astype(np.float32).reshape(-1, 1)

    x = np.concatenate([element_oh, residue_oh, is_peptide, centered_coords], axis=1)

    pos = torch.from_numpy(centered_coords)
    edge_index = knn_graph(pos, k=min(k_neighbors, max(n_atoms - 1, 1)), loop=False)
    row, col = edge_index
    distances = torch.norm(pos[row] - pos[col], dim=-1)
    edge_weight = 1.0 / (1.0 + distances)

    data = Data(
        x=torch.from_numpy(x).float(),
        pos=pos.float(),
        edge_index=edge_index.long(),
        edge_weight=edge_weight.float(),
        is_peptide=torch.from_numpy(is_peptide.squeeze(-1)).float(),
        sample_id=metadata["sample_id"],
        pdbid=metadata["pdbid"],
        frame_idx=int(frame_idx),
        n_atoms=n_atoms,
    )

    if target_key == "rmsf":
        data.y = torch.from_numpy(np.asarray(sample["atom_rmsf"], dtype=np.float32)[keep_mask]).float()
    elif target_key == "mmgbsa":
        data.y = torch.tensor(
            [float(np.asarray(sample["frame_features"]["mmgbsa"], dtype=np.float32)[frame_idx])],
            dtype=torch.float32,
        )
    else:
        raise ValueError(f"Unknown target_key: {target_key}")

    return data


def compute_normalization_stats(graphs: Iterable[Data]) -> NormalizationStats:
    xs = []
    targets = []
    for graph in graphs:
        xs.append(graph.x.numpy())
        targets.append(graph.y.numpy())

    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(targets, axis=0)

    x_mean = x_all.mean(axis=0)
    x_std = x_all.std(axis=0)
    x_std[x_std < 1e-6] = 1.0

    target_std = float(y_all.std())
    return NormalizationStats(
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        target_mean=float(y_all.mean()),
        target_std=target_std if target_std > 1e-6 else 1.0,
    )


def compute_normalization_stats_streaming(dataset, max_graphs: int | None = None) -> NormalizationStats:
    x_sum = None
    x_sq_sum = None
    y_sum = 0.0
    y_sq_sum = 0.0
    x_count = 0
    y_count = 0

    n_graphs = len(dataset) if max_graphs is None else min(len(dataset), max_graphs)
    for idx in range(n_graphs):
        graph = dataset.get(idx)
        x = graph.x.numpy()
        y = graph.y.numpy()

        if x_sum is None:
            x_sum = x.sum(axis=0, dtype=np.float64)
            x_sq_sum = np.square(x, dtype=np.float64).sum(axis=0)
        else:
            x_sum += x.sum(axis=0, dtype=np.float64)
            x_sq_sum += np.square(x, dtype=np.float64).sum(axis=0)

        y_sum += float(y.sum(dtype=np.float64))
        y_sq_sum += float(np.square(y, dtype=np.float64).sum())
        x_count += x.shape[0]
        y_count += y.shape[0]

    x_mean = x_sum / x_count
    x_var = np.maximum(x_sq_sum / x_count - np.square(x_mean), 1e-12)
    x_std = np.sqrt(x_var)
    x_std[x_std < 1e-6] = 1.0

    y_mean = y_sum / y_count
    y_var = max(y_sq_sum / y_count - y_mean**2, 1e-12)
    y_std = float(np.sqrt(y_var))

    return NormalizationStats(
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        target_mean=float(y_mean),
        target_std=y_std if y_std > 1e-6 else 1.0,
    )


class RMSFGraphDataset(Dataset):
    def __init__(
        self,
        lmdb_path: str | Path,
        sample_ids: Sequence[str],
        cache_dir: str | Path,
        normalization: NormalizationStats | None = None,
        k_neighbors: int = 12,
        prune_c_hydrogens: bool = False,
    ):
        super().__init__()
        self.lmdb = PepDynLMDB(lmdb_path)
        self.sample_ids = list(sample_ids)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.normalization = normalization
        self.k_neighbors = int(k_neighbors)
        self.prune_c_hydrogens = bool(prune_c_hydrogens)

    def len(self) -> int:
        return len(self.sample_ids)

    def _cache_path(self, sample_id: str) -> Path:
        digest = hashlib.md5(
            f"rmsf_{sample_id}_k{self.k_neighbors}_prune{int(self.prune_c_hydrogens)}".encode()
        ).hexdigest()[:10]
        return self.cache_dir / f"{sample_id}_{digest}.pt"

    def get(self, idx: int) -> Data:
        sample_id = self.sample_ids[idx]
        cache_path = self._cache_path(sample_id)
        if cache_path.exists():
            graph = torch.load(cache_path, weights_only=False)
        else:
            graph = build_atom_graph(
                self.lmdb.get(sample_id),
                frame_idx=0,
                k_neighbors=self.k_neighbors,
                target_key="rmsf",
                prune_c_hydrogens=self.prune_c_hydrogens,
            )
            torch.save(graph, cache_path)

        if self.normalization is not None:
            graph = graph.clone()
            x_mean = torch.from_numpy(self.normalization.x_mean)
            x_std = torch.from_numpy(self.normalization.x_std)
            graph.x = (graph.x - x_mean) / x_std
            graph.y_norm = (graph.y - self.normalization.target_mean) / self.normalization.target_std
        return graph


class MMGBSAGraphDataset(Dataset):
    def __init__(
        self,
        lmdb_path: str | Path,
        entries: Sequence[Tuple[str, int]],
        cache_dir: str | Path,
        normalization: NormalizationStats | None = None,
        k_neighbors: int = 12,
        prune_c_hydrogens: bool = False,
    ):
        super().__init__()
        self.lmdb = PepDynLMDB(lmdb_path)
        self.entries = list(entries)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.normalization = normalization
        self.k_neighbors = int(k_neighbors)
        self.prune_c_hydrogens = bool(prune_c_hydrogens)

    def len(self) -> int:
        return len(self.entries)

    def _cache_path(self, sample_id: str, frame_idx: int) -> Path:
        digest = hashlib.md5(
            f"mmgbsa_{sample_id}_frame{frame_idx}_k{self.k_neighbors}_prune{int(self.prune_c_hydrogens)}".encode()
        ).hexdigest()[:10]
        return self.cache_dir / f"{sample_id}_frame{frame_idx}_{digest}.pt"

    def get(self, idx: int) -> Data:
        sample_id, frame_idx = self.entries[idx]
        cache_path = self._cache_path(sample_id, frame_idx)
        if cache_path.exists():
            graph = torch.load(cache_path, weights_only=False)
        else:
            graph = build_atom_graph(
                self.lmdb.get(sample_id),
                frame_idx=frame_idx,
                k_neighbors=self.k_neighbors,
                target_key="mmgbsa",
                prune_c_hydrogens=self.prune_c_hydrogens,
            )
            torch.save(graph, cache_path)

        if self.normalization is not None:
            graph = graph.clone()
            x_mean = torch.from_numpy(self.normalization.x_mean)
            x_std = torch.from_numpy(self.normalization.x_std)
            graph.x = (graph.x - x_mean) / x_std
            graph.y_norm = (graph.y - self.normalization.target_mean) / self.normalization.target_std
        return graph


def build_mmgbsa_entries(sample_ids: Sequence[str], lmdb_path: str | Path, frame_mode: str) -> List[Tuple[str, int]]:
    lmdb = PepDynLMDB(lmdb_path)
    entries: List[Tuple[str, int]] = []
    for sample_id in tqdm(sample_ids):
        sample = lmdb.get(sample_id)
        n_frames = int(sample["metadata"]["n_frames"])
        if frame_mode == "first":
            entries.append((sample_id, 0))
        elif frame_mode == "all":
            entries.extend((sample_id, frame_idx) for frame_idx in range(n_frames))
        else:
            raise ValueError(f"Unknown frame_mode: {frame_mode}")
    return entries


def save_split_table(split_df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(path, index=False)


def save_normalization(stats: NormalizationStats, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats.to_dict(), indent=2))


def load_normalization(path: str | Path) -> NormalizationStats:
    payload = json.loads(Path(path).read_text())
    return NormalizationStats.from_dict(payload)
