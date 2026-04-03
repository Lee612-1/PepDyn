from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch_geometric.loader import DataLoader

from .data import (
    MMGBSAGraphDataset,
    RMSFGraphDataset,
    PepDynLMDB,
    build_mmgbsa_entries,
    compute_normalization_stats,
    make_split_table,
    make_split_table_from_key_files,
)
from .model import MMGBSAGCN, RMSFGCN
from .train import load_config, pick_device, set_seed


def resolve_split_df(config, lmdb):
    split_cfg = config["split"]
    if split_cfg.get("train_keys_path") and split_cfg.get("test_keys_path"):
        return make_split_table_from_key_files(
            lmdb_dataset=lmdb,
            train_keys_path=split_cfg["train_keys_path"],
            test_keys_path=split_cfg["test_keys_path"],
            val_ratio_from_train=float(split_cfg.get("val_ratio_from_train", 0.1)),
            seed=int(config["seed"]),
        )
    return make_split_table(
        lmdb_dataset=lmdb,
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        seed=int(config["seed"]),
    )


def benchmark_train_loop(model, loader, device, task: str, max_steps: int) -> float:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    times = []

    for step, batch in enumerate(loader):
        if step >= max_steps:
            break
        t0 = time.perf_counter()
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        if task == "rmsf":
            loss = criterion(pred, batch.y_norm)
        else:
            loss = criterion(pred, batch.y_norm.view(-1))
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize() if device.type == "cuda" else None
        times.append(time.perf_counter() - t0)
    return sum(times) / max(len(times), 1)


def build_rmsf_loader(config, split_df, prune_c_hydrogens: bool, cache_dir: Path):
    train_ids = split_df.loc[split_df["split"] == "train", "sample_id"].tolist()
    raw_dataset = RMSFGraphDataset(
        config["data"]["lmdb_path"],
        train_ids,
        cache_dir / "raw_train",
        None,
        int(config["data"]["k_neighbors"]),
        prune_c_hydrogens,
    )
    stats = compute_normalization_stats([raw_dataset.get(i) for i in range(min(len(raw_dataset), 8))])
    dataset = RMSFGraphDataset(
        config["data"]["lmdb_path"],
        train_ids,
        cache_dir / "train",
        stats,
        int(config["data"]["k_neighbors"]),
        prune_c_hydrogens,
    )
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False)
    return loader, len(train_ids)


def build_mmgbsa_loader(config, split_df, frame_mode: str, prune_c_hydrogens: bool, cache_dir: Path):
    train_ids = split_df.loc[split_df["split"] == "train", "sample_id"].tolist()
    train_entries = build_mmgbsa_entries(train_ids, config["data"]["lmdb_path"], frame_mode=frame_mode)
    raw_dataset = MMGBSAGraphDataset(
        config["data"]["lmdb_path"],
        train_entries,
        cache_dir / "raw_train",
        None,
        int(config["data"]["k_neighbors"]),
        prune_c_hydrogens,
    )
    stats = compute_normalization_stats([raw_dataset.get(i) for i in range(min(len(raw_dataset), 8))])
    dataset = MMGBSAGraphDataset(
        config["data"]["lmdb_path"],
        train_entries,
        cache_dir / "train",
        stats,
        int(config["data"]["k_neighbors"]),
        prune_c_hydrogens,
    )
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False)
    return loader, len(train_entries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rmsf-config", type=str, required=True)
    parser.add_argument("--mmgbsa-config", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=5)
    args = parser.parse_args()

    rmsf_config = load_config(args.rmsf_config)
    mmgbsa_config = load_config(args.mmgbsa_config)
    set_seed(int(rmsf_config["seed"]))

    lmdb = PepDynLMDB(rmsf_config["data"]["lmdb_path"])
    split_df = resolve_split_df(rmsf_config, lmdb)
    device = pick_device(rmsf_config["training"]["device"])
    benchmark_root = Path(mmgbsa_config["output_dir"]).parent / "runtime_benchmark"
    benchmark_root.mkdir(parents=True, exist_ok=True)

    rows = []

    for prune_flag in [False, True]:
        rmsf_loader, rmsf_graphs = build_rmsf_loader(rmsf_config, split_df, prune_flag, benchmark_root / f"rmsf_prune_{int(prune_flag)}")
        rmsf_graph = next(iter(rmsf_loader))
        rmsf_model = RMSFGCN(
            in_channels=rmsf_graph.x.shape[1],
            hidden_dim=int(rmsf_config["model"]["hidden_dim"]),
            num_layers=int(rmsf_config["model"]["num_layers"]),
            dropout=float(rmsf_config["model"]["dropout"]),
        ).to(device)
        rmsf_step_sec = benchmark_train_loop(rmsf_model, rmsf_loader, device, "rmsf", args.max_steps)
        rows.append(
            {
                "task": "rmsf",
                "setting": "first_frame",
                "prune_c_hydrogens": prune_flag,
                "seconds_per_step": rmsf_step_sec,
                "train_graphs_per_epoch": rmsf_graphs,
                "estimated_epoch_hours": rmsf_step_sec * rmsf_graphs / 3600.0,
                "estimated_total_hours": rmsf_step_sec * rmsf_graphs * int(rmsf_config["training"]["epochs"]) / 3600.0,
            }
        )

        for setting, frame_mode in [("mmgbsa_first_same_epoch", "first"), ("mmgbsa_all_same_epoch", "all")]:
            mmgbsa_loader, mmgbsa_graphs = build_mmgbsa_loader(
                mmgbsa_config,
                split_df,
                frame_mode,
                prune_flag,
                benchmark_root / f"{setting}_prune_{int(prune_flag)}",
            )
            mmgbsa_graph = next(iter(mmgbsa_loader))
            mmgbsa_model = MMGBSAGCN(
                in_channels=mmgbsa_graph.x.shape[1],
                hidden_dim=int(mmgbsa_config["model"]["hidden_dim"]),
                num_layers=int(mmgbsa_config["model"]["num_layers"]),
                dropout=float(mmgbsa_config["model"]["dropout"]),
            ).to(device)
            step_sec = benchmark_train_loop(mmgbsa_model, mmgbsa_loader, device, "mmgbsa", args.max_steps)
            epochs = int(mmgbsa_config["training"]["epochs"])
            rows.append(
                {
                    "task": "mmgbsa",
                    "setting": setting,
                    "prune_c_hydrogens": prune_flag,
                    "seconds_per_step": step_sec,
                    "train_graphs_per_epoch": mmgbsa_graphs,
                    "estimated_epoch_hours": step_sec * mmgbsa_graphs / 3600.0,
                    "estimated_total_hours": step_sec * mmgbsa_graphs * epochs / 3600.0,
                }
            )

        first_graphs = next(row["train_graphs_per_epoch"] for row in rows if row["setting"] == "mmgbsa_first_same_epoch" and row["prune_c_hydrogens"] == prune_flag)
        all_graphs = next(row["train_graphs_per_epoch"] for row in rows if row["setting"] == "mmgbsa_all_same_epoch" and row["prune_c_hydrogens"] == prune_flag)
        base_budget = first_graphs * int(mmgbsa_config["training"]["epochs"])
        matched_epochs = max(1, math.ceil(base_budget / all_graphs))
        all_step_sec = next(row["seconds_per_step"] for row in rows if row["setting"] == "mmgbsa_all_same_epoch" and row["prune_c_hydrogens"] == prune_flag)
        rows.append(
            {
                "task": "mmgbsa",
                "setting": "mmgbsa_all_same_graph_budget",
                "prune_c_hydrogens": prune_flag,
                "seconds_per_step": all_step_sec,
                "train_graphs_per_epoch": all_graphs,
                "estimated_epoch_hours": all_step_sec * all_graphs / 3600.0,
                "estimated_total_hours": all_step_sec * all_graphs * matched_epochs / 3600.0,
                "matched_epochs": matched_epochs,
                "matched_total_graph_budget": base_budget,
            }
        )

    results = pd.DataFrame(rows)
    out_csv = benchmark_root / "runtime_estimates.csv"
    results.to_csv(out_csv, index=False)
    print(results.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
