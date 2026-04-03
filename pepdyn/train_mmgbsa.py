from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch import nn
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from .data import (
    MMGBSAGraphDataset,
    PepDynLMDB,
    build_mmgbsa_entries,
    compute_normalization_stats_streaming,
    make_split_table,
    make_split_table_from_key_files,
    save_normalization,
    save_split_table,
)
from .metrics import regression_metrics
from .model import MMGBSAGCN
from .plotting import (
    plot_dataset_overview,
    plot_mmgbsa_error,
    plot_mmgbsa_regime_comparison,
    plot_parity,
    plot_training_curve,
    set_plot_style,
)
from .train import denormalize, load_config, pick_device, save_json, set_seed


def run_epoch(model, loader, optimizer, device, desc: str):
    model.train()
    total_loss = 0.0
    total_graphs = 0
    criterion = nn.MSELoss()

    progress = tqdm(loader, desc=desc, leave=False)
    for batch in progress:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y_norm.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * batch.num_graphs
        total_graphs += batch.num_graphs
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(total_graphs, 1)


@torch.no_grad()
def evaluate(model, loader, device, stats, desc: str):
    model.eval()
    rows = []
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = batch.to(device)
        pred = model(batch)
        pred = denormalize(pred, stats.target_mean, stats.target_std).cpu().numpy()
        true = batch.y.view(-1).cpu().numpy()
        for sid, pdbid, frame_idx, pred_value, true_value in zip(
            list(batch.sample_id),
            list(batch.pdbid),
            batch.frame_idx.cpu().numpy().tolist(),
            pred.tolist(),
            true.tolist(),
        ):
            rows.append(
                {
                    "sample_id": sid,
                    "pdbid": pdbid,
                    "frame_idx": int(frame_idx),
                    "pred_mmgbsa": float(pred_value),
                    "true_mmgbsa": float(true_value),
                }
            )
    pred_df = pd.DataFrame(rows)
    metrics = regression_metrics(pred_df["true_mmgbsa"], pred_df["pred_mmgbsa"])
    return metrics, pred_df


def compute_structure_frame_correlations(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for sid, group in pred_df.groupby("sample_id"):
        group = group.sort_values("frame_idx").reset_index(drop=True)

        true_vals = group["true_mmgbsa"].to_numpy(dtype=float)
        pred_vals = group["pred_mmgbsa"].to_numpy(dtype=float)

        if len(group) < 2:
            pearson_val = np.nan
            spearman_val = np.nan
        else:
            true_constant = np.allclose(true_vals, true_vals[0])
            pred_constant = np.allclose(pred_vals, pred_vals[0])

            if true_constant or pred_constant:
                pearson_val = np.nan
                spearman_val = np.nan
            else:
                try:
                    pearson_val = pearsonr(true_vals, pred_vals)[0]
                except Exception:
                    pearson_val = np.nan
                try:
                    spearman_val = spearmanr(true_vals, pred_vals)[0]
                except Exception:
                    spearman_val = np.nan

        rows.append(
            {
                "sample_id": sid,
                "pdbid": group["pdbid"].iloc[0],
                "n_frames": int(len(group)),
                "pearson": float(pearson_val) if not np.isnan(pearson_val) else np.nan,
                "spearman": float(spearman_val) if not np.isnan(spearman_val) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def plot_correlation_hist(
    corr_list,
    corrtype,
    output_path: Path,
    title="Correlation Distribution",
    bins=20,
):
    corr = np.array(corr_list, dtype=float)
    corr = corr[~np.isnan(corr)]

    if len(corr) == 0:
        print(f"No valid {corrtype} values; skip histogram.")
        return np.nan

    sns.set_style("white")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.linewidth": 1.0,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    mean_val = np.mean(corr)

    plt.figure(figsize=(5.5, 4))
    sns.histplot(
        corr,
        bins=bins,
        stat="density",
        color="#936AB1",
        alpha=0.85,
        edgecolor=None,
    )
    sns.kdeplot(
        corr,
        color="#4B2E83",
        linewidth=2,
    )

    plt.axvline(mean_val, color="black", linestyle="--", linewidth=1)

    plt.xlabel(f"Correlation ({corrtype})")
    plt.ylabel("Probability density")
    plt.title(f"{title} (mean = {mean_val:.2f})", pad=10)

    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", dpi=300)
    plt.close()

    return float(mean_val)


def plot_best_structure_scatter(
    pred_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    output_path: Path,
):
    valid_corr = corr_df.dropna(subset=["pearson", "spearman"]).copy()
    if valid_corr.empty:
        print("No valid per-structure correlation found; skip best-structure scatter plot.")
        return

    best_row = valid_corr.sort_values("pearson", ascending=False).iloc[0]
    best_sid = best_row["sample_id"]

    best_df = (
        pred_df.loc[pred_df["sample_id"] == best_sid]
        .sort_values("frame_idx")
        .reset_index(drop=True)
    )

    sns.set_style("white")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.linewidth": 1.0,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    plt.figure(figsize=(5.2, 4.4))
    sns.scatterplot(
        data=best_df,
        x="true_mmgbsa",
        y="pred_mmgbsa",
        s=45,
        alpha=0.85,
        edgecolor=None,
    )

    x_min = min(best_df["true_mmgbsa"].min(), best_df["pred_mmgbsa"].min())
    x_max = max(best_df["true_mmgbsa"].max(), best_df["pred_mmgbsa"].max())
    plt.plot([x_min, x_max], [x_min, x_max], linestyle="--", linewidth=1, color="black")

    plt.xlabel("True MM/GBSA (kcal/mol)")
    plt.ylabel("Predicted MM/GBSA (kcal/mol)")
    plt.title(
        f"Best structure: {best_row['pdbid']} | "
        f"Pearson={best_row['pearson']:.2f}, "
        f"Spearman={best_row['spearman']:.2f}",
        pad=10,
    )

    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", dpi=300)
    plt.close()


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


def train_one_setting(
    config,
    split_df: pd.DataFrame,
    output_dir: Path,
    setting_name: str,
    frame_mode: str,
    epochs: int,
):
    train_ids = split_df.loc[split_df["split"] == "train", "sample_id"].tolist()
    val_ids = split_df.loc[split_df["split"] == "val", "sample_id"].tolist()
    test_ids = split_df.loc[split_df["split"] == "test", "sample_id"].tolist()
    prune_c_h = bool(config["data"].get("prune_c_hydrogens", False))

    train_entries = build_mmgbsa_entries(
        train_ids,
        config["data"]["lmdb_path"],
        frame_mode=frame_mode,
    )
    val_entries = build_mmgbsa_entries(
        val_ids,
        config["data"]["lmdb_path"],
        frame_mode="first",
    )
    test_entries = build_mmgbsa_entries(
        test_ids,
        config["data"]["lmdb_path"],
        frame_mode="all",
    )

    cache_dir = output_dir / "cache"
    train_raw = MMGBSAGraphDataset(
        config["data"]["lmdb_path"],
        train_entries,
        cache_dir / "raw_train",
        None,
        int(config["data"]["k_neighbors"]),
        prune_c_h,
    )
    norm_max_graphs = config["training"].get("normalization_max_graphs")
    stats = compute_normalization_stats_streaming(train_raw, norm_max_graphs)
    save_normalization(stats, output_dir / "normalization.json")

    datasets = {
        "train": MMGBSAGraphDataset(
            config["data"]["lmdb_path"],
            train_entries,
            cache_dir / "train",
            stats,
            int(config["data"]["k_neighbors"]),
            prune_c_h,
        ),
        "val": MMGBSAGraphDataset(
            config["data"]["lmdb_path"],
            val_entries,
            cache_dir / "val",
            stats,
            int(config["data"]["k_neighbors"]),
            prune_c_h,
        ),
        "test": MMGBSAGraphDataset(
            config["data"]["lmdb_path"],
            test_entries,
            cache_dir / "test",
            stats,
            int(config["data"]["k_neighbors"]),
            prune_c_h,
        ),
    }

    train_loader = DataLoader(
        datasets["train"],
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 0)),
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=int(config["training"]["eval_batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=int(config["training"]["eval_batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
    )

    sample_graph = datasets["train"].get(0)
    device = pick_device(config["training"]["device"])
    model = MMGBSAGCN(
        in_channels=sample_graph.x.shape[1],
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_layers=int(config["model"]["num_layers"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    best_state = None
    best_rmse = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            desc=f"{setting_name} Train {epoch}/{epochs}",
        )
        val_metrics, _ = evaluate(
            model,
            val_loader,
            device,
            stats,
            desc=f"{setting_name} Val {epoch}/{epochs}",
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_rmse": val_metrics["rmse"],
            }
        )
        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save(best_state, output_dir / "best_model.pt")

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "history.csv", index=False)
    plot_training_curve(
        history_df,
        "val_rmse",
        "Validation RMSE",
        output_dir / "figures" / "fig_training_curve",
    )

    test_metrics, pred_df = evaluate(
        model,
        test_loader,
        device,
        stats,
        desc=f"{setting_name} Test",
    )
    pred_df.to_csv(output_dir / "pred_mmgbsa_all_frames.csv", index=False)

    plot_parity(
        pred_df,
        x="true_mmgbsa",
        y="pred_mmgbsa",
        xlabel="Reference MM/GBSA (kcal/mol)",
        ylabel="Predicted MM/GBSA (kcal/mol)",
        title=f"{setting_name} (all test frames)",
        output_path=output_dir / "figures" / "fig_parity_mmgbsa_all_frames",
    )
    plot_mmgbsa_error(
        pred_df,
        output_dir / "figures" / "fig_mmgbsa_abs_error_all_frames",
    )

    corr_df = compute_structure_frame_correlations(pred_df)
    corr_df.to_csv(output_dir / "structure_frame_correlations.csv", index=False)

    mean_pearson = plot_correlation_hist(
        corr_df["pearson"].tolist(),
        corrtype="Pearson",
        title="Per-structure Pearson distribution",
        bins=20,
        output_path=output_dir / "figures" / "fig_pearson_hist.pdf",
    )

    mean_spearman = plot_correlation_hist(
        corr_df["spearman"].tolist(),
        corrtype="Spearman",
        title="Per-structure Spearman distribution",
        bins=20,
        output_path=output_dir / "figures" / "fig_spearman_hist.pdf",
    )

    plot_best_structure_scatter(
        pred_df=pred_df,
        corr_df=corr_df,
        output_path=output_dir / "figures" / "fig_best_structure_true_vs_pred.pdf",
    )

    total_graphs_seen = len(train_entries) * epochs

    pearson_valid = corr_df["pearson"].dropna()
    spearman_valid = corr_df["spearman"].dropna()

    metrics = {
        "setting": setting_name,
        "frame_mode": frame_mode,
        "epochs": epochs,
        "num_train_graphs_per_epoch": len(train_entries),
        "num_val_graphs": len(val_entries),
        "num_test_graphs": len(test_entries),
        "total_train_graphs_seen": total_graphs_seen,
        "prune_c_hydrogens": prune_c_h,
        "test_metrics": test_metrics,
        "per_structure_correlation": {
            "num_structures": int(len(corr_df)),
            "num_valid_pearson": int(pearson_valid.shape[0]),
            "num_valid_spearman": int(spearman_valid.shape[0]),
            "mean_pearson": None if pd.isna(mean_pearson) else float(mean_pearson),
            "mean_spearman": None if pd.isna(mean_spearman) else float(mean_spearman),
            "median_pearson": None if pearson_valid.empty else float(pearson_valid.median()),
            "median_spearman": None if spearman_valid.empty else float(spearman_valid.median()),
            "max_pearson": None if pearson_valid.empty else float(pearson_valid.max()),
            "max_spearman": None if spearman_valid.empty else float(spearman_valid.max()),
        },
    }
    save_json(metrics, output_dir / "metrics.json")
    return metrics


def build_setting_plan(config, split_df: pd.DataFrame):
    train_ids = split_df.loc[split_df["split"] == "train", "sample_id"].tolist()
    base_epochs = int(config["training"]["epochs"])

    full_plan = [
        {
            "setting": "first_same_epoch",
            "frame_mode": "first",
            "epochs": base_epochs,
            "comparison_group": "same_epoch",
        },
        {
            "setting": "all_same_epoch",
            "frame_mode": "all",
            "epochs": base_epochs,
            "comparison_group": "same_epoch",
        },
    ]

    regimes = config.get("regimes", ["all"])
    if regimes == ["first"]:
        return [item for item in full_plan if item["setting"] == "first_same_epoch"]
    if regimes == ["all"]:
        return [item for item in full_plan if item["setting"] in {"all_same_epoch"}]
    return full_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["seed"]))

    root_output_dir = Path(config["output_dir"])
    root_output_dir.mkdir(parents=True, exist_ok=True)
    (root_output_dir / "figures").mkdir(parents=True, exist_ok=True)
    set_plot_style()

    lmdb = PepDynLMDB(config["data"]["lmdb_path"])
    split_df = resolve_split_df(config, lmdb)
    save_split_table(split_df, root_output_dir / "split_table.csv")
    plot_dataset_overview(split_df, root_output_dir / "figures" / "fig_dataset_overview")

    setting_plan = build_setting_plan(config, split_df)
    summary_rows = []
    for item in setting_plan:
        setting_output_dir = root_output_dir / item["setting"]
        (setting_output_dir / "figures").mkdir(parents=True, exist_ok=True)
        metrics = train_one_setting(
            config=config,
            split_df=split_df,
            output_dir=setting_output_dir,
            setting_name=item["setting"],
            frame_mode=item["frame_mode"],
            epochs=int(item["epochs"]),
        )
        summary_rows.append(
            {
                "setting": item["setting"],
                "comparison_group": item["comparison_group"],
                "frame_mode": item["frame_mode"],
                "epochs": item["epochs"],
                "num_train_graphs_per_epoch": metrics["num_train_graphs_per_epoch"],
                "total_train_graphs_seen": metrics["total_train_graphs_seen"],
                **metrics["test_metrics"],
                "mean_pearson": metrics["per_structure_correlation"]["mean_pearson"],
                "mean_spearman": metrics["per_structure_correlation"]["mean_spearman"],
                "median_pearson": metrics["per_structure_correlation"]["median_pearson"],
                "median_spearman": metrics["per_structure_correlation"]["median_spearman"],
                "max_pearson": metrics["per_structure_correlation"]["max_pearson"],
                "max_spearman": metrics["per_structure_correlation"]["max_spearman"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(root_output_dir / "mmgbsa_setting_summary.csv", index=False)
    plot_mmgbsa_regime_comparison(
        summary_df,
        root_output_dir / "figures" / "fig_mmgbsa_setting_comparison",
        label_col="setting",
    )
    save_json({"settings": summary_rows}, root_output_dir / "comparison_metrics.json")
    print(summary_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()