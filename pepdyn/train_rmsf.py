from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

from .data import (
    PepDynLMDB,
    RMSFGraphDataset,
    compute_normalization_stats_streaming,
    make_split_table,
    make_split_table_from_key_files,
    save_normalization,
    save_split_table,
)
from .metrics import regression_metrics
from .model import RMSFGCN
from .plotting import plot_dataset_overview, plot_parity, plot_training_curve, set_plot_style
from .train import denormalize, load_config, pick_device, save_json, set_seed


def run_epoch(model, loader, optimizer, device, epoch: int, total_epochs: int):
    model.train()
    total_loss = 0.0
    total_graphs = 0
    criterion = nn.MSELoss()

    progress = tqdm(loader, desc=f"RMSF Train {epoch}/{total_epochs}", leave=False)
    for batch in progress:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y_norm)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * batch.num_graphs
        total_graphs += batch.num_graphs
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(total_graphs, 1)


@torch.no_grad()
def evaluate(model, loader, device, stats, desc: str = "RMSF Eval"):
    model.eval()
    true_all = []
    pred_all = []

    # ✅ 新增两个 list
    pearson_list = []
    spearman_list = []

    for batch in tqdm(loader, desc=desc, leave=False):
        batch = batch.to(device)
        pred = model(batch)
        pred = denormalize(pred, stats.target_mean, stats.target_std).cpu().numpy()
        true = batch.y.cpu().numpy()

        true_all.append(true)
        pred_all.append(pred)

        # ✅ 每个 batch 单独算相关系数（不影响原 metrics）
        if len(true) > 1 and np.std(true) > 1e-12 and np.std(pred) > 1e-12:
            pearson_list.append(float(pearsonr(true, pred).statistic))
            spearman_list.append(float(spearmanr(true, pred).statistic))
        else:
            pearson_list.append(float("nan"))
            spearman_list.append(float("nan"))

    y_true = pd.Series(np.concatenate(true_all))
    y_pred = pd.Series(np.concatenate(pred_all))
    metrics = regression_metrics(y_true, y_pred)

    pred_df = pd.DataFrame({"true_rmsf": y_true, "pred_rmsf": y_pred})

    # ✅ 只是在返回时多加两个 list
    return metrics, pred_df, pearson_list, spearman_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["seed"]))

    output_dir = Path(config["output_dir"])
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    lmdb = PepDynLMDB(config["data"]["lmdb_path"])
    print('lmdb loaded.')
    split_cfg = config["split"]
    if split_cfg.get("train_keys_path") and split_cfg.get("test_keys_path"):
        split_df = make_split_table_from_key_files(
            lmdb_dataset=lmdb,
            train_keys_path=split_cfg["train_keys_path"],
            test_keys_path=split_cfg["test_keys_path"],
            val_ratio_from_train=float(split_cfg.get("val_ratio_from_train", 0.1)),
            seed=int(config["seed"]),
        )
    else:
        split_df = make_split_table(
            lmdb_dataset=lmdb,
            train_ratio=float(split_cfg["train_ratio"]),
            val_ratio=float(split_cfg["val_ratio"]),
            test_ratio=float(split_cfg["test_ratio"]),
            seed=int(config["seed"]),
        )
    save_split_table(split_df, output_dir / "split_table.csv")
    plot_dataset_overview(split_df, figures_dir / "fig_dataset_overview")
    print('Splitting completed.')

    train_ids = split_df.loc[split_df["split"] == "train", "sample_id"].tolist()
    val_ids = split_df.loc[split_df["split"] == "val", "sample_id"].tolist()
    test_ids = split_df.loc[split_df["split"] == "test", "sample_id"].tolist()
    cache_dir = output_dir / "cache"

    prune_c_h = bool(config["data"].get("prune_c_hydrogens", False))
    train_raw = RMSFGraphDataset(
        config["data"]["lmdb_path"],
        train_ids,
        cache_dir / "raw_train",
        None,
        int(config["data"]["k_neighbors"]),
        prune_c_h,
    )
    norm_max_graphs = config["training"].get("normalization_max_graphs")
    stats = compute_normalization_stats_streaming(train_raw, norm_max_graphs)
    save_normalization(stats, output_dir / "normalization.json")
    print('Normalization stats computed and saved.')
    datasets = {
        split: RMSFGraphDataset(
            config["data"]["lmdb_path"],
            ids,
            cache_dir / split,
            stats,
            int(config["data"]["k_neighbors"]),
            prune_c_h,
        )
        for split, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]
    }
    print('Datasets created.')
    train_loader = DataLoader(datasets["train"], batch_size=int(config["training"]["batch_size"]), shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=int(config["training"]["eval_batch_size"]), shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=int(config["training"]["eval_batch_size"]), shuffle=False)

    sample_graph = datasets["train"].get(0)
    device = pick_device(config["training"]["device"])
    model = RMSFGCN(
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

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, epoch, int(config["training"]["epochs"]))
        val_metrics, _, _, _ = evaluate(model, val_loader, device, stats, desc=f"RMSF Val {epoch}/{int(config['training']['epochs'])}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_rmse": val_metrics["rmse"]})
        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(best_state, output_dir / "best_model.pt")
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "history.csv", index=False)
    plot_training_curve(history_df, "val_rmse", "Validation RMSE", figures_dir / "fig_training_curve")

    test_metrics, pred_df, pearson_list, spearman_list = evaluate(model, test_loader, device, stats, desc="RMSF Test")
    pred_df.to_csv(output_dir / "pred_rmsf_atom_level.csv", index=False)
    plot_parity(
        pred_df.sample(n=min(5000, len(pred_df)), random_state=int(config["seed"])),
        x="true_rmsf",
        y="pred_rmsf",
        xlabel="Reference atom RMSF (A)",
        ylabel="Predicted atom RMSF (A)",
        title="First-frame to atom RMSF parity",
        output_path=figures_dir / "fig_parity_rmsf",
    )


    def plot_correlation_hist(corr_list, corrtype, title="Correlation Distribution", bins=20):
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        # 👉 风格设置（核心）
        sns.set_style("white")  # 干净背景
        plt.rcParams.update({
            "font.size": 11,
            "axes.linewidth": 1.0,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        })

        # 数据处理
        corr = np.array(corr_list, dtype=float)
        corr = corr[~np.isnan(corr)]
        mean_val = np.mean(corr)

        # 画图
        plt.figure(figsize=(5.5, 4))

        # ✅ 直方图 + KDE
        sns.histplot(
            corr,
            bins=bins,
            stat="density",
            color="#936AB1",
            alpha=0.85,
            edgecolor=None
        )

        sns.kdeplot(
            corr,
            color="#4B2E83",   # 深一点的紫色
            linewidth=2
        )

        # ✅ 均值线
        plt.axvline(mean_val, color="black", linestyle="--", linewidth=1)

        # 标签
        plt.xlabel(f"Correlation ({corrtype})")
        plt.ylabel("Probability density")
        plt.title(f"{title} (mean = {mean_val:.2f})", pad=10)

        # ✅ 去掉上右边框（关键！）
        sns.despine()

        plt.tight_layout()

        plt.savefig(
            figures_dir / f"fig_{corrtype.lower().replace(' ', '_')}_hist.pdf",
            format="pdf",
            dpi=300
        )

        plt.show()

        return mean_val
    
    pearson_mean = plot_correlation_hist(pearson_list, "Pearson", title="Pearson Correlation")
    spearman_mean = plot_correlation_hist(spearman_list, "Spearman", title="Spearman Correlation")
    print(f"Mean Pearson correlation across structures: {pearson_mean:.4f}")
    print(f"Mean Spearman correlation across structures: {spearman_mean:.4f}")

    metrics = {
        "config": config,
        "dataset": {
            "num_train_samples": len(train_ids),
            "num_val_samples": len(val_ids),
            "num_test_samples": len(test_ids),
            "input_definition": "frame_0_only",
            "target_definition": "atom_rmsf_over_trajectory",
            "prune_c_hydrogens": prune_c_h,
        },
        "rmsf_test": test_metrics,
        "correlation_test": {
            "pearson_mean": pearson_mean,
            "spearman_mean": spearman_mean,
        },
    }
    save_json(metrics, output_dir / "metrics.json")
    print(pd.Series(metrics["rmsf_test"]).to_json(indent=2))


if __name__ == "__main__":
    main()
