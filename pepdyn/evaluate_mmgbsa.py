from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from .data import (
    MMGBSAGraphDataset,
    PepDynLMDB,
    build_mmgbsa_entries,
    make_split_table,
    make_split_table_from_key_files,
)
from .model import MMGBSAGCN
from .plotting import set_plot_style
from .train import denormalize, load_config, pick_device, set_seed


# ----------------------------
# utils
# ----------------------------
class NormalizationStats:
    def __init__(self, target_mean: float, target_std: float):
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)


def load_normalization_stats(path: Path) -> NormalizationStats:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return NormalizationStats(
        target_mean=data["target_mean"],
        target_std=data["target_std"],
    )


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


@torch.no_grad()
def evaluate_all_frames(model, loader, device, stats, desc: str):
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

    return pd.DataFrame(rows)


def safe_corr(x, y, method: str):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    # 常数数组时 scipy 会 warning，这里直接返回 nan
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan

    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    if method == "spearman":
        return float(spearmanr(x, y).statistic)
    raise ValueError(f"Unknown method: {method}")


def compute_per_structure_correlations(pred_df: pd.DataFrame, group_col: str = "sample_id") -> pd.DataFrame:
    rows = []

    for gid, g in pred_df.groupby(group_col):
        g = g.sort_values("frame_idx").reset_index(drop=True)

        pearson = safe_corr(g["true_mmgbsa"], g["pred_mmgbsa"], method="pearson")
        spearman = safe_corr(g["true_mmgbsa"], g["pred_mmgbsa"], method="spearman")

        row = {
            group_col: gid,
            "pdbid": g["pdbid"].iloc[0] if "pdbid" in g.columns and len(g) > 0 else None,
            "n_frames": len(g),
            "pearson": pearson,
            "spearman": spearman,
            "mean_true": float(g["true_mmgbsa"].mean()),
            "mean_pred": float(g["pred_mmgbsa"].mean()),
            "mae": float(np.mean(np.abs(g["true_mmgbsa"] - g["pred_mmgbsa"]))),
            "rmse": float(np.sqrt(np.mean((g["true_mmgbsa"] - g["pred_mmgbsa"]) ** 2))),
        }
        rows.append(row)

    corr_df = pd.DataFrame(rows)

    # 排序：先 pearson，再 spearman
    corr_df = corr_df.sort_values(
        by=["pearson", "spearman", "n_frames"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    return corr_df


# ----------------------------
# plotting
# ----------------------------
def plot_correlation_hist(corr_list, corrtype, output_path, title="Correlation Distribution", bins=20):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("white")
    plt.rcParams.update({
        "font.size": 11,
        "axes.linewidth": 1.0,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    corr = np.array(corr_list, dtype=float)
    corr = corr[~np.isnan(corr)]
    mean_val = np.mean(corr) if len(corr) > 0 else np.nan

    plt.figure(figsize=(5.5, 4))

    sns.histplot(
        corr,
        bins=bins,
        stat="density",
        color="#936AB1",
        alpha=0.85,
        edgecolor=None,
    )

    if len(corr) >= 2:
        sns.kdeplot(
            corr,
            color="#4B2E83",
            linewidth=2,
        )

    if np.isfinite(mean_val):
        plt.axvline(mean_val, color="black", linestyle="--", linewidth=1)

    plt.xlabel(f"Correlation ({corrtype})")
    plt.ylabel("Probability density")
    if np.isfinite(mean_val):
        plt.title(f"{title} (mean = {mean_val:.2f})", pad=10)
    else:
        plt.title(title, pad=10)

    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", dpi=300)
    plt.close()

    return mean_val


def plot_best_structure_scatter(best_df: pd.DataFrame, best_row: pd.Series, group_col: str, output_path: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("white")
    plt.rcParams.update({
        "font.size": 11,
        "axes.linewidth": 1.0,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    x = best_df["true_mmgbsa"].to_numpy(dtype=float)
    y = best_df["pred_mmgbsa"].to_numpy(dtype=float)

    xmin = float(np.min([x.min(), y.min()]))
    xmax = float(np.max([x.max(), y.max()]))

    plt.figure(figsize=(5.2, 4.6))
    plt.scatter(x, y, s=28, alpha=0.85)

    # y=x 参考线
    plt.plot([xmin, xmax], [xmin, xmax], linestyle="--", linewidth=1.2, color="black")

    title = (
        f"Best structure: {best_row[group_col]}\n"
        f"Pearson = {best_row['pearson']:.3f}, Spearman = {best_row['spearman']:.3f}, "
        f"n = {int(best_row['n_frames'])}"
    )
    if "pdbid" in best_row and pd.notna(best_row["pdbid"]):
        title = (
            f"Best structure: {best_row[group_col]} ({best_row['pdbid']})\n"
            f"Pearson = {best_row['pearson']:.3f}, Spearman = {best_row['spearman']:.3f}, "
            f"n = {int(best_row['n_frames'])}"
        )

    plt.xlabel("True MM/GBSA (kcal/mol)")
    plt.ylabel("Predicted MM/GBSA (kcal/mol)")
    plt.title(title, pad=10)

    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", dpi=300)
    plt.close()


# ----------------------------
# main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="训练时使用的 config")
    parser.add_argument(
        "--setting-dir",
        type=str,
        default="results/mmgbsa_full/all_same_epoch",
        help="包含 best_model.pt 和 normalization.json 的目录",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        default="sample_id",
        choices=["sample_id", "pdbid"],
        help="按哪个字段定义‘一个结构’来计算相关性",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录；默认写到 setting-dir/re_eval_all_frames",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["seed"]))
    set_plot_style()

    setting_dir = Path(args.setting_dir)
    output_dir = Path(args.output_dir) if args.output_dir else (setting_dir / "re_eval_all_frames")
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_path = setting_dir / "best_model.pt"
    norm_path = setting_dir / "normalization.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not norm_path.exists():
        raise FileNotFoundError(f"Normalization file not found: {norm_path}")

    # 1) 复原 split
    lmdb = PepDynLMDB(config["data"]["lmdb_path"])
    split_df = resolve_split_df(config, lmdb)
    test_ids = split_df.loc[split_df["split"] == "test", "sample_id"].tolist()

    # 2) test 全 frame
    prune_c_h = bool(config["data"].get("prune_c_hydrogens", False))
    test_entries = build_mmgbsa_entries(
        test_ids,
        config["data"]["lmdb_path"],
        frame_mode="all",
    )

    stats = load_normalization_stats(norm_path)

    test_dataset = MMGBSAGraphDataset(
        config["data"]["lmdb_path"],
        test_entries,
        output_dir / "cache_test_all_frames",
        stats,
        int(config["data"]["k_neighbors"]),
        prune_c_h,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config["training"]["eval_batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
    )

    # 3) 建模型并加载 best_model.pt
    sample_graph = test_dataset.get(0)
    device = pick_device(config["training"]["device"])

    model = MMGBSAGCN(
        in_channels=sample_graph.x.shape[1],
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_layers=int(config["model"]["num_layers"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)

    # 4) 推理
    pred_df = evaluate_all_frames(
        model=model,
        loader=test_loader,
        device=device,
        stats=stats,
        desc="Re-evaluate test(all frames)",
    )
    pred_df.to_csv(output_dir / "pred_mmgbsa_test_all_frames.csv", index=False)

    # 5) 每个结构的 Pearson / Spearman
    corr_df = compute_per_structure_correlations(pred_df, group_col=args.group_col)
    corr_df.to_csv(output_dir / f"per_{args.group_col}_correlations.csv", index=False)

    # 6) 取相关性最好的结构，画 true vs pred 散点图
    valid_corr_df = corr_df.dropna(subset=["pearson"]).copy()
    if len(valid_corr_df) == 0:
        raise RuntimeError("No valid structure-level correlations could be computed.")

    best_row = valid_corr_df.iloc[0]
    best_key = best_row[args.group_col]
    best_df = pred_df.loc[pred_df[args.group_col] == best_key].copy()
    best_df = best_df.sort_values("frame_idx").reset_index(drop=True)
    best_df.to_csv(output_dir / f"best_{args.group_col}_predictions.csv", index=False)

    plot_best_structure_scatter(
        best_df=best_df,
        best_row=best_row,
        group_col=args.group_col,
        output_path=figures_dir / "fig_best_structure_true_vs_pred.pdf",
    )

    # 7) Pearson / Spearman histogram
    pearson_mean = plot_correlation_hist(
        corr_df["pearson"].tolist(),
        corrtype="Pearson",
        title=f"Per-{args.group_col} Pearson Distribution",
        bins=20,
        output_path=figures_dir / "fig_pearson_hist.pdf",
    )
    spearman_mean = plot_correlation_hist(
        corr_df["spearman"].tolist(),
        corrtype="Spearman",
        title=f"Per-{args.group_col} Spearman Distribution",
        bins=20,
        output_path=figures_dir / "fig_spearman_hist.pdf",
    )

    # 8) summary
    summary = {
        "setting_dir": str(setting_dir),
        "group_col": args.group_col,
        "num_test_structures": int(corr_df[args.group_col].nunique()),
        "num_test_frames": int(len(pred_df)),
        "mean_pearson": None if pd.isna(pearson_mean) else float(pearson_mean),
        "mean_spearman": None if pd.isna(spearman_mean) else float(spearman_mean),
        "best_structure": {
            args.group_col: best_row[args.group_col],
            "pdbid": None if pd.isna(best_row.get("pdbid", np.nan)) else best_row["pdbid"],
            "n_frames": int(best_row["n_frames"]),
            "pearson": None if pd.isna(best_row["pearson"]) else float(best_row["pearson"]),
            "spearman": None if pd.isna(best_row["spearman"]) else float(best_row["spearman"]),
            "rmse": float(best_row["rmse"]),
            "mae": float(best_row["mae"]),
        },
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()