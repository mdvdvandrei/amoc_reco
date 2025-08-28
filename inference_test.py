#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cross-validation inference & visualisation for AMOC predictors.

Adds:
  • 10-subplot member-wise plots with ±1 σ envelope
  • helper to grab member IDs only once
"""

# ───────── Imports ────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from test_dataset_zarr import PreprocessedCMIP6Dataset
from models import ResidualCNN, SimpleCNN, SimpleViT
from dataset_for_cesm2_LE import PreprocessedCMIP6Dataset_LE


# ───────── CONFIGURATION ─────────────────────────────────────────────────────
torch.set_num_threads(8)
torch.set_num_interop_threads(8)

BASE_DIR        = Path.cwd()
YEARLY_MEAN_DIR = BASE_DIR / "yearly_mean_data_old"
RUN_NAME        = "baseline_approach"
WEIGHTS_ROOT    = BASE_DIR / "weights" / RUN_NAME

LPF            = "LPF120"
SELECTED_LATS  = [26.5]
SCENARIO       = "historical"
TARGET         = "y"
LAST_30_YEARS_FLAG = False
test_model     = "CESM2"

VAR_CONFIG = {
    "tos": {
        "x_vars": ["tos"],
        "subdir": "tos",
        "filename": "best_model_tos.pth"
    },
    #"zos_minus_basin_mean": {
    #    "x_vars": ["zos_minus_basin_mean"],
    #    "subdir": "zos_minus_basin_mean",
    #    "filename": "best_model_zos_minus_basin_mean.pth"
    #},
}


# ───────── HELPERS ───────────────────────────────────────────────────────────
def compute_metrics(preds: np.ndarray, targets: np.ndarray):
    preds    = preds.ravel()
    targets  = targets.ravel()
    ss_res   = ((targets - preds) ** 2).sum()
    ss_tot   = ((targets - targets.mean()) ** 2).sum()
    r2       = 1 - ss_res / ss_tot if ss_tot else np.nan
    corr     = np.corrcoef(preds, targets)[0, 1] if preds.size else np.nan
    mse      = np.mean((targets - preds) ** 2)
    return round(r2, 2), round(corr, 2), round(mse, 3)


def load_model(model_cls, input_shape, output_dim, weights_path, device):
    model = model_cls(input_shape[0], output_dim).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def get_preds_and_targets(model, dataset, device):
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    preds, tgts = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device).float()).cpu().numpy())
            tgts.append(y.numpy())
    return np.concatenate(preds, 0), np.concatenate(tgts, 0)


def get_member_ids(ds):
    """Return an array with the member-ID for every sample in `ds`."""
    return np.array([ds.get_sample_info(i)["member"] for i in range(len(ds))])


# ── Existing CV plots (unchanged) ────────────────────────────────────────────
def plot_cv_results(cv_preds, cv_targets, out_dir: Path, model_name: str, var: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    cv_preds = np.asarray(cv_preds)
    x        = np.arange(len(cv_preds[0]))
    cv_targets = np.squeeze(cv_targets)

    # 1. Classic CV plot
    plt.figure(figsize=(20, 6))
    for fold_preds in cv_preds:
        plt.plot(x, fold_preds * 1.16, alpha=0.4, color='gray', lw=1)

    mean_preds = cv_preds.mean(axis=0)
    plt.plot(x, mean_preds * 1.16, label="Mean Prediction", color='blue', lw=3)
    plt.plot(x, cv_targets.T * 1.16, label="Ground Truth", color='red', lw=2)

    plt.title(f"Cross-validation results  {test_model}  {var}")
    plt.xlabel("Time Index")
    plt.ylabel("Sv")
    plt.legend()
    plt.grid(True)

    file_main = out_dir / f"{var}_cv_results.png"
    plt.savefig(file_main, dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Std-over-folds + residuals
    std_preds  = cv_preds.std(axis=0)
    res        = cv_targets.T.ravel() - mean_preds.ravel()

    split_cfg = {
        "piControl":  (1,   0),
        "historical": (10, 1855),
        "ssp126":     (3, 2020),
        "ssp245":     (3, 2020),
        "ssp585":     (3, 2020),
    }
    n_seg, x_shift = split_cfg.get(SCENARIO, (1, 0))

    segments_std   = np.array_split(std_preds, n_seg)
    segments_res   = np.array_split(res,       n_seg)
    segments_preds = np.array_split(mean_preds,         n_seg)
    segments_true  = np.array_split(cv_targets.T.ravel(), n_seg)

    mean_std  = np.mean(segments_std,   axis=0) * 1.16
    mean_res  = np.mean(segments_res,   axis=0) * 1.16
    mean_pred = np.mean(segments_preds, axis=0) * 1.16
    mean_true = np.mean(segments_true,  axis=0) * 1.16

    t      = np.arange(mean_std.size)
    x_axis = x_shift + t if x_shift else t

    # 2a. σ(t)
    plt.figure(figsize=(12, 4))
    plt.plot(x_axis, mean_std, lw=2, color="steelblue")
    plt.title(f"Across-fold STD – {test_model} – {var} – {SCENARIO}")
    plt.xlabel("Year" if x_shift else "Time index")
    plt.ylabel("Std (Sv)")
    plt.grid(True)
    plt.savefig(out_dir / f"{var}_std.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2b. Mean prediction vs truth
    plt.figure(figsize=(12, 4))
    plt.plot(x_axis, mean_pred, lw=2, color="steelblue", label="prediction")
    plt.fill_between(x_axis, mean_pred.ravel() - 2 * mean_std.ravel(), mean_pred.ravel() + 2 * mean_std.ravel(),
                alpha=0.25, label="±2σ")
    plt.plot(x_axis, mean_true, lw=2, color="red",        label="ground truth")
    plt.title(f"Mean prediction over members – {test_model} – {var}")
    plt.xlabel("Year" if x_shift else "Time index")
    plt.ylabel("Sv")
    plt.grid(True)
    plt.legend()
    plt.savefig(out_dir / f"{var}_mean_preds_over_members.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # 2c. Residuals
    plt.figure(figsize=(12, 4))
    plt.plot(x_axis, mean_res, lw=2, color="steelblue")
    
    plt.title(f"Residual (target – mean pred) – {test_model} – {var}")
    plt.xlabel("Year" if x_shift else "Time index")
    plt.ylabel("Sv")
    plt.grid(True)
    plt.savefig(out_dir / f"{var}_res.png", dpi=150, bbox_inches="tight")
    plt.close()

# ── NEW: member-wise 10-subplot figure – with year axis ─────────────────────
# ── NEW: member-wise figure with unified year axis ───────────────────────────
# ── member-wise figure с общей осью 1855-2015 ────────────────────────────────
def plot_memberwise(cv_preds, cv_targets, member_ids,
                    out_dir: Path, var: str, scenario: str, test_model: str):
    """
    Рисуем ≤10 членов ансамбля на единой шкале лет (1855-2015 для historical).
    Предполагаем, что у каждого члена ровно столько же временных точек,
    сколько лет в диапазоне; если меньше — дополняем NaN-ами с конца.
    """

    # ── 1. общий временной вектор ------------------------------------------------
    if scenario == "historical":
        years = np.arange(1855, 2015)                     # 1855 … 2014
    else:
        # fallback: просто индексы
        years = np.arange(cv_preds.shape[1])

    n_t = len(years)                                      # 161 для historical

    # ── 2. среднее и σ по фолдам (для фона) -------------------------------
    mean_p = cv_preds.mean(axis=0).ravel()                # (Ntot,)
    std_p  = cv_preds.std( axis=0).ravel()
    truth  = cv_targets.ravel()

    # для заливки ±1σ нужно полноразмерно на каждую позицию,
    # поэтому берём член-независимые mean/std
    shade_low  = (mean_p - std_p*2) * 1.16
    shade_high = (mean_p + std_p*2) * 1.16

    # ── 3. готовим сетку сабплотов ----------------------------------------
    uniq_mems = sorted(np.unique(member_ids))[:10]
    ncols, nrows = 2, int(np.ceil(len(uniq_mems) / 2))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(20, 3.5 * nrows),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    for i, mem in enumerate(uniq_mems):
        ax   = axes[i]
        idxs = np.where(member_ids == mem)[0]             # позиции этого члена

        # вытаскиваем предсказания/правду конкретного члена
        p_raw = mean_p[idxs] * 1.16
        t_raw = truth [idxs] * 1.16

        # если длина вдруг меньше n_t → дополняем NaN с конца
        if p_raw.size < n_t:
            pad = n_t - p_raw.size
            p_raw = np.concatenate([p_raw, np.full(pad, np.nan)])
            t_raw = np.concatenate([t_raw, np.full(pad, np.nan)])

        # заливка ±1σ (полностью, без маски)
        ax.fill_between(years, shade_low[idxs], shade_high[idxs],
                        alpha=0.25, label="±2σ")

        # линии
        ax.plot(years, p_raw, lw=2,            label="Prediction")
        ax.plot(years, t_raw, lw=2, color="red", label="Ground truth")

        ax.set_title(f"{test_model}  {mem}")
        ax.set_xlim(years[0], years[-1])
        ax.grid(True)
        if i == 0:
            ax.legend()

    # гасим пустые панели, если членов <10
    for ax in axes[len(uniq_mems):]:
        ax.axis("off")

    fig.suptitle(f"{var} – {scenario} – member-wise predictions", fontsize=16)
    fig.text(0.5,  0.04, "Year", ha="center")
    fig.text(0.06, 0.5,  "Sv",   va="center", rotation="vertical")
    fig.tight_layout(rect=[0.06, 0.06, 1, 0.95])

    out_dir.mkdir(parents=True, exist_ok=True)
    file_out = out_dir / f"{var}_memberwise.png"
    fig.savefig(file_out, dpi=150)
    plt.close(fig)
    print(f"  → saved member-wise plot to {file_out}")


# ───────── MAIN ──────────────────────────────────────────────────────────────
def main():
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    date_str = datetime.now().strftime("%Y-%m-%d")

    results = []
    summary = {var: {"r2": [], "corr": [], "mse": [],
                     "std_pred": [], "std_true": []}
               for var in VAR_CONFIG}

    # Discover trained-weights sub-folders
    model_names = [d.name for d in (WEIGHTS_ROOT / "tos").iterdir() if d.is_dir()]

    # Store per-variable across-fold predictions / targets / member IDs
    all_cv_preds    = {var: [] for var in VAR_CONFIG}
    all_cv_targets  = {var: [] for var in VAR_CONFIG}
    all_member_ids  = {var: None for var in VAR_CONFIG}

    for model_name in model_names:
        print(f"\nProcessing {model_name} …")
        row = {"model": model_name}

        for var, cfg in VAR_CONFIG.items():
            weight_path = WEIGHTS_ROOT / cfg["subdir"] / model_name / cfg["filename"]
            print(f"  Running {var} with weights {weight_path.name}")

            ds = PreprocessedCMIP6Dataset_LE(
                zarr_dir="/home/am334/link_am334/moc_mmodel/monthly_stream_zarr",
                models=[test_model],
                x_vars=cfg["x_vars"],
                scenarios=[SCENARIO],
                target_group=TARGET,
                output_type="max",
                member_selection="all",
                selected_lats=SELECTED_LATS,
                lpf=LPF
            )
            if len(ds) == 0:
                print(f"  ⚠️  Empty dataset for {var}")
                continue

            # Grab member IDs once
            member_ids = get_member_ids(ds)
            if all_member_ids[var] is None:
                all_member_ids[var] = member_ids

            x0, y0   = ds[0]
            out_dim  = int(np.prod(y0.shape))
            model    = load_model(ResidualCNN, x0.shape, out_dim, weight_path, device)
            preds, tgts = get_preds_and_targets(model, ds, device)

            all_cv_preds[var].append(preds)
            all_cv_targets[var].append(tgts)

            r2, corr, mse = compute_metrics(preds, tgts)
            std_pred, std_true = np.std(preds), np.std(tgts)

            row.update({
                f"r2_{var}":       r2,
                f"corr_{var}":     corr,
                f"mse_{var}":      round(mse, 4),
                f"std_pred_{var}": round(std_pred, 3),
                f"std_true_{var}": round(std_true, 3),
            })

            summary[var]["r2"].append(r2)
            summary[var]["corr"].append(corr)
            summary[var]["mse"].append(mse)
            summary[var]["std_pred"].append(std_pred)
            summary[var]["std_true"].append(std_true)

        results.append(row)

    # ── Plotting ─────────────────────────────────────────────────────────────
    out_root = BASE_DIR / "cv_results" / date_str

    for var in VAR_CONFIG:
        cv_preds_var   = np.array(all_cv_preds[var])          # (folds, N)
        cv_targets_var = all_cv_targets[var][0]               # any fold’s targets
        member_ids_var = all_member_ids[var]

        # old CV & std plots
        plot_cv_results(cv_preds_var, cv_targets_var, out_root,
                        f"{model_names[0]}-…", var)

        # new member-wise plot
        plot_memberwise(cv_preds_var, cv_targets_var, member_ids_var,
                        out_root, var, SCENARIO, test_model)

    # ── Save CSVs ────────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df_file = BASE_DIR / "results" / \
        f"cv_results_{LPF}_{SCENARIO}_{RUN_NAME}_LAST_30yrs_{LAST_30_YEARS_FLAG}.csv"
    df.to_csv(df_file, index=False)
    print("\nDetailed results →", df_file)

    avg_rows = []
    for var, m in summary.items():
        avg_rows.append({
            "variable":      var,
            "avg_r2":        round(np.nanmean(m["r2"]),        2),
            "avg_mse":       round(np.nanmean(m["mse"]),       3),
            "avg_corr":      round(np.nanmean(m["corr"]),      2),
            "avg_std_pred":  round(np.nanmean(m["std_pred"]),  3),
            "avg_std_true":  round(np.nanmean(m["std_true"]),  3),
        })
    avg_df = pd.DataFrame(avg_rows)
    avg_file = BASE_DIR / "results" / \
        f"cv_avg_metrics_{LPF}_{SCENARIO}_{RUN_NAME}_LAST_30yrs_{LAST_30_YEARS_FLAG}.csv"
    avg_df.to_csv(avg_file, index=False)
    print("Summary metrics →", avg_file)
    print(avg_df)


if __name__ == "__main__":
    main()
