#!/usr/bin/env python
# coding: utf-8
"""
BNN‑based CV inference & visualization for AMOC predictors
---------------------------------------------------------
* Uses **ResidualCNNHet** (μ & logσ²) head
* Generates four figures per variable:
    1. **bnn_cv** – BNN μ ± 2σ envelope vs ground truth
    2. **cv_results** – classic per‑fold μ traces vs truth
    3. **mean_over_members** – ensemble‑average μ ± 2σ vs ensemble‑avg truth
    4. **memberwise** – ≤10 subplots per ensemble member with ±2σ
* Saves per‑fold R² / Corr / MSE to `bnn_cv_metrics.csv`
"""

import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from test_dataset_zarr import PreprocessedCMIP6Dataset
from dataset_for_cesm2_LE import PreprocessedCMIP6Dataset_LE
from models import ResidualCNNHet
from train_weights_BNN import ResidualCNNHetSE

# ── Configuration ──────────────────────────────────────────────────
torch.set_num_threads(8)
torch.set_num_interop_threads(8)
BASE_DIR     = Path.cwd()
DATA_DIR     = BASE_DIR / "monthly_stream_zarr"
WEIGHTS_ROOT = BASE_DIR / "weights" / "bnn_test_god_bless_it_work"
LPF          = "LPF120"
SCENARIO     = "historical"
TARGET       = "y"
TEST_MODEL   = "CESM2"
SELECTED_LATS= [26.5]
VAR_CONFIG   = {
    "tos": {"x_vars": ["tos"], "subdir": "tos", "filename": "best.pt"},
    #"zos_minus_basin_mean": {"x_vars": ["zos_minus_basin_mean"], "subdir": "zos_minus_basin_mean", "filename": "best.pt"},
}

# ── Helpers ──────────────────────────────────────────────────────────
def compute_metrics(preds, targets):
    p, t = preds.ravel(), targets.ravel()
    ss_res = ((t-p)**2).sum()
    ss_tot = ((t-t.mean())**2).sum()
    r2   = 1 - ss_res/ss_tot if ss_tot else np.nan
    corr = np.corrcoef(p, t)[0,1] if p.size else np.nan
    mse  = np.mean((t-p)**2)
    return round(r2,2), round(corr,2), round(mse,3)

# ── Model loading & inference ───────────────────────────────────────
def load_bnn(in_shape, out_dim, ckpt, device):
    model = ResidualCNNHet(in_shape[0], out_dim).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model

def infer_bnn(model, dataset, device, batch_size=8):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    mus, sigs, tgts = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            mu, logv = model(x)
            # soft clamp log-variance
            logv = 8.0 * torch.tanh(logv / 8.0)
            mus.append(mu.cpu().numpy())
            sigs.append(np.exp(0.5 * logv.cpu().numpy()))
            tgts.append(y.numpy())
    return np.concatenate(mus,0), np.concatenate(sigs,0), np.concatenate(tgts,0)

def get_member_ids(ds):
    return np.array([ds.get_sample_info(i)["member"] for i in range(len(ds))])

# ── Plotting ─────────────────────────────────────────────────────────
def plot_bnn_cv(mus, sigs, tgt, out_dir: Path, var):
    out_dir.mkdir(parents=True, exist_ok=True)
    mu  = mus.mean(axis=0).ravel()
    sig = sigs.mean(axis=0).ravel()
    x   = np.arange(mu.size)
    plt.figure(figsize=(20,5))
    plt.plot(x, tgt.ravel(), 'r-', lw=2, label='Truth')
    plt.plot(x, mu,          'b-', lw=2, label='Mean μ')
    plt.fill_between(x, mu-2*sig, mu+2*sig, color='blue', alpha=0.3, label='±2σ')
    plt.title(f"BNN CV — {TEST_MODEL} — {var}")
    plt.xlabel('Time index'); plt.ylabel('Sv'); plt.grid(); plt.legend()
    plt.savefig(out_dir/f"{var}_bnn_cv.png", dpi=150); plt.close()


def plot_cv_folds(mus, tgt, out_dir: Path, var):
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(mus.shape[1])
    plt.figure(figsize=(20,6))
    for m in mus: plt.plot(x, m.ravel(), color='gray', alpha=0.4, lw=1)
    plt.plot(x, mus.mean(0).ravel(), 'b-', lw=3, label='Mean μ')
    plt.plot(x, tgt.ravel(), 'r-', lw=2, label='Truth')
    plt.title(f"CV folds — {TEST_MODEL} — {var}")
    plt.xlabel('Time index'); plt.ylabel('Sv'); plt.grid(); plt.legend()
    plt.savefig(out_dir/f"{var}_cv_results.png", dpi=150); plt.close()


def plot_avg_over_members(mean_mu, mean_sig, targets, member_ids,
                          out_dir: Path, var, scenario, model_name):
    out_dir.mkdir(parents=True, exist_ok=True)
    uniq = np.unique(member_ids)
    preds, sigs, trues = [], [], []
    for m in uniq:
        mask = (member_ids == m)
        preds.append(  mean_mu[mask])
        sigs.append(   mean_sig[mask])
        trues.append(targets[mask])
    avg_pred = np.stack(preds, 0).mean(0)
    avg_sig  = np.stack(sigs,  0).mean(0)
    avg_true = np.stack(trues, 0).mean(0)
    years = np.arange(1855,1855+avg_pred.size) if scenario=='historical' else np.arange(avg_pred.size)
    plt.figure(figsize=(12,5))
    plt.plot(years, avg_true, 'r-', lw=2, label='Truth (avg)')
    plt.plot(years, avg_pred, 'b-', lw=2, label='Pred (avg)')
    plt.fill_between(years, avg_pred-2*avg_sig, avg_pred+2*avg_sig, color='blue', alpha=0.25, label='±2σ')
    plt.title(f"Avg over members — {model_name} — {var}")
    plt.xlabel('Year' if scenario=='historical' else 'Time index')
    plt.ylabel('Sv'); plt.grid(); plt.legend()
    plt.savefig(out_dir/f"{var}_mean_over_members.png", dpi=150); plt.close()


def plot_memberwise(mus, sigs, tgt, member_ids,
                    out_dir: Path, var, scenario, model_name):
    years = np.arange(1855,2015) if scenario=='historical' else np.arange(mus.shape[1])
    mean_mu  = mus.mean(0).ravel()
    mean_sig = sigs.mean(0).ravel()
    low, high = mean_mu-2*mean_sig, mean_mu+2*mean_sig
    uniq = np.unique(member_ids)[:10]
    nrows, ncols = int(np.ceil(len(uniq)/2)), 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(20,3.5*nrows), sharex=True, sharey=True)
    axes = axes.flatten()
    for i,m in enumerate(uniq):
        ax = axes[i]
        idx = np.where(member_ids==m)[0]
        ax.fill_between(years[:idx.size], low[idx], high[idx], alpha=0.25, label='±2σ')
        ax.plot(years[:idx.size], mean_mu[idx], 'b-', lw=2, label='Prediction')
        ax.plot(years[:idx.size], tgt[idx], 'r-', lw=2,label='Ground truth' )
        ax.set_title(f"{model_name} — member {m}"); ax.grid(True)
        if i==0: ax.legend()
    for ax in axes[len(uniq):]: ax.axis('off')
    fig.suptitle(f"Member-wise — {var}", fontsize=16)
    fig.text(0.5,0.04,'Year', ha='center'); fig.text(0.06,0.5,'Sv', va='center', rotation='vertical')
    fig.tight_layout(rect=[0.06,0.06,1,0.95])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir/f"{var}_memberwise.png", dpi=150); plt.close(fig)

# ── Main ───────────────────────────────────────────────────────
if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    date   = datetime.now().strftime('%Y-%m-%d')
    metrics, mus_all, sigs_all, tgt_all, ids_all = [], {}, {}, {}, {}
    for var,cfg in VAR_CONFIG.items():
        ds = PreprocessedCMIP6Dataset_LE(
            zarr_dir=str(DATA_DIR), models=[TEST_MODEL], x_vars=cfg['x_vars'],
            scenarios=[SCENARIO], target_group="y", output_type='max',
            member_selection='all', selected_lats=SELECTED_LATS, lpf=LPF
        )
        mids = get_member_ids(ds); ids_all[var]=mids
        mus_all[var], sigs_all[var], tgt_all[var] = [], [], None
        for fold in sorted((WEIGHTS_ROOT/cfg['subdir']).iterdir()):
            ckpt=fold/cfg['filename']
            if not ckpt.exists(): continue
            x0,y0=ds[0]; out_dim=int(np.prod(y0.shape))
            model=load_bnn(x0.shape, out_dim, ckpt, device)
            mu,sig,tgt = infer_bnn(model, ds, device)
            mus_all[var].append(mu); sigs_all[var].append(sig); tgt_all[var]=tgt
            r2,corr,mse = compute_metrics(mu,tgt)
            metrics.append({'var':var,'fold':fold.name,'r2':r2,'corr':corr,'mse':mse})
    out_root=BASE_DIR/'bnn_cv_results'/date
    for var in VAR_CONFIG:
        mus = np.stack(mus_all[var]); sigs=np.stack(sigs_all[var]); tgt=tgt_all[var]; mids=ids_all[var]
        var_dir=out_root/var
        plot_bnn_cv(mus,sigs,tgt,var_dir,var)
        plot_cv_folds(mus,tgt,var_dir,var)
        plot_avg_over_members(mus.mean(0).ravel(), sigs.mean(0).ravel(), tgt.ravel(), mids, var_dir, var, SCENARIO, TEST_MODEL)
        plot_memberwise(mus,sigs,tgt,mids,var_dir,var,SCENARIO,TEST_MODEL)
    pd.DataFrame(metrics).to_csv(BASE_DIR/'bnn_cv_metrics.csv',index=False)
    print("Saved metrics → bnn_cv_metrics.csv")
