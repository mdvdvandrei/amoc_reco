#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===============================================================
#  Bayes-GLS trend with heteroskedasticity and AR(1) correlation
#  Integrated into your BNN inference script
#  - Uses μ(t) from the BNN as the latent signal to trend
#  - Uses total_sigma(t) (or aleatoric) as observational σ(t)
#  - Optional AR(1) correlation estimated via Yule–Walker
#  - Produces CI for slope (trend) and draws trend lines
# ===============================================================

import os
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import scipy.stats as st
import numpy.linalg as npl

# (optional) dask tuned to be synchronous for reproducibility
import dask
dask.config.set(scheduler='synchronous')

# ─── Your model and dataset imports ─────────────────────────────────────────
from models import ResidualCNNHet
from obs_datasets import (
    lowpass_filter,
    PreprocessedDataset,
    PreprocessedSSHDatasetFromZarr
)

# ===============================================================
#  Utility: robust symmetric inverse via Cholesky with jitter
# ===============================================================
def safe_inv_sym(A: np.ndarray, jitter: float = 1e-6) -> np.ndarray:
    """Stable inversion of a symmetric PD matrix using Cholesky + jitter."""
    A = np.asarray(A, float)
    A = 0.5 * (A + A.T)  # force symmetry
    diag_scale = np.mean(np.diag(A))
    if not np.isfinite(diag_scale) or diag_scale <= 0:
        diag_scale = 1.0
    A = A + jitter * diag_scale * np.eye(A.shape[0])
    L = npl.cholesky(A)
    Linv = npl.solve(L, np.eye(A.shape[0]))
    return Linv.T @ Linv

# ===============================================================
#  AR(1) correlation matrix (Toeplitz)
# ===============================================================
def make_corr_AR1(T: int, rho: float) -> np.ndarray:
    idx = np.arange(T)
    return rho ** np.abs(idx[:, None] - idx[None, :])

# ===============================================================
#  Quick AR(1) estimate on (optionally) σ-normalized series
# ===============================================================
def estimate_rho_yw(y: np.ndarray, sigma_t: np.ndarray | None = None) -> float:
    r = np.asarray(y, float)
    if sigma_t is not None:
        s = np.asarray(sigma_t, float)
        s = np.where(s > 1e-12, s, 1.0)
        r = r / s
    r = r - r.mean()
    T = len(r)
    if T < 3:
        return 0.0
    c0 = float(np.dot(r, r) / T)
    if c0 <= 0 or not np.isfinite(c0):
        return 0.0
    c1 = float(np.dot(r[1:], r[:-1]) / (T - 1))
    rho = c1 / c0
    rho = float(np.clip(rho, -0.98, 0.98))  # keep well inside PD region
    return rho

# ===============================================================
#  Bayes-GLS trend (flat prior on β) with heteroskedasticity and AR(1)
#  Returns slope, intercept, 95% CI for slope, cov(β)
# ===============================================================
def bayes_gls_trend_ci(
    y: np.ndarray,
    sigma_t: np.ndarray,
    t_idx: np.ndarray | None = None,
    rho: float | None = None,
    alpha: float = 0.05
):
    """
    Posterior for β in y ~ N(Xβ, Σ) with flat prior on β:
      β | y ~ N(β̂, (X^T Σ^{-1} X)^{-1}),
    where Σ = D^{1/2} R D^{1/2}, D = diag(σ_t^2), R = AR(1) if rho is not None.

    Parameters
    ----------
    y : array-like, shape (T,)
        Series to trend (e.g., μ(t) from BNN).
    sigma_t : array-like, shape (T,)
        Time-varying observational std (e.g., total_sigma or aleatoric).
    t_idx : array-like, shape (T,), optional
        Time index to use in design matrix; if None, uses 0..T-1.
    rho : float in (-1, 1), optional
        AR(1) coefficient; if None or ~0, assumes no correlation.
    alpha : float
        Significance level for CI (default 0.05 -> 95% CI).

    Returns
    -------
    slope, intercept, low_slope, up_slope, cov_beta
    """
    y = np.asarray(y, float)
    T = len(y)
    x = np.arange(T, dtype=float) if t_idx is None else np.asarray(t_idx, float)
    X = np.c_[np.ones(T), x]

    sigma_t = np.asarray(sigma_t, float)
    sigma2 = np.clip(sigma_t**2, 1e-12, np.inf)
    Dsqrt = np.sqrt(sigma2)

    if (rho is None) or (abs(rho) < 1e-12):
        Sigma = np.diag(sigma2)
    else:
        R = make_corr_AR1(T, rho)
        Sigma = (Dsqrt[:, None] * R) * Dsqrt[None, :]

    Sinv = safe_inv_sym(Sigma)
    XtSinv = X.T @ Sinv
    cov_beta = safe_inv_sym(XtSinv @ X)       # posterior covariance of β
    beta_hat = cov_beta @ (XtSinv @ y)        # posterior mean (GLS estimator)

    slope = float(beta_hat[1])
    intercept = float(beta_hat[0])

    z = st.norm.ppf(1 - alpha / 2.0)         # large-sample normal quantile
    se_slope = float(np.sqrt(cov_beta[1, 1]))
    low_slope = slope - z * se_slope
    up_slope  = slope + z * se_slope

    return slope, intercept, low_slope, up_slope, cov_beta

# ===============================================================
#  BNN forward pass to get μ(t) and σ(t)
# ===============================================================
def get_bnn_predictions(dataset, weight_pattern, in_ch, out_dim,
                        device=None, batch_size=16):
    """
    For every checkpoint matching weight_pattern:
      • loads ResidualCNNHet(in_ch, out_dim)
      • predicts μ and logσ² across the dataset

    Returns:
      all_mus     (M×N array of μ’s)
      all_sigs    (M×N array of aleatoric σ’s)
      mu_mean     (N,)  mean μ over folds
      alea_sigma  (N,)  mean aleatoric σ over folds
      epi_sigma   (N,)  std of μ across folds (epistemic)
      total_sigma (N,)  sqrt(alea² + epi²)
    """
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpts = sorted(glob.glob(weight_pattern))
    if not ckpts:
        raise RuntimeError(f"No checkpoints match '{weight_pattern}'")
    print(f"Found {len(ckpts)} checkpoints.")

    all_mus, all_sigs = [], []
    for ck in ckpts:
        model = ResidualCNNHet(in_ch, out_dim).to(device)
        model.load_state_dict(torch.load(ck, map_location=device))
        model.eval()

        mus, sigs = [], []
        with torch.no_grad():
            for x in loader:
                # ensure shape (B,C,H,W)
                if isinstance(x, torch.Tensor):
                    pass
                else:
                    x = torch.as_tensor(x)

                if x.ndim == 3:
                    x = x.unsqueeze(1)
                x = x.to(device=device, dtype=torch.float32)

                mu, logvar = model(x)
                # soft-clamp logvar to [-8,8]
                logvar = 8.0 * torch.tanh(logvar / 8.0)

                mu_np  = mu.squeeze(-1).detach().cpu().numpy() * 1.16
                sig_np = np.exp(0.5 * logvar.squeeze(-1).detach().cpu().numpy()) * 1.16

                mus.append(mu_np)
                sigs.append(sig_np)

        all_mus.append(np.concatenate(mus, axis=0))
        all_sigs.append(np.concatenate(sigs, axis=0))
        print(f" • Loaded {os.path.basename(ck)} → {all_mus[-1].shape[0]} samples")

    all_mus  = np.stack(all_mus, axis=0)   # (folds, N)
    all_sigs = np.stack(all_sigs, axis=0)  # (folds, N)

    mu_mean    = all_mus.mean(axis=0)
    alea_sigma = all_sigs.mean(axis=0)
    epi_sigma  = all_mus.std(axis=0)
    total_sigma= np.sqrt(alea_sigma**2 + epi_sigma**2)

    return all_mus, all_sigs, mu_mean, alea_sigma, epi_sigma, total_sigma

# ===============================================================
#  Main
# ===============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ===========================================================
    # 1) SST Dataset & BNN Weights
    # ===========================================================
    sst_file = "processed_HadISST_sst.nc"
    sst_dataset = PreprocessedDataset(
        sst_file,
        variable="__xarray_dataarray_variable__",
        lpf="LPF120",
        order=5, fs=1,
        monthly=True,
        minus_basin_mean=False
    )

    # Determine input dims
    x0 = sst_dataset[0]
    if x0.ndim == 2:
        x0 = x0[np.newaxis]
    in_ch = x0.shape[0]

    sst_pattern = os.path.join(
        "weights/bnn_test_god_bless_it_work",
        "tos", "*", "best.pt"
    )

    (mus_sst, sigs_sst,
     mu_sst, alea_sst,
     epi_sst, tot_sst) = get_bnn_predictions(
        sst_dataset, sst_pattern, in_ch, 1,
        device=device, batch_size=16
    )

    ds_sst = xr.open_dataset(sst_file)
    times  = ds_sst.coords["time"].values

    # ===========================================================
    # Bayes-GLS trend on μ_sst with σ(t) from BNN
    # Choose which σ(t) to use: total or aleatoric
    # ===========================================================
    use_total_sigma = True
    sigma_series = tot_sst if use_total_sigma else alea_sst

    # Exact window
    start1, end1 = np.datetime64('1870-01-01'), np.datetime64('2014-12-31')
    mask1 = (times >= start1) & (times <= end1)
    y1 = mu_sst[mask1]
    s1 = sigma_series[mask1]

    # Estimate AR(1) on σ-normalized series (or set fixed rho here)
    rho1 = estimate_rho_yw(y1, sigma_t=s1)
    print(f"[SST] Estimated rho ≈ {rho1:.3f}")

    # Bayes-GLS slope CI (95%)
    slope1, int1, low1, up1, covb1 = bayes_gls_trend_ci(
        y=y1,
        sigma_t=s1,
        t_idx=np.arange(mask1.sum(), dtype=float),
        rho=rho1,
        alpha=0.05
    )

    # Convert to Sv/century (assuming monthly cadence → * 12 * 100)
    factor = 12 * 100
    t1_century = slope1 * factor
    ci1_half   = ((up1 - low1) / 2.0) * factor
    annot_sst = (
        f"Bayes-GLS trend {start1.astype('M8[Y]').item().year}–"
        f"{end1.astype('M8[Y]').item().year}: "
        f"{t1_century:.2f} ± {2*ci1_half:.2f} Sv/century (ρ≈{rho1:.2f})"
    )

    # Trend line for the window
    x1 = np.arange(mask1.sum(), dtype=float)
    trend_line1 = slope1 * x1 + int1

    # ===========================================================
    # Plot SST μ and CI (aleatoric) + trend line
    # ===========================================================
    plt.figure(figsize=(12, 3.8))
    plt.plot(times, mu_sst, lw=2, label='μ (BNN)', color='steelblue')
    plt.fill_between(times, mu_sst - 2*alea_sst, mu_sst + 2*alea_sst,
                     alpha=0.25, label='±2σ aleatoric', facecolor='steelblue')
    plt.plot(times[mask1], trend_line1, ls='--', lw=2, color='crimson', label='Bayes-GLS trend')
    plt.text(0.01, 0.97, annot_sst, transform=plt.gca().transAxes, va='top', color='crimson')

    plt.title("BNN Inference SST (μ with Bayes-GLS trend)")
    plt.xlabel("Time")
    plt.ylabel("AMOC, Sv")
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs("bnn_real_world_rec", exist_ok=True)
    plt.savefig("bnn_real_world_rec/sst_bayes_gls_trend.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ===========================================================
    # 2) SSH Dataset & BNN Weights
    # ===========================================================
    ssh_duacs_path = "duacs_data_6_may.zarr"
    ssh_duacs_dataset = PreprocessedSSHDatasetFromZarr(
        ssh_duacs_path,
        variable='__xarray_dataarray_variable__',
        lpf='LPF24',
        order=5,
        fs=1,
        monthly=True
    )

    x0_ssh = ssh_duacs_dataset[0]
    if x0_ssh.ndim == 2:
        x0_ssh = x0_ssh[np.newaxis]
    in_ch_ssh = x0_ssh.shape[0]

    ssh_pattern = os.path.join(
        "weights/bnn_test_god_bless_it_work_zos_24",
        "zos_minus_basin_mean", "*", "best.pt"
    )

    (mus_ssh, sigs_ssh,
     mu_ssh, alea_ssh,
     epi_ssh, tot_ssh) = get_bnn_predictions(
        ssh_duacs_dataset, ssh_pattern, in_ch_ssh, 1,
        device=device, batch_size=16
    )

    # Times for SSH (read from zarr/mfdataset as you originally did)
    ds_ssh = xr.open_mfdataset(ssh_duacs_path)
    times_ssh = ds_ssh.coords["time"].values

    # Bayes-GLS trend on full SSH span (or set your own window)
    sigma_ssh = tot_ssh if use_total_sigma else alea_ssh
    rho_ssh = estimate_rho_yw(mu_ssh, sigma_t=sigma_ssh)
    print(f"[SSH] Estimated rho ≈ {rho_ssh:.3f}")

    slope_ssh, int_ssh, low_ssh, up_ssh, covb_ssh = bayes_gls_trend_ci(
        y=mu_ssh,
        sigma_t=sigma_ssh,
        t_idx=np.arange(len(mu_ssh), dtype=float),
        rho=rho_ssh,
        alpha=0.05
    )
    t_ssh_century = slope_ssh * factor
    ci_ssh_half   = ((up_ssh - low_ssh) / 2.0) * factor
    annot_ssh = f"Bayes-GLS trend: {t_ssh_century:.2f} ± {2*ci_ssh_half:.2f} Sv/century (ρ≈{rho_ssh:.2f})"

    trend_line_ssh = slope_ssh * np.arange(len(mu_ssh)) + int_ssh

    # RAPID + Ekman as in your code
    ds_moc = xr.open_dataset('moc_vertical.nc')
    rapid_array = ds_moc['stream_function_mar'].resample(time='1M').mean()
    rapid_array_max = rapid_array.max(dim='depth')  # time-series
    rapid_array_max_f = lowpass_filter(rapid_array_max, 1/24, 5, 1, 30)

    ds_ekman = xr.open_dataset("moc_transports (3).nc")
    ekman = lowpass_filter(ds_ekman['t_ek10'].resample(time='1M').mean(), 1/24, pad=48)
    rapid_array_max_f = lowpass_filter(rapid_array_max, 1/24, 5, 1, 48) - ekman

    # ===========================================================
    # Plot SSH μ, aleatoric CI, RAPID(-Ekman), and Bayes-GLS trend
    # ===========================================================
    plt.figure(figsize=(12, 3.8))
    plt.plot(times_ssh, mu_ssh, lw=2, label='μ (BNN, SSH)', color="purple")
    plt.fill_between(times_ssh, mu_ssh - 2*alea_ssh, mu_ssh + 2*alea_ssh,
                     alpha=0.25, facecolor="purple", label='±2σ aleatoric')
    plt.plot(times_ssh, trend_line_ssh, ls='--', lw=2, color='indigo', label='Bayes-GLS trend')

    # Align RAPID series (as in your original code)
    try:
        plt.plot(
            rapid_array_max.time[24:], (rapid_array_max_f[24:] - 13.0),
            label="RAPID w/o Ekman (shifted by -13 Sv)",
            color='black', lw=3.0, alpha=0.8
        )
    except Exception as e:
        print(f"[WARN] Could not overlay RAPID: {e}")

    plt.text(0.01, 0.97, annot_ssh, transform=plt.gca().transAxes, va='top', color='indigo')

    plt.title("BNN Inference SSH (μ with Bayes-GLS trend)")
    plt.xlabel("Time")
    plt.ylabel("AMOC, Sv")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("bnn_real_world_rec/ssh_bayes_gls_trend.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("[DONE] Saved figures to bnn_real_world_rec/")
