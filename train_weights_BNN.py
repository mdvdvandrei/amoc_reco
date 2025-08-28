#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Two-stage (μ-only -> σ-only) trainer, FP32 only (no AMP), Hydra-configured.
# Baseline ResidualCNNHet (LeakyReLU), flat skip; plots each epoch in σ-stage.
# Includes proper device handling for logvar_prior to avoid CPU/CUDA mismatch.

import os
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR

import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import pandas as pd

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd

# Optional wandb (safe to ignore if not installed)
try:
    import wandb
    WANDB_OK = True
except Exception:
    WANDB_OK = False

# ----- Global perf knobs (safe without AMP) -----
torch.set_default_dtype(torch.float32)
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

# ----------------- Plotting (every σ-epoch) -----------------
def plot_timeseries_uncert(mu, std, targets, epoch, var, base_dir="val_plots/aleatoric"):
    root = get_original_cwd()
    out_dir = os.path.join(root, base_dir, datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(out_dir, exist_ok=True)
    idx = np.arange(len(mu))
    plt.figure(figsize=(12, 5))
    plt.plot(idx, targets[:, 0], label="Ground Truth", lw=1)
    plt.plot(idx, mu[:, 0], label="Prediction μ", lw=1)
    plt.fill_between(idx, mu[:, 0]-2*std[:, 0], mu[:, 0]+2*std[:, 0], alpha=0.3, label="±2σ")
    plt.title(f"{var} – epoch {epoch}")
    plt.xlabel("Sample"); plt.ylabel(var)
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{var}_e{epoch}.png"))
    plt.close()

# ----------------- Model (baseline, no SE) -----------------
class ResidualBlock_new(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, p_drop=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=True)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.act   = nn.LeakyReLU(0.01, inplace=True)
        self.drop  = nn.Dropout(p_drop)
        self.short = (
            nn.Identity() if (stride == 1 and in_channels == out_channels) else
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = out + self.short(x)
        return self.act(out)

class ResidualCNNHetBaseline(nn.Module):
    """
    Residual CNN with μ/logσ² heads + flat skip to both.
    If model_args.flat_hw provided -> fixed Linear; else LazyLinear.
    """
    def __init__(self, in_channels: int, out_dim: int, flat_hw=None, p_drop: float = 0.3, tanh_scale: float = 8.0):
        super().__init__()
        self.out_dim = out_dim
        self.tanh_scale = tanh_scale

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 1, 1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.layer1 = ResidualBlock_new(16,  32,  2, p_drop)
        self.layer2 = ResidualBlock_new(32,  64,  2, p_drop)
        self.layer3 = ResidualBlock_new(64,  128, 2, p_drop)
        self.layer4 = ResidualBlock_new(128, 256, 2, p_drop)
        self.pool   = nn.AdaptiveAvgPool2d(1)

        self.mu_head = nn.Sequential(
            nn.Linear(256, 64), nn.LeakyReLU(0.01, inplace=True), nn.Dropout(0.5),
            nn.Linear(64, out_dim)
        )
        self.log_head = nn.Sequential(
            nn.Linear(256, 64), nn.LeakyReLU(0.01, inplace=True), nn.Dropout(0.5),
            nn.Linear(64, out_dim)
        )

        if flat_hw:
            H, W = flat_hw
            flat_dim = in_channels * H * W
            self.skip_mu  = nn.Linear(flat_dim, out_dim)
            self.skip_log = nn.Linear(flat_dim, out_dim)
        else:
            self.skip_mu  = nn.LazyLinear(out_dim)
            self.skip_log = nn.LazyLinear(out_dim)

        # Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        b = x.size(0)
        flat = x.view(b, -1)
        h = self.stem(x)
        h = self.layer4(self.layer3(self.layer2(self.layer1(h))))
        trunk = self.pool(h).view(b, -1)
        mu     = self.mu_head(trunk)  + self.skip_mu(flat)
        logvar = self.log_head(trunk) + self.skip_log(flat)
        s = self.tanh_scale
        logvar = s * torch.tanh(logvar / s)
        return mu, logvar

# ----------------- Metrics & helpers -----------------
def gaussian_crps_torch(mu, sigma, y, eps=1e-8):
    sigma = torch.clamp(sigma, min=eps)
    z = (y - mu) / sigma
    Phi = 0.5 * (1.0 + torch.special.erf(z / math.sqrt(2.0)))
    phi = (1.0 / math.sqrt(2.0*math.pi)) * torch.exp(-0.5 * z * z)
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))

def set_requires_grad(module: nn.Module, flag: bool):
    if module is None: return
    for p in module.parameters(): p.requires_grad = flag

def set_bn_eval(module: nn.Module):
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()

@torch.no_grad()
def welford_stats(loader, device):
    count = 0
    mean = None
    M2   = None
    for _, y in loader:
        y = y.to(device=device, dtype=torch.float32)
        b = y.size(0)
        batch_mean = y.mean(dim=0)
        batch_var  = y.var(dim=0, unbiased=False)
        if mean is None:
            mean = batch_mean
            M2   = batch_var * b
            count = b
        else:
            delta = batch_mean - mean
            tot   = count + b
            mean  = mean + delta * (b / max(1, tot))
            M2    = M2 + batch_var*b + (delta**2) * (count*b / max(1, tot))
            count = tot
    var = M2 / max(1, count)
    std = torch.sqrt(var + 1e-6)
    return mean.detach(), var.detach(), std.detach()

def get_prior_on(like_tensor, model):
    """Ensure prior buffer is on same device as like_tensor."""
    prior = getattr(model, "logvar_prior", None)
    if prior is not None and prior.device != like_tensor.device:
        prior = prior.to(like_tensor.device)
    return prior

# ----------------- Epoch loops (FP32 only) -----------------
def mse_epoch_mu_only(model, loader, opt, device, clip_grad=1.0):
    model.train()
    total = 0.0
    mse = nn.MSELoss()
    for x, y in loader:
        x = x.to(device, non_blocking=True, memory_format=torch.channels_last, dtype=torch.float32)
        y = y.to(device, non_blocking=True, dtype=torch.float32)
        opt.zero_grad(set_to_none=True)
        mu, _ = model(x)
        loss = mse(mu, y)
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        opt.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

def nll_epoch_sigma_only(model, loader, opt, device, loss_fn, var_lambda, clip_grad=1.0):
    model.train()
    set_bn_eval(model)
    total = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True, memory_format=torch.channels_last, dtype=torch.float32)
        y = y.to(device, non_blocking=True, dtype=torch.float32)
        opt.zero_grad(set_to_none=True)
        mu, logvar = model(x)
        mu = mu.detach()
        var = torch.exp(logvar)
        nll = loss_fn(mu, y, var)
        prior = get_prior_on(logvar, model)
        if prior is not None:
            var_pen = var_lambda * torch.mean((logvar - prior)**2)
        else:
            var_pen = var_lambda * torch.mean(logvar**2)
        loss = nll + var_pen
        loss.backward()
        if clip_grad is not None:
            trainable = [p for g in opt.param_groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(trainable, clip_grad)
        opt.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def val_epoch(model, loader, loss_fn, device, train_stats=None):
    model.eval()
    losses, mus, stds, tgts = [], [], [], []
    for x, y in tqdm(loader, desc="Val", leave=False):
        x = x.to(device, non_blocking=True, memory_format=torch.channels_last, dtype=torch.float32)
        y = y.to(device, non_blocking=True, dtype=torch.float32)
        mu, logvar = model(x)
        var = torch.exp(logvar)
        loss = loss_fn(mu, y, var)
        losses.append(loss * x.size(0))
        mus.append(mu)
        stds.append(torch.sqrt(var))
        tgts.append(y)

    mu_all  = torch.cat(mus,  dim=0)
    std_all = torch.cat(stds, dim=0)
    tgt_all = torch.cat(tgts, dim=0)

    resid = tgt_all - mu_all
    ss_res = torch.sum(resid**2, dim=0)
    ss_tot = torch.sum((tgt_all - tgt_all.mean(dim=0))**2, dim=0).clamp_min(1e-12)
    r2 = 1.0 - ss_res / ss_tot

    xm = mu_all - mu_all.mean(dim=0)
    ym = tgt_all - tgt_all.mean(dim=0)
    corr = torch.sum(xm*ym, dim=0) / (torch.sqrt(torch.sum(xm**2, dim=0))*torch.sqrt(torch.sum(ym**2, dim=0)) + 1e-12)

    rmse = torch.sqrt(torch.mean((mu_all - tgt_all)**2))
    crps_model = gaussian_crps_torch(mu_all, std_all, tgt_all).mean()

    crps_baseline = None
    if train_stats is not None:
        base_mu, base_std = train_stats["mean"].to(mu_all.device), train_stats["std"].to(mu_all.device)
        base_mu  = base_mu.expand_as(mu_all)
        base_std = base_std.expand_as(std_all)
        crps_baseline = gaussian_crps_torch(base_mu, base_std, tgt_all).mean()

    avg_loss = (torch.stack(losses).sum() / tgt_all.size(0)).item()
    return (avg_loss,
            float(corr.mean().item()),
            float(r2.mean().item()),
            float(rmse.item()),
            float(crps_model.item()),
            None if crps_baseline is None else float(crps_baseline.item()),
            mu_all.detach().cpu().numpy(),
            std_all.detach().cpu().numpy(),
            tgt_all.detach().cpu().numpy())

# ----------------- Param-group builders -----------------
def build_param_groups_stage1(model, wd_base=1e-4, skip_mult=5.0):
    decay, no_decay, skip_decay, skip_nodecay = [], [], [], []
    norm_types = (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)
    for name, module in model.named_modules():
        for pn, p in getattr(module, 'named_parameters', lambda recurse=False: [])(recurse=False):
            if not p.requires_grad: continue
            full = f"{name}.{pn}" if name else pn
            is_bias = pn.endswith("bias")
            is_norm = isinstance(module, norm_types)
            is_skip = full.startswith("skip_mu") or full.startswith("skip_log")
            if is_bias or is_norm: (skip_nodecay if is_skip else no_decay).append(p)
            else:                  (skip_decay   if is_skip else decay).append(p)
    groups = []
    if decay:        groups.append({"params": decay,        "weight_decay": wd_base})
    if no_decay:     groups.append({"params": no_decay,     "weight_decay": 0.0})
    if skip_decay:   groups.append({"params": skip_decay,   "weight_decay": wd_base * skip_mult})
    if skip_nodecay: groups.append({"params": skip_nodecay, "weight_decay": 0.0})
    return groups

def build_param_groups_sigma_only(model, wd_sigma=3e-4, skip_mult=5.0):
    decay, no_decay, skip_decay, skip_nodecay = [], [], [], []
    keep_prefixes = ("log_head", "skip_log")
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if not any(name.startswith(k) for k in keep_prefixes): continue
        is_bias = name.endswith("bias")
        is_norm = any(tok in name.lower() for tok in ["bn", "batchnorm", "layernorm", "groupnorm"])
        is_skip = name.startswith("skip_log")
        if is_bias or is_norm: (skip_nodecay if is_skip else no_decay).append(p)
        else:                  (skip_decay   if is_skip else decay).append(p)
    groups = []
    if decay:        groups.append({"params": decay,        "weight_decay": wd_sigma})
    if no_decay:     groups.append({"params": no_decay,     "weight_decay": 0.0})
    if skip_decay:   groups.append({"params": skip_decay,   "weight_decay": wd_sigma * skip_mult})
    if skip_nodecay: groups.append({"params": skip_nodecay, "weight_decay": 0.0})
    return groups

# ----------------- Build model (Hydra) -----------------
def build_model(input_ch, out_dim, cfg, device):
    args = cfg.get("model_args", {})
    def get_arg(name, default):
        if isinstance(args, dict): return args.get(name, default)
        return getattr(args, name, default)

    if cfg.model in ("ResidualCNNHet", "ResidualCNNHetBaseline"):
        flat_hw    = get_arg("flat_hw", [144, 108])
        flat_hw    = tuple(flat_hw) if flat_hw is not None else None
        p_drop     = float(get_arg("dropout", 0.3))
        tanh_scale = float(get_arg("sigma_tanh_scale", 8.0))
        m = ResidualCNNHetBaseline(input_ch, out_dim, flat_hw=flat_hw, p_drop=p_drop, tanh_scale=tanh_scale).to(device)
        if device.type == "cuda":
            m = m.to(memory_format=torch.channels_last)
        # Optional compile (disabled by default to avoid Triton build issues)
        use_compile = bool(getattr(cfg, "use_compile", False))
        if device.type == "cuda" and use_compile:
            try:
                m = torch.compile(m, backend="aot_eager", mode="max-autotune")
            except Exception as e:
                print(f"[WARN] torch.compile failed ({e}); continuing without compile.")
        return m

    raise ValueError(f"Unknown model {cfg.model}")

# ----------------- Main (Hydra) -----------------
@hydra.main(config_path="conf", config_name="config_bnn")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    experiment = f"CMIP6_BNN_{cfg.model}_{cfg.save_name}"
    if WANDB_OK:
        wandb.init(project="CMIP6_BNN", name=experiment, config=OmegaConf.to_container(cfg, resolve=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    from dataset_for_cesm2_LE import PreprocessedCMIP6Dataset_LE
    base_dir = os.path.join(get_original_cwd(), "monthly_stream_zarr")

    models_all = [m.strip() for m in cfg.models.split(",")]
    val_models = [v.strip() for v in cfg.val_model.split(",")]
    scenarios  = [s.strip() for s in cfg.scenarios.split(",")]
    x_vars     = [v.strip() for v in cfg.x_vars.split(",")]
    results = []

    def ds_has_data(ds):
        try:
            _ = ds[0]
            return True
        except Exception:
            return False

    for v_model in val_models:
        train_models = [m for m in models_all if m != v_model]
        for var in x_vars:
            ds_train = PreprocessedCMIP6Dataset_LE(
                zarr_dir=base_dir, models=train_models, x_vars=[var],
                scenarios=scenarios, target_group=cfg.target_var, output_type=cfg.output_type,
                member_selection="first", selected_lats=[float(l) for l in cfg.selected_lats.split(",")],
                lpf=cfg.lpf, noise=cfg.noise if cfg.noise not in (None, "null", "None", "") else None
            )
            ds_val = PreprocessedCMIP6Dataset_LE(
                zarr_dir=base_dir, models=[v_model], x_vars=[var],
                scenarios=[cfg.val_scenario], target_group=cfg.target_var, output_type=cfg.output_type,
                member_selection="all", selected_lats=[float(l) for l in cfg.selected_lats.split(",")],
                lpf=cfg.lpf, noise=None
            )
            if not ds_has_data(ds_train):
                print(f"[WARN] No TRAIN data for {v_model}/{var} (x_vars={cfg.x_vars}). Skipping.")
                continue
            if not ds_has_data(ds_val):
                print(f"[WARN] No VAL data for {v_model}/{var} (x_vars={cfg.x_vars}). Skipping.")
                continue

            out_dim  = ds_val[0][1].shape[0]
            input_ch = ds_val[0][0].shape[0]
            model   = build_model(input_ch, out_dim, cfg, device)
            loss_fn = nn.GaussianNLLLoss(eps=1e-6)

            train_loader = DataLoader(
                ds_train, batch_size=cfg.training.batch_size, shuffle=True,
                num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4
            )
            val_loader   = DataLoader(
                ds_val,   batch_size=cfg.training.batch_size, shuffle=False,
                num_workers=4,  pin_memory=True, persistent_workers=True, prefetch_factor=2
            )

            # Train stats + σ warm-start (pre-tanh bias on correct device)
            train_mean, train_var, train_std = welford_stats(train_loader, device)
            train_stats = {"mean": train_mean.detach().cpu(), "std": train_std.detach().cpu()}
            with torch.no_grad():
                last_lin = [m for m in model.log_head if isinstance(m, nn.Linear)][-1]
                nn.init.zeros_(last_lin.weight)
                target = torch.log(train_var.float() + 1e-6)        # desired post-squash logvar
                s = getattr(model, "tanh_scale", 8.0)
                pre = torch.clamp(target / s, -0.999, 0.999)
                pre = torch.atanh(pre) * s
                last_lin.bias.copy_(pre.detach().cpu())              # bias params live on CPU until first forward for LazyLinear safety
                # Register prior on the SAME device as model
                model.register_buffer("logvar_prior", target.detach().to(device))

            # ---- Stage 1: μ-only ----
            set_requires_grad(model.log_head, False)
            set_requires_grad(model.skip_log, False)

            wd_base = getattr(cfg, "weight_decay", 1e-4)
            use_fused = getattr(AdamW, "fused", False) and (device.type == "cuda")
            param_groups = build_param_groups_stage1(model, wd_base=wd_base, skip_mult=5.0)
            optim = AdamW(param_groups, lr=cfg.training.learning_rate, fused=use_fused)
            sched = CosineAnnealingLR(optim, T_max=cfg.training.mu_epochs)

            best_val = float("inf")
            weight_dir = os.path.join(get_original_cwd(), "weights", cfg.save_name, var, v_model)
            os.makedirs(weight_dir, exist_ok=True)
            best_stage1_path = os.path.join(weight_dir, "best_val_stage1.pt")

            for epoch in range(cfg.training.mu_epochs):
                tr_mse = mse_epoch_mu_only(model, train_loader, optim, device)
                sched.step()
                val_loss, corr, r2, rmse, crps, crps_base, mu, std, tgt = val_epoch(
                    model, val_loader, loss_fn, device, train_stats=train_stats
                )
                if WANDB_OK:
                    wandb.log({
                        f"{var}_stage":"mu_only",
                        f"{var}_train_mse": tr_mse,
                        f"{var}_val_loss": val_loss,
                        f"{var}_corr": corr,
                        f"{var}_r2": r2,
                        f"{var}_rmse": rmse,
                        f"{var}_crps": crps,
                        f"{var}_crps_baseline": crps_base if crps_base is not None else np.nan,
                        "lr_mu": optim.param_groups[0]["lr"],
                        "epoch": epoch
                    })
                print(f"[μ-only] {var} e{epoch:03d} | MSE {tr_mse:.4f} | ValLoss {val_loss:.4f} | "
                      f"R2 {r2:.3f} | RMSE {rmse:.3f} | CRPS {crps:.3f}")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(model.state_dict(), best_stage1_path)

            if os.path.isfile(best_stage1_path):
                ckpt = torch.load(best_stage1_path, map_location=device)
                model.load_state_dict(ckpt, strict=True)
                print(f"Loaded best Stage-1 weights from: {best_stage1_path}")
                if hasattr(model, "logvar_prior") and model.logvar_prior is not None:
                    model.logvar_prior = model.logvar_prior.to(device)  # ensure device after load

            # ---- Stage 2: σ-only ----
            for mod in [model.stem, model.layer1, model.layer2, model.layer3, model.layer4, model.pool, model.mu_head, model.skip_mu]:
                set_requires_grad(mod, False)
                set_bn_eval(mod)

            set_requires_grad(model.log_head, True)
            set_requires_grad(model.skip_log, True)

            wd_sigma = 3e-4
            sigma_groups = build_param_groups_sigma_only(model, wd_sigma=wd_sigma, skip_mult=5.0)
            optim = AdamW(sigma_groups, lr=cfg.training.learning_rate, fused=use_fused)
            sched = CosineAnnealingWarmRestarts(optim, T_0=cfg.training.t0, T_mult=2)

            best_crps = float("inf")
            patience = cfg.training.early_stop_patience
            wait = 0
            epochs_full = cfg.training.epochs - cfg.training.mu_epochs

            def var_lambda_for_epoch(local_epoch):
                return cfg.training.var_lambda0 * math.exp(- local_epoch / max(1.0, cfg.training.var_lambda_tau))

            for step in range(epochs_full):
                global_epoch = cfg.training.mu_epochs + step
                lam = var_lambda_for_epoch(step)
                tr_loss = nll_epoch_sigma_only(model, train_loader, optim, device, loss_fn, lam)
                val_loss, corr, r2, rmse, crps, crps_base, mu, std, tgt = val_epoch(
                    model, val_loader, loss_fn, device, train_stats=train_stats
                )
                sched.step()

                crps_impr = None
                if crps_base is not None and crps_base > 0:
                    crps_impr = 100.0 * (crps_base - crps) / crps_base

                if WANDB_OK:
                    wandb.log({
                        f"{var}_stage":"sigma_only",
                        f"{var}_train_loss": tr_loss,
                        f"{var}_val_loss": val_loss,
                        f"{var}_corr": corr,
                        f"{var}_r2": r2,
                        f"{var}_rmse": rmse,
                        f"{var}_crps": crps,
                        f"{var}_crps_baseline": crps_base if crps_base is not None else np.nan,
                        f"{var}_crps_improvement_pct": crps_impr if crps_impr is not None else np.nan,
                        f"{var}_var_lambda": lam,
                        "lr_sigma": optim.param_groups[0]["lr"],
                        "epoch": global_epoch
                    })

                print(f"[σ-only] {var} e{global_epoch:03d} | Val {val_loss:.4f} | R2 {r2:.3f} | RMSE {rmse:.3f} | "
                      f"CRPS {crps:.3f} (base {crps_base:.3f} ⇒ Δ {crps_impr:+.1f}%) | λ {lam:.2e} | wait {wait}")

                # Plot EVERY σ-epoch
                plot_timeseries_uncert(mu, std, tgt, global_epoch, var)

                if crps < best_crps:
                    best_crps = crps
                    wait = 0
                    torch.save(model.state_dict(), os.path.join(weight_dir, "best_crps.pt"))
                else:
                    wait += 1
                if wait >= patience:
                    print(f"Early stopping on CRPS (patience={patience}).")
                    break

            results.append({"var": var, "val_model": v_model, "best_val": best_val, "best_crps": best_crps})

    # Save summary
    out_csv = os.path.join(get_original_cwd(), "results_bnn.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    if WANDB_OK: wandb.finish()
    print(f"Saved results to {out_csv}")

if __name__ == "__main__":
    main()
