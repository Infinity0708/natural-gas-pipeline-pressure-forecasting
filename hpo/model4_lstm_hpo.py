# hpo/model4_lstm_hpo.py
import os
import sys
import json
import math
import argparse

import numpy as np
import torch
import optuna

# allow "from src..." when running from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model4_lstm_train import (
    set_seed, get_device, safe_makedirs,
    prepare_data, SeqDataset, LSTMRegressor, eval_epoch
)
from torch.utils.data import DataLoader
import torch.nn as nn


def objective(trial, base_args, built, y_scaler):
    device = get_device()

    hidden = trial.suggest_categorical("hidden", [32, 64, 128, 256])
    layers = trial.suggest_int("layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    fc = trial.suggest_categorical("fc", [32, 64, 128])

    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)

    model = LSTMRegressor(
        n_features=built.n_features,
        hidden=hidden,
        layers=layers,
        dropout=dropout,
        fc=fc,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    dl_tr = DataLoader(SeqDataset(built.X_tr, built.y_tr, built.base_tr, built.t_tr),
                       batch_size=batch_size, shuffle=False, drop_last=False)
    dl_va = DataLoader(SeqDataset(built.X_va, built.y_va, built.base_va, built.t_va),
                       batch_size=batch_size, shuffle=False, drop_last=False)

    best = float("inf")
    bad = 0

    for ep in range(1, base_args.max_epochs + 1):
        model.train()
        losses = []
        for Xb, yb, _, _ in dl_tr:
            Xb = Xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            yhat = model(Xb)
            loss = loss_fn(yhat, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), base_args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        val_rmse_abs, _, _, _ = eval_epoch(model, dl_va, device, y_scaler, base_args.target_mode)

        # report to optuna
        trial.report(val_rmse_abs, step=ep)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_rmse_abs + 1e-12 < best:
            best = val_rmse_abs
            bad = 0
        else:
            bad += 1
        if bad >= base_args.patience:
            break

    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="raw/数据处理结果.csv")
    ap.add_argument("--target", type=str, default="出站压力-连木沁压气站")
    ap.add_argument("--target_mode", type=str, choices=["abs", "delta"], default="delta")
    ap.add_argument("--lookback", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=1)

    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--max_epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--drop_compressor_states", action="store_true")
    ap.add_argument("--include_target_lag", action="store_true")

    ap.add_argument("--outdir", type=str, default="runs/model4_lstm/hpo_delta_h1")
    args = ap.parse_args()

    set_seed(args.seed)
    safe_makedirs(args.outdir)

    built, x_scaler, y_scaler, _ = prepare_data(
        csv_path=args.csv,
        target=args.target,
        lookback=args.lookback,
        horizon=args.horizon,
        target_mode=args.target_mode,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        drop_compressor_states=args.drop_compressor_states,
        include_target_lag=args.include_target_lag,
    )

    print("=== LSTM HPO (Optuna) ===")
    print(f"target: {args.target} | target_mode: {args.target_mode}")
    print(f"lookback: {args.lookback} | horizon: {args.horizon}")
    print(f"features: {built.n_features}")
    print(f"trials: {args.trials} | max_epochs: {args.max_epochs} | patience: {args.patience}")
    print(f"outdir: {args.outdir}")

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(lambda t: objective(t, args, built, y_scaler), n_trials=args.trials)

    best = {
        "best_value(val_rmse_abs)": float(study.best_value),
        "best_params": dict(study.best_params),
        "fixed_args": {
            "csv": args.csv,
            "target": args.target,
            "target_mode": args.target_mode,
            "lookback": args.lookback,
            "horizon": args.horizon,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "drop_compressor_states": bool(args.drop_compressor_states),
            "include_target_lag": bool(args.include_target_lag),
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "grad_clip": args.grad_clip,
        }
    }

    out_path = os.path.join(args.outdir, "model4_lstm_hpo_best.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved: {out_path}")
    print(f"best_value(val_rmse_abs): {study.best_value}")
    print(f"best_params: {study.best_params}")


if __name__ == "__main__":
    main()
