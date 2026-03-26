import os
import json
import argparse
import random
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from src.model3_bilstm_train import (
    set_seed, get_device, ensure_dir,
    load_dataframe, pick_feature_columns, make_sequences, split_by_ratio,
    SeqDataset, BiLSTMRegressor, rmse, mae, r2_score
)

try:
    import optuna
except Exception as e:
    optuna = None


@torch.no_grad()
def quick_eval_rmse(model, loader, device, y_scaler: StandardScaler, target_mode: str) -> float:
    model.eval()
    ys_true_abs, ys_pred_abs = [], []
    for Xb, yb_scaled, base_y, _ in loader:
        Xb = Xb.to(device)
        pred_scaled = model(Xb).cpu().numpy()
        y_true = y_scaler.inverse_transform(yb_scaled.numpy()).reshape(-1)
        y_pred = y_scaler.inverse_transform(pred_scaled).reshape(-1)
        base_y = base_y.numpy().reshape(-1)
        if target_mode == "delta":
            y_true_abs = base_y + y_true
            y_pred_abs = base_y + y_pred
        else:
            y_true_abs = y_true
            y_pred_abs = y_pred
        ys_true_abs.append(y_true_abs)
        ys_pred_abs.append(y_pred_abs)
    y_true_abs = np.concatenate(ys_true_abs, axis=0)
    y_pred_abs = np.concatenate(ys_pred_abs, axis=0)
    return rmse(y_true_abs, y_pred_abs)


def objective(trial, base_args, df, feature_cols):
    device = get_device()

    # search space
    hidden = trial.suggest_categorical("hidden", [32, 64, 128, 256])
    layers = trial.suggest_int("layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    fc = trial.suggest_categorical("fc", [32, 64, 128])
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    # build sequences
    X, y, base_y, t_out = make_sequences(
        df, feature_cols, base_args.target,
        lookback=base_args.lookback,
        horizon=base_args.horizon,
        target_mode=base_args.target_mode
    )
    (Xtr, ytr, btr, ttr), (Xva, yva, bva, tva), _ = split_by_ratio(
        X, y, base_y, t_out,
        train_ratio=base_args.train_ratio,
        val_ratio=base_args.val_ratio
    )

    # scale
    F = Xtr.shape[-1]
    x_scaler = StandardScaler()
    Xtr2 = x_scaler.fit_transform(Xtr.reshape(-1, F)).reshape(Xtr.shape)
    Xva2 = x_scaler.transform(Xva.reshape(-1, F)).reshape(Xva.shape)

    y_scaler = StandardScaler()
    ytr_scaled = y_scaler.fit_transform(ytr.reshape(-1, 1)).reshape(-1)
    yva_scaled = y_scaler.transform(yva.reshape(-1, 1)).reshape(-1)

    dl_tr = DataLoader(SeqDataset(Xtr2, ytr_scaled, btr, ttr), batch_size=batch_size, shuffle=True)
    dl_va = DataLoader(SeqDataset(Xva2, yva_scaled, bva, tva), batch_size=batch_size, shuffle=False)

    model = BiLSTMRegressor(n_features=F, hidden=hidden, layers=layers, dropout=dropout, fc=fc).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best = float("inf")
    bad = 0

    for epoch in range(1, base_args.max_epochs + 1):
        model.train()
        for Xb, yb, _, _ in dl_tr:
            Xb = Xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(Xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val_rmse = quick_eval_rmse(model, dl_va, device, y_scaler, base_args.target_mode)
        if val_rmse < best - 1e-9:
            best = val_rmse
            bad = 0
        else:
            bad += 1

        trial.report(best, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if bad >= base_args.patience:
            break

    return best


def main():
    if optuna is None:
        raise RuntimeError("optuna is not installed. Run: python -m pip install optuna")

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--target", type=str, required=True)
    ap.add_argument("--target_mode", type=str, default="delta", choices=["abs", "delta"])
    ap.add_argument("--lookback", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=1)

    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--max_epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)

    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--drop_compressor_states", action="store_true")
    ap.add_argument("--include_target_lag", action="store_true")

    ap.add_argument("--outdir", type=str, default="runs/model3_bilstm/hpo")
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.outdir)

    df = load_dataframe(args.csv)
    if args.target not in df.columns:
        raise ValueError(f"target column not found: {args.target}")

    feature_cols = pick_feature_columns(
        df, args.target,
        drop_compressor_states=args.drop_compressor_states,
        include_target_lag=args.include_target_lag
    )

    print("=== BiLSTM HPO (Optuna) ===")
    print(f"target: {args.target} | target_mode: {args.target_mode}")
    print(f"lookback: {args.lookback} | horizon: {args.horizon}")
    print(f"features: {len(feature_cols)}")
    print(f"trials: {args.trials} | max_epochs: {args.max_epochs} | patience: {args.patience}")
    print(f"outdir: {args.outdir}")

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, args, df, feature_cols), n_trials=args.trials)

    best = {
        "best_value(val_rmse_abs)": float(study.best_value),
        "best_params": dict(study.best_params),
        "fixed": {
            "target": args.target,
            "target_mode": args.target_mode,
            "lookback": args.lookback,
            "horizon": args.horizon,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "drop_compressor_states": args.drop_compressor_states,
            "include_target_lag": args.include_target_lag,
        },
        "feature_count": len(feature_cols),
    }

    best_path = os.path.join(args.outdir, "model3_bilstm_hpo_best.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    # save trials csv
    rows = []
    for tr in study.trials:
        r = {"trial": tr.number, "value": tr.value, "state": str(tr.state)}
        for k, v in tr.params.items():
            r[k] = v
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "model3_bilstm_hpo_trials.csv"), index=False, encoding="utf-8-sig")

    print("[OK] saved:", best_path)
    print("best_value(val_rmse_abs):", float(study.best_value))
    print("best_params:", study.best_params)


if __name__ == "__main__":
    main()
