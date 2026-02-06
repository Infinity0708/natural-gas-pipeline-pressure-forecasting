# src/model2_cnn1d_training.py
from __future__ import annotations

import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


# -----------------------------
# Utils
# -----------------------------
def read_csv_auto(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# -----------------------------
# Dataset / Model
# -----------------------------
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)                 # (N, L, F)
        self.y = torch.from_numpy(y).unsqueeze(-1)   # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class CNN1DRegressor(nn.Module):
    """Input (B,L,F) -> (B,F,L) conv over time."""
    def __init__(self, n_features: int, channels: int = 128, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel_size, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (B,L,F)->(B,F,L)
        return self.net(x)


# -----------------------------
# Windowing + split
# -----------------------------
def guess_time_col(df: pd.DataFrame) -> str:
    if "时间" in df.columns:
        return "时间"
    return df.columns[0]


def split_time(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    n_tr = int(n * train_ratio)
    n_va = int(n * val_ratio)
    tr = df.iloc[:n_tr].copy().reset_index(drop=True)
    va = df.iloc[n_tr:n_tr+n_va].copy().reset_index(drop=True)
    te = df.iloc[n_tr+n_va:].copy().reset_index(drop=True)
    return tr, va, te


def make_windows(df_s: pd.DataFrame, feature_cols: list[str], target_col: str, lookback: int, horizon: int):
    X = df_s[feature_cols].to_numpy(dtype=np.float32)
    y = df_s[target_col].to_numpy(dtype=np.float32)
    t = df_s["time"].to_numpy()

    X_out, y_out, t_out = [], [], []
    start = lookback - 1
    end = len(df_s) - 1 - horizon

    for i in range(start, end + 1):
        x_win = X[i - lookback + 1:i + 1]
        y_tar = y[i + horizon]
        if np.any(np.isnan(x_win)) or np.isnan(y_tar):
            continue
        X_out.append(x_win)
        y_out.append(y_tar)
        t_out.append(t[i + horizon])

    if len(X_out) == 0:
        raise RuntimeError("No samples after windowing (NaNs or lookback/horizon too large).")

    return np.stack(X_out, 0), np.asarray(y_out, dtype=np.float32), np.asarray(t_out)


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# -----------------------------
# Train / Eval
# -----------------------------
def eval_loader(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy().squeeze()
            ys.append(yb.numpy().squeeze())
            ps.append(pred)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return y, p


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="raw/数据处理结果.csv")
    ap.add_argument("--target", default="出站压力-连木沁压气站")
    ap.add_argument("--lookback", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=1)

    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--patience", type=int, default=15)

    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--kernel_size", type=int, default=7)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--drop_compressor_states", action="store_true")
    ap.add_argument("--include_target_lag", action="store_true", default=True)

    ap.add_argument("--outdir", default="runs/model2_cnn1d")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # output files (exactly like your screenshot)
    f_best = outdir / "model2_cnn1d_best.pt"
    f_report = outdir / "model2_cnn1d_report.json"
    f_pred = outdir / "model2_cnn1d_test_predictions.csv"
    f_loss_png = outdir / "model2_cnn1d_loss_curve.png"
    f_fore_png = outdir / "model2_cnn1d_forecast_curve.png"

    # -------------------- load data --------------------
    df = read_csv_auto(Path(args.csv))
    df.columns = [str(c).strip() for c in df.columns]

    tcol = guess_time_col(df)
    df["time"] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    for c in df.columns:
        if c in ["time", tcol]:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if args.target not in df.columns:
        raise ValueError(f"Target not found: {args.target}")

    # features: all numeric except time + target
    feature_cols = [c for c in df.columns if c not in ["time", tcol, args.target]]
    if args.drop_compressor_states:
        feature_cols = [c for c in feature_cols if not str(c).startswith("压缩机启停-")]

    tr, va, te = split_time(df, 0.7, 0.15)

    # scalers (fit on train only)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    Xtr = x_scaler.fit_transform(tr[feature_cols].to_numpy(dtype=np.float32))
    Xva = x_scaler.transform(va[feature_cols].to_numpy(dtype=np.float32))
    Xte = x_scaler.transform(te[feature_cols].to_numpy(dtype=np.float32))

    ytr = y_scaler.fit_transform(tr[[args.target]].to_numpy(dtype=np.float32)).squeeze()
    yva = y_scaler.transform(va[[args.target]].to_numpy(dtype=np.float32)).squeeze()
    yte = y_scaler.transform(te[[args.target]].to_numpy(dtype=np.float32)).squeeze()

    tr_s = tr.copy(); va_s = va.copy(); te_s = te.copy()
    tr_s[feature_cols] = Xtr; va_s[feature_cols] = Xva; te_s[feature_cols] = Xte
    tr_s[args.target] = ytr; va_s[args.target] = yva; te_s[args.target] = yte

    final_features = feature_cols.copy()
    if args.include_target_lag:
        tr_s["__target_lag"] = ytr
        va_s["__target_lag"] = yva
        te_s["__target_lag"] = yte
        final_features.append("__target_lag")

    X_train, y_train, _ = make_windows(tr_s, final_features, args.target, args.lookback, args.horizon)
    X_val, y_val, _ = make_windows(va_s, final_features, args.target, args.lookback, args.horizon)
    X_test, y_test, t_test = make_windows(te_s, final_features, args.target, args.lookback, args.horizon)

    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SeqDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(SeqDataset(X_test, y_test), batch_size=256, shuffle=False)

    # -------------------- model --------------------
    device = get_device()
    model = CNN1DRegressor(
        n_features=X_train.shape[-1],
        channels=args.channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout
    ).to(device)

    opt = Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # -------------------- train loop (LSTM-style logging) --------------------
    best_val = float("inf")
    best_epoch = -1
    bad = 0

    hist_train = []
    hist_val = []

    print("=== CNN1D Training (LSTM-style logs) ===")
    print("device:", device)
    print("target:", args.target)
    print("lookback:", args.lookback, "horizon:", args.horizon)
    print("features:", len(final_features))
    print("train/val/test samples:", len(X_train), len(X_val), len(X_test))
    print("outdir:", outdir)

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses))

        # val (scaled)
        yv, pv = eval_loader(model, val_loader, device=device)
        val_rmse_scaled = float(np.sqrt(mean_squared_error(yv, pv)))
        val_loss = float(mean_squared_error(yv, pv))

        hist_train.append(train_loss)
        hist_val.append(val_loss)

        print(
            f"Epoch {ep:03d}/{args.epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_rmse_scaled={val_rmse_scaled:.6f} | "
            f"bad={bad}/{args.patience}"
        )

        if val_rmse_scaled < best_val - 1e-8:
            best_val = val_rmse_scaled
            best_epoch = ep
            bad = 0
            torch.save(model.state_dict(), f_best)
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    # -------------------- test eval (original scale) --------------------
    model.load_state_dict(torch.load(f_best, map_location="cpu"))
    model.to(device)

    yt_s, yp_s = eval_loader(model, test_loader, device=device)
    yt = y_scaler.inverse_transform(yt_s.reshape(-1, 1)).squeeze()
    yp = y_scaler.inverse_transform(yp_s.reshape(-1, 1)).squeeze()

    test_rmse = rmse(yt, yp)
    test_mae = float(mean_absolute_error(yt, yp))
    test_r2 = float(r2_score(yt, yp))

    # also compute train/val metrics (original scale) for report consistency
    ytr_s2, ypr_s2 = eval_loader(model, DataLoader(SeqDataset(X_train, y_train), batch_size=256, shuffle=False), device=device)
    yva_s2, ypv_s2 = eval_loader(model, DataLoader(SeqDataset(X_val, y_val), batch_size=256, shuffle=False), device=device)

    ytr_o = y_scaler.inverse_transform(ytr_s2.reshape(-1, 1)).squeeze()
    ypr_o = y_scaler.inverse_transform(ypr_s2.reshape(-1, 1)).squeeze()
    yva_o = y_scaler.inverse_transform(yva_s2.reshape(-1, 1)).squeeze()
    ypv_o = y_scaler.inverse_transform(ypv_s2.reshape(-1, 1)).squeeze()

    report = {
        "model": "CNN1D",
        "device": device,
        "target": args.target,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "n_features": len(final_features),
        "n_samples": {"train": int(len(X_train)), "val": int(len(X_val)), "test": int(len(X_test))},
        "best_epoch": int(best_epoch),
        "best_val_rmse_scaled": float(best_val),
        "hparams": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "channels": args.channels,
            "kernel_size": args.kernel_size,
            "dropout": args.dropout,
            "drop_compressor_states": bool(args.drop_compressor_states),
            "include_target_lag": bool(args.include_target_lag),
        },
        "metrics": {
            "train": {
                "rmse": rmse(ytr_o, ypr_o),
                "mae": float(mean_absolute_error(ytr_o, ypr_o)),
                "r2": float(r2_score(ytr_o, ypr_o)),
            },
            "val": {
                "rmse": rmse(yva_o, ypv_o),
                "mae": float(mean_absolute_error(yva_o, ypv_o)),
                "r2": float(r2_score(yva_o, ypv_o)),
            },
            "test": {
                "rmse": test_rmse,
                "mae": test_mae,
                "r2": test_r2,
            }
        }
    }

    with open(f_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # -------------------- save predictions --------------------
    pd.DataFrame({
        "time": pd.to_datetime(t_test),
        "y_true": yt,
        "y_pred": yp
    }).to_csv(f_pred, index=False, encoding="utf-8-sig")

    # -------------------- plots --------------------
    # loss curve
    fig = plt.figure(figsize=(8, 3))
    plt.plot(hist_train, label="train_loss")
    plt.plot(hist_val, label="val_loss")
    plt.title("CNN1D Loss Curve")
    plt.xlabel("epoch")
    plt.ylabel("MSE (scaled)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(f_loss_png, dpi=200)
    plt.close(fig)

    # forecast curve (first K points)
    K = min(2000, len(yt))
    tt = pd.to_datetime(t_test[:K])
    fig = plt.figure(figsize=(10, 4))
    plt.plot(tt, yt[:K], label="True")
    plt.plot(tt, yp[:K], label="Pred")
    plt.title(f"CNN1D Forecast Curve (first {K} test points)")
    plt.xlabel("time")
    plt.ylabel("target (original scale)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(f_fore_png, dpi=200)
    plt.close(fig)

    print("=== CNN1D Training Done ===")
    print("saved:", f_best)
    print("saved:", f_report)
    print("saved:", f_fore_png)
    print("saved:", f_loss_png)
    print("saved:", f_pred)
    print("test_mae:", test_mae)
    print("test_rmse:", test_rmse)


if __name__ == "__main__":
    main()
