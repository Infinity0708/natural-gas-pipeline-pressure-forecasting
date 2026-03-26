from __future__ import annotations
import os
import json
from pathlib import Path
import random

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


# -------------------------
# utils
# -------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_csv_auto(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def guess_time_col(df: pd.DataFrame) -> str:
    if "时间" in df.columns:
        return "时间"
    return df.columns[0]


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# -------------------------
# dataset / model
# -------------------------
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y_scaled: np.ndarray):
        # X: (N, L, F), y_scaled: (N,)
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y_scaled.astype(np.float32)).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class CNN1DRegressor(nn.Module):
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
        # x: (B, L, F) -> (B, F, L)
        x = x.transpose(1, 2)
        return self.net(x)


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


# -------------------------
# windowing (supports abs/delta target)
# -------------------------
def make_windows_from_arrays(
    time_arr: np.ndarray,
    X_feat_scaled: np.ndarray,       # (T, F)
    target_raw: np.ndarray,          # (T,)
    lookback: int,
    horizon: int,
    target_mode: str = "abs",        # "abs" or "delta"
):
    """
    Build supervised samples AFTER row-scaling features.
    - X window uses scaled features.
    - y is computed on RAW target:
        abs:   y = P_{t+h}
        delta: y = P_{t+h} - P_t   (P_t is last value in window)
    Also returns:
      - y_abs: P_{t+h} (raw)
      - last_raw: P_t (raw)
      - t_out: time at t+h
    """
    X_out, y_out, y_abs_out, last_out, t_out = [], [], [], [], []

    start = lookback - 1
    end = len(time_arr) - 1 - horizon
    for i in range(start, end + 1):
        x_win = X_feat_scaled[i - lookback + 1:i + 1]  # (L, F)

        p_t = float(target_raw[i])              # last in window
        p_th = float(target_raw[i + horizon])   # target at horizon

        if np.any(np.isnan(x_win)) or np.isnan(p_t) or np.isnan(p_th):
            continue

        if target_mode == "delta":
            y_val = p_th - p_t
        else:
            y_val = p_th

        X_out.append(x_win)
        y_out.append(y_val)
        y_abs_out.append(p_th)
        last_out.append(p_t)
        t_out.append(time_arr[i + horizon])

    if len(X_out) == 0:
        raise RuntimeError("No samples after windowing (NaNs or lookback/horizon too large).")

    return (
        np.stack(X_out, 0).astype(np.float32),
        np.asarray(y_out, dtype=np.float32),
        np.asarray(y_abs_out, dtype=np.float32),
        np.asarray(last_out, dtype=np.float32),
        np.asarray(t_out),
    )


def main():
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", default="raw/processed_data.csv")
    ap.add_argument("--target", default="出站压力-连木沁压气站")
    ap.add_argument("--lookback", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=1)

    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.1)

    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--patience", type=int, default=15)

    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--kernel_size", type=int, default=7)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--drop_compressor_states", action="store_true")

    # --- FIXED: target lag switch really works ---
    # default: include target lag (recommended)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--include_target_lag", action="store_true", help="Include target lag as a feature (default ON).")
    g.add_argument("--no_target_lag", action="store_true", help="Disable target lag feature.")

    # --- NEW: target mode ---
    ap.add_argument("--target_mode", choices=["abs", "delta"], default="abs",
                    help="abs: predict P(t+h); delta: predict P(t+h)-P(t) then reconstruct.")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="runs/model2_cnn1d")
    args = ap.parse_args()

    set_seed(args.seed)

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    f_best = outdir / "model2_cnn1d_best.pt"
    f_report = outdir / "model2_cnn1d_report.json"
    f_pred = outdir / "model2_cnn1d_test_predictions.csv"
    f_loss_png = outdir / "model2_cnn1d_loss_curve.png"
    f_fore_png = outdir / "model2_cnn1d_forecast_curve.png"

    # default include target lag ON, unless explicitly disabled
    include_target_lag = True
    if args.no_target_lag:
        include_target_lag = False
    if args.include_target_lag:
        include_target_lag = True

    df = read_csv_auto(Path(args.csv))
    df.columns = [str(c).strip() for c in df.columns]

    tcol = guess_time_col(df)
    df["time"] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    # numeric coercion
    for c in df.columns:
        if c in ["time", tcol]:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if args.target not in df.columns:
        raise ValueError(f"Target not found: {args.target}")

    # feature cols
    feature_cols = [c for c in df.columns if c not in ["time", tcol, args.target]]
    if args.drop_compressor_states:
        feature_cols = [c for c in feature_cols if not str(c).startswith("压缩机启停-")]

    # working df (raw)
    work = df[["time", args.target] + feature_cols].copy()

    # add target lag feature (raw) if enabled
    feat_cols2 = list(feature_cols)
    if include_target_lag:
        work["__target_lag"] = work[args.target].astype(float)  # lag feature (raw)
        feat_cols2 = feat_cols2 + ["__target_lag"]

    # prepare arrays
    time_arr = work["time"].to_numpy()
    target_raw = work[args.target].to_numpy(dtype=np.float32)

    # fit x_scaler on TRAIN ROWS only (raw rows)
    n_rows = len(work)
    n_tr_rows = int(n_rows * args.train_ratio)
    train_rows = work.iloc[:n_tr_rows].copy()

    x_scaler = StandardScaler()
    x_scaler.fit(train_rows[feat_cols2].to_numpy(np.float32))

    # transform features for FULL rows
    X_feat_scaled = x_scaler.transform(work[feat_cols2].to_numpy(np.float32)).astype(np.float32)

    # WINDOW ON FULL SERIES FIRST (time-aware)
    X_all, y_raw_all, y_abs_all, last_raw_all, t_out = make_windows_from_arrays(
        time_arr=time_arr,
        X_feat_scaled=X_feat_scaled,
        target_raw=target_raw,
        lookback=args.lookback,
        horizon=args.horizon,
        target_mode=args.target_mode
    )

    # split SAMPLES by time order (t_out already time-ordered)
    N = len(t_out)
    N_tr = int(N * args.train_ratio)
    N_va = int(N * args.val_ratio)

    X_train = X_all[:N_tr]
    X_val = X_all[N_tr:N_tr + N_va]
    X_test = X_all[N_tr + N_va:]

    y_train_raw = y_raw_all[:N_tr]
    y_val_raw = y_raw_all[N_tr:N_tr + N_va]
    y_test_raw = y_raw_all[N_tr + N_va:]

    y_train_abs = y_abs_all[:N_tr]
    y_val_abs = y_abs_all[N_tr:N_tr + N_va]
    y_test_abs = y_abs_all[N_tr + N_va:]

    last_train = last_raw_all[:N_tr]
    last_val = last_raw_all[N_tr:N_tr + N_va]
    last_test = last_raw_all[N_tr + N_va:]

    t_test = t_out[N_tr + N_va:]

    # fit y_scaler on TRAIN SAMPLES only (IMPORTANT)
    y_scaler = StandardScaler()
    y_scaler.fit(y_train_raw.reshape(-1, 1))

    y_train = y_scaler.transform(y_train_raw.reshape(-1, 1)).astype(np.float32).squeeze()
    y_val = y_scaler.transform(y_val_raw.reshape(-1, 1)).astype(np.float32).squeeze()
    y_test = y_scaler.transform(y_test_raw.reshape(-1, 1)).astype(np.float32).squeeze()

    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SeqDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(SeqDataset(X_test, y_test), batch_size=256, shuffle=False)

    device = get_device()
    model = CNN1DRegressor(
        n_features=X_train.shape[-1],
        channels=args.channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout
    ).to(device)

    opt = Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_epoch = -1
    bad = 0
    hist_train, hist_val = [], []

    print("=== CNN1D Training (LSTM-style logs) ===")
    print("device:", device)
    print("target:", args.target)
    print("target_mode:", args.target_mode)
    print("lookback:", args.lookback, "horizon:", args.horizon)
    print("features:", X_train.shape[-1])
    print("train/val/test samples:", len(X_train), len(X_val), len(X_test))
    print("outdir:", outdir)
    print("split ratios:", args.train_ratio, args.val_ratio, 1.0 - args.train_ratio - args.val_ratio)
    print("seed:", args.seed)
    print("drop_compressor_states:", bool(args.drop_compressor_states), "| include_target_lag:", bool(include_target_lag))

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

        yv_s, pv_s = eval_loader(model, val_loader, device=device)
        val_rmse_scaled = float(np.sqrt(mean_squared_error(yv_s, pv_s)))
        val_loss = float(mean_squared_error(yv_s, pv_s))

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

    # ---- TEST on original pressure scale ----
    model.load_state_dict(torch.load(f_best, map_location="cpu"))
    model.to(device)

    yt_s, yp_s = eval_loader(model, test_loader, device=device)
    yt_raw = y_scaler.inverse_transform(yt_s.reshape(-1, 1)).squeeze()
    yp_raw = y_scaler.inverse_transform(yp_s.reshape(-1, 1)).squeeze()

    # reconstruct to absolute pressure if delta mode
    if args.target_mode == "delta":
        yt = last_test + yt_raw
        yp = last_test + yp_raw
        y_true_abs = y_test_abs
    else:
        yt = yt_raw
        yp = yp_raw
        y_true_abs = y_test_abs

    # metrics (absolute pressure space)
    test_rmse = rmse(y_true_abs, yp)
    test_mae = float(mean_absolute_error(y_true_abs, yp))
    test_r2 = float(r2_score(y_true_abs, yp))

    # train/val original-scale metrics for report
    def eval_abs_for_split(X_split, y_scaled_split, last_split, y_abs_split):
        loader = DataLoader(SeqDataset(X_split, y_scaled_split), batch_size=256, shuffle=False)
        y_s, p_s = eval_loader(model, loader, device=device)
        y_raw = y_scaler.inverse_transform(y_s.reshape(-1, 1)).squeeze()
        p_raw = y_scaler.inverse_transform(p_s.reshape(-1, 1)).squeeze()
        if args.target_mode == "delta":
            p_abs = last_split + p_raw
        else:
            p_abs = p_raw
        return y_abs_split, p_abs

    ytr_abs, ypr_abs = eval_abs_for_split(X_train, y_train, last_train, y_train_abs)
    yva_abs, ypv_abs = eval_abs_for_split(X_val, y_val, last_val, y_val_abs)

    report = {
        "model": "CNN1D",
        "device": device,
        "seed": args.seed,
        "target": args.target,
        "target_mode": args.target_mode,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "split": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": 1.0 - args.train_ratio - args.val_ratio
        },
        "n_features": int(X_train.shape[-1]),
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
            "include_target_lag": bool(include_target_lag),
        },
        "metrics": {
            "train": {"rmse": rmse(ytr_abs, ypr_abs), "mae": float(mean_absolute_error(ytr_abs, ypr_abs)), "r2": float(r2_score(ytr_abs, ypr_abs))},
            "val":   {"rmse": rmse(yva_abs, ypv_abs), "mae": float(mean_absolute_error(yva_abs, ypv_abs)), "r2": float(r2_score(yva_abs, ypv_abs))},
            "test":  {"rmse": test_rmse, "mae": test_mae, "r2": test_r2},
        }
    }

    with open(f_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # save predictions (absolute pressure)
    pd.DataFrame({
        "time": pd.to_datetime(t_test),
        "y_true": y_true_abs,
        "y_pred": yp
    }).to_csv(f_pred, index=False, encoding="utf-8-sig")

    # loss curve
    fig = plt.figure(figsize=(8, 3))
    plt.plot(hist_train, label="train_loss")
    plt.plot(hist_val, label="val_loss")
    plt.title("CNN1D Loss Curve")
    plt.xlabel("epoch")
    plt.ylabel("MSE (scaled target space)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(f_loss_png, dpi=200)
    plt.close(fig)

    # forecast curve
    K = min(2000, len(y_true_abs))
    tt = pd.to_datetime(t_test[:K])
    fig = plt.figure(figsize=(10, 4))
    plt.plot(tt, y_true_abs[:K], label="True")
    plt.plot(tt, yp[:K], label="Pred")
    plt.title(f"CNN1D Forecast Curve (first {K} test points)")
    plt.xlabel("time")
    plt.ylabel("pressure (original scale)")
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
