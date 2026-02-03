import os
import json
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt


# -----------------------------
# Utils
# -----------------------------
def read_csv_auto(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # last try no encoding
    return pd.read_csv(path)


def safe_name(s: str) -> str:
    # for filenames / plotting labels (avoid CJK font warnings)
    keep = []
    for ch in str(s):
        if ch.isalnum() or ch in ["-", "_", "."]:
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def train_val_test_split_time(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    df_train = df.iloc[:n_train].copy().reset_index(drop=True)
    df_val = df.iloc[n_train:n_train + n_val].copy().reset_index(drop=True)
    df_test = df.iloc[n_train + n_val:].copy().reset_index(drop=True)
    return df_train, df_val, df_test


def make_supervised_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    lookback: int,
    horizon: int,
):
    """
    Build X: (N, lookback, F), y: (N,)
    Only uses past features up to t, predicts y at t+horizon.
    """
    X, y, t_out = [], [], []
    feat = df[feature_cols].to_numpy(dtype=np.float32)
    tgt = df[target_col].to_numpy(dtype=np.float32)
    time_arr = df["time"].to_numpy()

    n = len(df)
    start = lookback - 1
    end = n - 1 - horizon
    for t in range(start, end + 1):
        x_win = feat[t - lookback + 1: t + 1, :]
        y_out = tgt[t + horizon]
        if np.any(np.isnan(x_win)) or np.isnan(y_out):
            continue
        X.append(x_win)
        y.append(y_out)
        t_out.append(time_arr[t + horizon])

    if len(X) == 0:
        return None, None, None

    X = np.stack(X, axis=0)   # (N, L, F)
    y = np.asarray(y, dtype=np.float32)
    t_out = np.asarray(t_out)
    return X, y, t_out


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Dataset
# -----------------------------
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)  # (N,L,F)
        self.y = torch.from_numpy(y).unsqueeze(-1)  # (N,1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------
# CNN1D model
# -----------------------------
class CNN1DForecaster(nn.Module):
    """
    Input: (B, L, F)
    We treat F as channels by transposing to (B, F, L) and apply Conv1d along time.
    """
    def __init__(self, n_features: int, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        pad = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel_size, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.AdaptiveAvgPool1d(1),  # (B, channels, 1)
            nn.Flatten(),             # (B, channels)
            nn.Linear(channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, L, F) -> (B, F, L)
        x = x.transpose(1, 2)
        return self.net(x)


# -----------------------------
# Training helpers
# -----------------------------
@dataclass
class TrainReport:
    model: str
    target: str
    lookback: int
    horizon: int
    n_features: int
    train_samples: int
    val_samples: int
    test_samples: int
    best_epoch: int
    best_val_rmse: float
    train_mae: float
    train_rmse: float
    val_mae: float
    val_rmse: float
    test_mae: float
    test_rmse: float
    params: dict


def eval_model(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            y_true.append(yb.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
    y_true = np.vstack(y_true).squeeze()
    y_pred = np.vstack(y_pred).squeeze()
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return mae, rmse, y_true, y_pred


def plot_loss_curve(losses_train, losses_val, out_path):
    plt.figure()
    plt.plot(losses_train, label="train")
    plt.plot(losses_val, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss (MSE)")
    plt.title("CNN1D Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_forecast_curve(t, y_true, y_pred, out_path, title="CNN1D Forecast Curve"):
    plt.figure(figsize=(10, 4))
    # plot a subset for readability
    n = len(y_true)
    k = min(n, 2000)
    plt.plot(pd.to_datetime(t[:k]), y_true[:k], label="Actual")
    plt.plot(pd.to_datetime(t[:k]), y_pred[:k], label="Forecast")
    plt.xlabel("time")
    plt.ylabel("target")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="raw/数据处理结果.csv")
    ap.add_argument("--target", default="出站压力-连木沁压气站")
    ap.add_argument("--lookback", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=1)

    # CNN params
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--kernel_size", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.2)

    # training
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    # features
    ap.add_argument("--max_features", type=int, default=0,
                    help="0=use all numeric features except time/target; >0=keep first N")
    ap.add_argument("--drop_compressor_states", type=int, default=0,
                    help="1=drop columns that start with '压缩机启停-'")

    # output
    ap.add_argument("--outdir", default="runs/model2_cnn1d")
    args = ap.parse_args()

    set_seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    df = read_csv_auto(args.csv)
    df.columns = [str(c).strip() for c in df.columns]

    time_col = "时间" if "时间" in df.columns else df.columns[0]
    df["time"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    if args.target not in df.columns:
        raise ValueError(f"Target not found: {args.target}")

    # numeric conversion
    for c in df.columns:
        if c in ["time", time_col]:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # feature cols
    numeric_cols = [c for c in df.columns if c not in ["time", time_col] and c != args.target]
    if args.drop_compressor_states == 1:
        numeric_cols = [c for c in numeric_cols if not str(c).startswith("压缩机启停-")]

    if args.max_features and args.max_features > 0:
        numeric_cols = numeric_cols[:args.max_features]

    if len(numeric_cols) == 0:
        raise ValueError("No feature columns found after filtering.")

    # split
    df_tr, df_va, df_te = train_val_test_split_time(df, 0.7, 0.15)

    # scale features + target (fit on train only)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    Xtr_raw = df_tr[numeric_cols].to_numpy(dtype=np.float32)
    Xva_raw = df_va[numeric_cols].to_numpy(dtype=np.float32)
    Xte_raw = df_te[numeric_cols].to_numpy(dtype=np.float32)

    ytr_raw = df_tr[[args.target]].to_numpy(dtype=np.float32)
    yva_raw = df_va[[args.target]].to_numpy(dtype=np.float32)
    yte_raw = df_te[[args.target]].to_numpy(dtype=np.float32)

    Xtr = x_scaler.fit_transform(Xtr_raw)
    Xva = x_scaler.transform(Xva_raw)
    Xte = x_scaler.transform(Xte_raw)

    ytr = y_scaler.fit_transform(ytr_raw)
    yva = y_scaler.transform(yva_raw)
    yte = y_scaler.transform(yte_raw)

    # overwrite scaled back into dfs
    df_tr_s = df_tr.copy()
    df_va_s = df_va.copy()
    df_te_s = df_te.copy()

    df_tr_s[numeric_cols] = Xtr
    df_va_s[numeric_cols] = Xva
    df_te_s[numeric_cols] = Xte

    df_tr_s[args.target] = ytr.squeeze()
    df_va_s[args.target] = yva.squeeze()
    df_te_s[args.target] = yte.squeeze()

    # build sequences
    Xtr_seq, ytr_seq, ttr = make_supervised_sequences(df_tr_s, numeric_cols, args.target, args.lookback, args.horizon)
    Xva_seq, yva_seq, tva = make_supervised_sequences(df_va_s, numeric_cols, args.target, args.lookback, args.horizon)
    Xte_seq, yte_seq, tte = make_supervised_sequences(df_te_s, numeric_cols, args.target, args.lookback, args.horizon)

    if Xtr_seq is None or Xva_seq is None or Xte_seq is None:
        raise RuntimeError("Too few samples after windowing (maybe too many NaNs or too large lookback/horizon).")

    # loaders
    train_loader = DataLoader(SeqDataset(Xtr_seq, ytr_seq), batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(SeqDataset(Xva_seq, yva_seq), batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(SeqDataset(Xte_seq, yte_seq), batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1DForecaster(
        n_features=Xtr_seq.shape[-1],
        channels=args.channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_epoch = -1
    bad = 0

    train_losses, val_losses = [], []
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))

        # val loss
        model.eval()
        v_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                v_losses.append(loss_fn(pred, yb).item())
        val_loss = float(np.mean(v_losses))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # early stopping on val RMSE (scaled space is fine; monotonic)
        _, val_rmse_scaled, _, _ = eval_model(model, val_loader, device)

        if val_rmse_scaled < best_val - 1e-6:
            best_val = val_rmse_scaled
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if bad >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate (scaled)
    train_mae_s, train_rmse_s, ytr_true_s, ytr_pred_s = eval_model(model, train_loader, device)
    val_mae_s, val_rmse_s, yva_true_s, yva_pred_s = eval_model(model, val_loader, device)
    test_mae_s, test_rmse_s, yte_true_s, yte_pred_s = eval_model(model, test_loader, device)

    # Inverse transform to original scale for reporting/plots
    def inv(y_scaled):
        y_scaled = y_scaled.reshape(-1, 1)
        return y_scaler.inverse_transform(y_scaled).squeeze()

    ytr_true = inv(ytr_true_s)
    ytr_pred = inv(ytr_pred_s)
    yva_true = inv(yva_true_s)
    yva_pred = inv(yva_pred_s)
    yte_true = inv(yte_true_s)
    yte_pred = inv(yte_pred_s)

    train_mae = float(mean_absolute_error(ytr_true, ytr_pred))
    train_rmse = float(np.sqrt(mean_squared_error(ytr_true, ytr_pred)))
    val_mae = float(mean_absolute_error(yva_true, yva_pred))
    val_rmse = float(np.sqrt(mean_squared_error(yva_true, yva_pred)))
    test_mae = float(mean_absolute_error(yte_true, yte_pred))
    test_rmse = float(np.sqrt(mean_squared_error(yte_true, yte_pred)))

    # Save outputs (names exactly as requested)
    out_pt = os.path.join(args.outdir, "model2_cnn1d_best.pt")
    out_report = os.path.join(args.outdir, "model2_cnn1d_report.json")
    out_fc = os.path.join(args.outdir, "model2_cnn1d_forecast_curve.png")
    out_lc = os.path.join(args.outdir, "model2_cnn1d_loss_curve.png")
    out_pred = os.path.join(args.outdir, "model2_cnn1d_test_predictions.csv")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "x_scaler_mean": x_scaler.mean_.tolist(),
            "x_scaler_scale": x_scaler.scale_.tolist(),
            "y_scaler_mean": y_scaler.mean_.tolist(),
            "y_scaler_scale": y_scaler.scale_.tolist(),
            "feature_cols": numeric_cols,
            "target": args.target,
            "lookback": args.lookback,
            "horizon": args.horizon,
            "params": {
                "channels": args.channels,
                "kernel_size": args.kernel_size,
                "dropout": args.dropout,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "patience": args.patience,
                "seed": args.seed,
            }
        },
        out_pt
    )

    report = TrainReport(
        model="CNN1D",
        target=args.target,
        lookback=args.lookback,
        horizon=args.horizon,
        n_features=len(numeric_cols),
        train_samples=int(len(ytr_true)),
        val_samples=int(len(yva_true)),
        test_samples=int(len(yte_true)),
        best_epoch=int(best_epoch),
        best_val_rmse=float(best_val),
        train_mae=float(train_mae),
        train_rmse=float(train_rmse),
        val_mae=float(val_mae),
        val_rmse=float(val_rmse),
        test_mae=float(test_mae),
        test_rmse=float(test_rmse),
        params={
            "channels": args.channels,
            "kernel_size": args.kernel_size,
            "dropout": args.dropout,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "seed": args.seed,
            "drop_compressor_states": args.drop_compressor_states,
            "max_features": args.max_features,
        }
    )
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)

    # plots (english title; avoid CJK)
    plot_loss_curve(train_losses, val_losses, out_lc)
    plot_forecast_curve(
        tte,
        yte_true,
        yte_pred,
        out_fc,
        title=f"CNN1D Forecast | horizon={args.horizon}h | lookback={args.lookback}h"
    )

    # save test predictions
    pd.DataFrame(
        {"time": pd.to_datetime(tte), "y_true": yte_true, "y_pred": yte_pred}
    ).to_csv(out_pred, index=False, encoding="utf-8-sig")

    print("=== CNN1D Training Done ===")
    print(f"device: {device}")
    print(f"target: {args.target}")
    print(f"lookback: {args.lookback}")
    print(f"horizon: {args.horizon}")
    print(f"features: {len(numeric_cols)}")
    print(f"train/val/test samples: {len(ytr_true)}/{len(yva_true)}/{len(yte_true)}")
    print(f"best_epoch: {best_epoch}  best_val_rmse(original): {val_rmse:.6f}")
    print(f"test_mae: {test_mae:.6f}")
    print(f"test_rmse: {test_rmse:.6f}")
    print(f"[OK] saved: {args.outdir}")


if __name__ == "__main__":
    main()
