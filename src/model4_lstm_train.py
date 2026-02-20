# src/model4_lstm_train.py
import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt


# ---------------------------
# Utils
# ---------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def safe_makedirs(p: str):
    os.makedirs(p, exist_ok=True)


def find_time_col(df: pd.DataFrame) -> str:
    # common candidates
    for c in ["时间", "time", "timestamp", "datetime", "Time", "DateTime"]:
        if c in df.columns:
            return c
    # fallback: if first column looks like datetime
    c0 = df.columns[0]
    try:
        pd.to_datetime(df[c0])
        return c0
    except Exception:
        return ""


def load_csv_any_encoding(path: str) -> pd.DataFrame:
    # Try utf-8-sig then gbk
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="gbk")


def plot_forecast_curve(y_true: np.ndarray, y_pred: np.ndarray, t: List[str], out_path: str, title: str):
    plt.figure()
    plt.plot(y_true, label="y_true")
    plt.plot(y_pred, label="y_pred")
    plt.legend()
    plt.title(title)
    plt.xlabel("test index")
    plt.ylabel("target")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_loss_curve(train_losses: List[float], val_losses: List[float], out_path: str, title: str):
    plt.figure()
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_rmse_scaled")
    plt.legend()
    plt.title(title)
    plt.xlabel("epoch")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------
# Dataset builder (time series -> supervised)
# ---------------------------
@dataclass
class BuiltData:
    X_tr: np.ndarray
    y_tr: np.ndarray
    base_tr: np.ndarray
    t_tr: List[str]

    X_va: np.ndarray
    y_va: np.ndarray
    base_va: np.ndarray
    t_va: List[str]

    X_te: np.ndarray
    y_te: np.ndarray
    base_te: np.ndarray
    t_te: List[str]

    feature_cols: List[str]
    n_features: int


def build_supervised(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    feature_cols: List[str],
    lookback: int,
    horizon: int,
    target_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
      X: (N, lookback, n_features)
      y: (N,)  -> scaled later
      base: (N,) base absolute target at time t (end of window), used for delta reconstruction
      t_out: list of output timestamps (string) at time t+horizon
    """
    values = df[feature_cols].to_numpy(dtype=np.float32)
    tgt = df[target_col].to_numpy(dtype=np.float32)
    times = df[time_col].astype(str).tolist()

    X_list, y_list, base_list, t_out = [], [], [], []

    T = len(df)
    end_min = lookback - 1
    end_max = T - 1 - horizon
    for end in range(end_min, end_max + 1):
        start = end - lookback + 1
        out_idx = end + horizon

        x = values[start:end + 1, :]
        p_t = float(tgt[end])
        p_f = float(tgt[out_idx])

        if target_mode == "delta":
            y = p_f - p_t
        else:
            y = p_f

        X_list.append(x)
        y_list.append(y)
        base_list.append(p_t)
        t_out.append(times[out_idx])

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    base = np.array(base_list, dtype=np.float32)
    return X, y, base, t_out


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, base: np.ndarray, t_out: List[str]):
        self.X = torch.from_numpy(X)          # (N, L, F)
        self.y = torch.from_numpy(y).float()  # (N,)
        self.base = torch.from_numpy(base).float()  # (N,)
        self.t_out = t_out

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.base[idx], self.t_out[idx]


def time_split_indices(n: int, train_ratio: float, val_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    tr = np.arange(0, n_train)
    va = np.arange(n_train, n_train + n_val)
    te = np.arange(n_train + n_val, n_train + n_val + n_test)
    return tr, va, te


def prepare_data(
    csv_path: str,
    target: str,
    lookback: int,
    horizon: int,
    target_mode: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    drop_compressor_states: bool,
    include_target_lag: bool,
) -> Tuple[BuiltData, StandardScaler, StandardScaler, str]:
    df = load_csv_any_encoding(csv_path)

    time_col = find_time_col(df)
    if not time_col:
        raise ValueError("Cannot find time column. Expect a column like '时间' or 'time'.")

    # parse time + sort
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    if target not in df.columns:
        raise ValueError(f"target column not found: {target}")

    # numeric columns only
    numeric_cols = []
    for c in df.columns:
        if c == time_col:
            continue
        if c == target:
            continue
        # try numeric coercion
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            df[c] = s
            numeric_cols.append(c)

    # optionally drop compressor states
    if drop_compressor_states:
        numeric_cols = [c for c in numeric_cols if "压缩机启停" not in c]

    # include target lag as input feature sequence (safe: within lookback window only)
    if include_target_lag:
        df["__target_lag_feature__"] = pd.to_numeric(df[target], errors="coerce")
        numeric_cols = numeric_cols + ["__target_lag_feature__"]

    # fill NaNs in features (pandas>=3 compatible)
    df[numeric_cols] = df[numeric_cols].interpolate(limit_direction="both")
    df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(0.0)

    # fill NaNs in target (pandas>=3 compatible)
    df[target] = pd.to_numeric(df[target], errors="coerce").interpolate(limit_direction="both")
    df[target] = df[target].ffill().bfill().fillna(0.0)

    # build supervised
    X_all, y_all, base_all, t_all = build_supervised(
        df=df, time_col=time_col, target_col=target, feature_cols=numeric_cols,
        lookback=lookback, horizon=horizon, target_mode=target_mode
    )

    n = X_all.shape[0]
    tr_idx, va_idx, te_idx = time_split_indices(n, train_ratio, val_ratio)

    # scale X using train only: fit on all train timesteps
    x_scaler = StandardScaler()
    Xtr_2d = X_all[tr_idx].reshape(-1, X_all.shape[-1])
    x_scaler.fit(Xtr_2d)

    def transform_X(X):
        X2 = X.reshape(-1, X.shape[-1])
        X2s = x_scaler.transform(X2).astype(np.float32)
        return X2s.reshape(X.shape)

    Xtr = transform_X(X_all[tr_idx])
    Xva = transform_X(X_all[va_idx])
    Xte = transform_X(X_all[te_idx])

    # scale y using train only
    y_scaler = StandardScaler()
    y_scaler.fit(y_all[tr_idx].reshape(-1, 1))

    ytr = y_scaler.transform(y_all[tr_idx].reshape(-1, 1)).reshape(-1).astype(np.float32)
    yva = y_scaler.transform(y_all[va_idx].reshape(-1, 1)).reshape(-1).astype(np.float32)
    yte = y_scaler.transform(y_all[te_idx].reshape(-1, 1)).reshape(-1).astype(np.float32)

    built = BuiltData(
        X_tr=Xtr, y_tr=ytr, base_tr=base_all[tr_idx], t_tr=[t_all[i] for i in tr_idx],
        X_va=Xva, y_va=yva, base_va=base_all[va_idx], t_va=[t_all[i] for i in va_idx],
        X_te=Xte, y_te=yte, base_te=base_all[te_idx], t_te=[t_all[i] for i in te_idx],
        feature_cols=numeric_cols,
        n_features=len(numeric_cols),
    )
    return built, x_scaler, y_scaler, time_col


# ---------------------------
# Model
# ---------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float, fc: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, fc),
            nn.ReLU(),
            nn.Linear(fc, 1),
        )

    def forward(self, x):
        # x: (B, L, F)
        out, _ = self.lstm(x)
        last = out[:, -1, :]       # (B, hidden)
        y = self.head(last).squeeze(-1)
        return y


# ---------------------------
# Train/Eval
# ---------------------------
@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    dl: DataLoader,
    device,
    y_scaler: StandardScaler,
    target_mode: str,
) -> Tuple[float, np.ndarray, np.ndarray, List[str]]:
    model.eval()
    preds_scaled, ys_scaled, bases, times = [], [], [], []

    for Xb, yb, baseb, tb in dl:
        Xb = Xb.to(device)
        yhat = model(Xb).detach().cpu().numpy()
        preds_scaled.append(yhat)
        ys_scaled.append(yb.numpy())
        bases.append(baseb.numpy())
        times.extend(tb)

    preds_scaled = np.concatenate(preds_scaled, axis=0).reshape(-1)
    ys_scaled = np.concatenate(ys_scaled, axis=0).reshape(-1)
    bases = np.concatenate(bases, axis=0).reshape(-1)

    # inverse scale y
    pred_y = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)
    true_y = y_scaler.inverse_transform(ys_scaled.reshape(-1, 1)).reshape(-1)

    # reconstruct abs if delta
    if target_mode == "delta":
        pred_abs = bases + pred_y
        true_abs = bases + true_y
    else:
        pred_abs = pred_y
        true_abs = true_y

    rmse_abs = float(math.sqrt(mean_squared_error(true_abs, pred_abs)))
    return rmse_abs, true_abs, pred_abs, times


def train_main(args):
    set_seed(args.seed)
    device = get_device()

    built, x_scaler, y_scaler, time_col = prepare_data(
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

    safe_makedirs(args.outdir)

    dl_tr = DataLoader(SeqDataset(built.X_tr, built.y_tr, built.base_tr, built.t_tr),
                       batch_size=args.batch_size, shuffle=False, drop_last=False)
    dl_va = DataLoader(SeqDataset(built.X_va, built.y_va, built.base_va, built.t_va),
                       batch_size=args.batch_size, shuffle=False, drop_last=False)
    dl_te = DataLoader(SeqDataset(built.X_te, built.y_te, built.base_te, built.t_te),
                       batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = LSTMRegressor(
        n_features=built.n_features,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        fc=args.fc,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    bad = 0

    train_losses = []
    val_losses = []

    print("=== LSTM Training (LSTM-style logs) ===")
    print(f"device: {device}")
    print(f"target: {args.target}")
    print(f"target_mode: {args.target_mode}")
    print(f"lookback: {args.lookback} horizon: {args.horizon}")
    print(f"features: {built.n_features}")
    print(f"train/val/test samples: {len(built.y_tr)} {len(built.y_va)} {len(built.y_te)}")
    print(f"outdir: {args.outdir}")
    print(f"split ratios: {args.train_ratio} {args.val_ratio} {1-args.train_ratio-args.val_ratio}")
    print(f"seed: {args.seed}")
    print(f"drop_compressor_states: {args.drop_compressor_states} | include_target_lag: {args.include_target_lag}")

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for Xb, yb, _, _ in dl_tr:
            Xb = Xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            yhat = model(Xb)
            loss = loss_fn(yhat, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        train_loss = float(np.mean(losses)) if losses else float("nan")

        # val RMSE in abs-space (used for early stopping)
        val_rmse_abs, _, _, _ = eval_epoch(model, dl_va, device, y_scaler, args.target_mode)

        train_losses.append(train_loss)
        val_losses.append(val_rmse_abs)

        if val_rmse_abs + 1e-12 < best_val:
            best_val = val_rmse_abs
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        print(f"Epoch {ep:03d}/{args.epochs} | train_loss={train_loss:.6f} | val_rmse_abs={val_rmse_abs:.6f} | bad={bad}/{args.patience}")

        if bad >= args.patience:
            print("Early stopping.")
            break

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # final test
    test_rmse, y_true, y_pred, t_out = eval_epoch(model, dl_te, device, y_scaler, args.target_mode)
    test_mae = float(mean_absolute_error(y_true, y_pred))
    test_r2 = float(r2_score(y_true, y_pred))

    # save artifacts
    pt_path = os.path.join(args.outdir, "model4_lstm_best.pt")
    torch.save(model.state_dict(), pt_path)

    report = {
        "model": "LSTM",
        "csv": args.csv,
        "target": args.target,
        "target_mode": args.target_mode,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "n_features": built.n_features,
        "feature_cols": built.feature_cols,
        "split": {"train_ratio": args.train_ratio, "val_ratio": args.val_ratio, "test_ratio": 1 - args.train_ratio - args.val_ratio},
        "n_samples": {"train": int(len(built.y_tr)), "val": int(len(built.y_va)), "test": int(len(built.y_te))},
        "hyperparams": {
            "hidden": args.hidden,
            "layers": args.layers,
            "dropout": args.dropout,
            "fc": args.fc,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "epochs": args.epochs,
            "patience": args.patience,
            "seed": args.seed,
            "drop_compressor_states": args.drop_compressor_states,
            "include_target_lag": args.include_target_lag,
        },
        "best_epoch": int(best_epoch),
        "best_val_rmse_abs": float(best_val),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_r2": float(test_r2),
        "device": str(device),
    }
    json_path = os.path.join(args.outdir, "model4_lstm_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    pred_csv = os.path.join(args.outdir, "model4_lstm_test_predictions.csv")
    pd.DataFrame({
        "time": t_out,
        "y_true": y_true,
        "y_pred": y_pred,
        "err": (y_pred - y_true),
    }).to_csv(pred_csv, index=False, encoding="utf-8-sig")

    plot_forecast_curve(
        y_true=y_true,
        y_pred=y_pred,
        t=t_out,
        out_path=os.path.join(args.outdir, "model4_lstm_forecast_curve.png"),
        title=f"LSTM forecast | {args.target} | mode={args.target_mode} | h={args.horizon}"
    )
    plot_loss_curve(
        train_losses=train_losses,
        val_losses=val_losses,
        out_path=os.path.join(args.outdir, "model4_lstm_loss_curve.png"),
        title=f"LSTM training | val_rmse_abs"
    )

    print("=== LSTM Training Done ===")
    print(f"saved: {pt_path}")
    print(f"saved: {json_path}")
    print(f"saved: {pred_csv}")
    print(f"test_mae: {test_mae}")
    print(f"test_rmse: {test_rmse}")
    print(f"test_r2: {test_r2}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="raw/数据处理结果.csv")
    ap.add_argument("--target", type=str, default="出站压力-连木沁压气站")
    ap.add_argument("--target_mode", type=str, choices=["abs", "delta"], default="delta")
    ap.add_argument("--lookback", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=1)

    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--fc", type=int, default=64)

    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)

    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--drop_compressor_states", action="store_true")
    ap.add_argument("--include_target_lag", action="store_true")

    ap.add_argument("--outdir", type=str, default="runs/model4_lstm/baseline")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_main(args)
