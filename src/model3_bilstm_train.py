import os
import json
import math
import argparse
import random
from dataclasses import asdict, dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y_mean = float(np.mean(y))
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    return float(1.0 - ss_res / (ss_tot + 1e-12))


# --------------------------
# Data
# --------------------------
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y_scaled: np.ndarray, base_y: np.ndarray, t_out: np.ndarray):
        """
        X: (N, L, F)
        y_scaled: (N, )
        base_y: (N, )  # P_t used to reconstruct abs when target_mode=delta
        t_out: (N, )   # timestamp string or int
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_scaled, dtype=torch.float32).view(-1, 1)
        self.base_y = base_y.astype(np.float32)
        self.t_out = t_out

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.base_y[idx], str(self.t_out[idx])


def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    # try to locate time column
    time_col = None
    for cand in ["时间", "time", "timestamp", "Time", "Datetime", "date"]:
        if cand in df.columns:
            time_col = cand
            break

    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.sort_values(time_col).reset_index(drop=True)
        df["__time__"] = df[time_col].astype("datetime64[ns]")
    else:
        df["__time__"] = np.arange(len(df))

    return df


def pick_feature_columns(
    df: pd.DataFrame,
    target: str,
    drop_compressor_states: bool,
    include_target_lag: bool
) -> List[str]:
    # numeric columns only
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # remove obvious non-features
    ban = {target}
    for c in ["__time__"]:
        if c in num_cols:
            num_cols.remove(c)

    feats = [c for c in num_cols if c not in ban]

    if drop_compressor_states:
        feats = [c for c in feats if not str(c).startswith("压缩机启停")]

    # optional: include target itself as an input feature (lagged in the window)
    if include_target_lag and (target in df.columns) and (pd.api.types.is_numeric_dtype(df[target])):
        if target not in feats:
            feats = feats + [target]

    return feats


def make_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    lookback: int,
    horizon: int,
    target_mode: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X: (N, L, F)
      y: (N,)   # y in original units for abs OR delta
      base_y: (N,)  # P_t for reconstructing abs when delta, else P_t for reference
      t_out: (N,)   # timestamp at t+h
    """
    feats = df[feature_cols].astype(np.float32).to_numpy()
    y_raw = df[target_col].astype(np.float32).to_numpy()
    times = df["__time__"].to_numpy()

    N = len(df)
    X_list, y_list, base_list, tout_list = [], [], [], []

    # we build sample ending at time t, predicting t+horizon
    # window: [t-lookback+1, ..., t]
    for t in range(lookback - 1, N - horizon):
        x_win = feats[t - lookback + 1: t + 1]  # (L,F)
        y_future = y_raw[t + horizon]
        y_now = y_raw[t]

        if target_mode == "abs":
            y = y_future
        elif target_mode == "delta":
            y = y_future - y_now
        else:
            raise ValueError("target_mode must be abs or delta")

        X_list.append(x_win)
        y_list.append(y)
        base_list.append(y_now)  # used for reconstruction in delta mode
        tout_list.append(times[t + horizon])

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32)
    base_y = np.array(base_list, dtype=np.float32)
    t_out = np.array(tout_list)

    return X, y, base_y, t_out


def split_by_ratio(X, y, base_y, t_out, train_ratio: float, val_ratio: float):
    n = X.shape[0]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if n_test <= 0:
        raise ValueError("Invalid split ratios: test set becomes empty")

    Xtr, ytr, btr, ttr = X[:n_train], y[:n_train], base_y[:n_train], t_out[:n_train]
    Xva, yva, bva, tva = X[n_train:n_train + n_val], y[n_train:n_train + n_val], base_y[n_train:n_train + n_val], t_out[n_train:n_train + n_val]
    Xte, yte, bte, tte = X[n_train + n_val:], y[n_train + n_val:], base_y[n_train + n_val:], t_out[n_train + n_val:]

    return (Xtr, ytr, btr, ttr), (Xva, yva, bva, tva), (Xte, yte, bte, tte)


# --------------------------
# Model
# --------------------------
class BiLSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float, fc: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, fc),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc, 1)
        )

    def forward(self, x):
        # x: (B,L,F)
        out, _ = self.lstm(x)     # (B,L,2H)
        last = out[:, -1, :]      # (B,2H)
        y = self.head(last)       # (B,1)
        return y


# --------------------------
# Report
# --------------------------
@dataclass
class Report:
    model: str
    device: str
    seed: int
    target: str
    target_mode: str
    lookback: int
    horizon: int
    split: Dict[str, float]
    n_features: int
    n_samples: Dict[str, int]
    best_epoch: int
    best_val_rmse_scaled: float
    hparams: Dict[str, object]
    metrics: Dict[str, Dict[str, float]]


# --------------------------
# Train / Eval
# --------------------------
@torch.no_grad()
def eval_epoch(model, loader, device, y_scaler: StandardScaler, target_mode: str):
    model.eval()
    ys_true_abs, ys_pred_abs = [], []
    ys_true_scaled, ys_pred_scaled = [], []

    for Xb, yb_scaled, base_y, _ in loader:
        Xb = Xb.to(device)
        yb_scaled = yb_scaled.to(device)

        pred_scaled = model(Xb)

        # scaled arrays for logging val_rmse_scaled
        ys_true_scaled.append(yb_scaled.cpu().numpy().reshape(-1))
        ys_pred_scaled.append(pred_scaled.cpu().numpy().reshape(-1))

        # inverse transform to original units (abs or delta)
        y_true = y_scaler.inverse_transform(yb_scaled.cpu().numpy()).reshape(-1)
        y_pred = y_scaler.inverse_transform(pred_scaled.cpu().numpy()).reshape(-1)

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

    y_true_scaled = np.concatenate(ys_true_scaled, axis=0)
    y_pred_scaled = np.concatenate(ys_pred_scaled, axis=0)

    out = {
        "rmse": rmse(y_true_abs, y_pred_abs),
        "mae": mae(y_true_abs, y_pred_abs),
        "r2": r2_score(y_true_abs, y_pred_abs),
        "rmse_scaled": rmse(y_true_scaled, y_pred_scaled),
    }
    return out


def train_main(args):
    set_seed(args.seed)
    device = get_device()

    df = load_dataframe(args.csv)
    if args.target not in df.columns:
        raise ValueError(f"target column not found: {args.target}")

    feature_cols = pick_feature_columns(
        df, args.target,
        drop_compressor_states=args.drop_compressor_states,
        include_target_lag=args.include_target_lag
    )

    X, y, base_y, t_out = make_sequences(
        df, feature_cols, args.target,
        lookback=args.lookback,
        horizon=args.horizon,
        target_mode=args.target_mode
    )

    (Xtr, ytr, btr, ttr), (Xva, yva, bva, tva), (Xte, yte, bte, tte) = split_by_ratio(
        X, y, base_y, t_out,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )

    # Scale features using train only
    F = Xtr.shape[-1]
    x_scaler = StandardScaler()
    Xtr2 = x_scaler.fit_transform(Xtr.reshape(-1, F)).reshape(Xtr.shape)
    Xva2 = x_scaler.transform(Xva.reshape(-1, F)).reshape(Xva.shape)
    Xte2 = x_scaler.transform(Xte.reshape(-1, F)).reshape(Xte.shape)

    # Scale target (abs or delta) using train only
    y_scaler = StandardScaler()
    ytr_scaled = y_scaler.fit_transform(ytr.reshape(-1, 1)).reshape(-1)
    yva_scaled = y_scaler.transform(yva.reshape(-1, 1)).reshape(-1)
    yte_scaled = y_scaler.transform(yte.reshape(-1, 1)).reshape(-1)

    ds_tr = SeqDataset(Xtr2, ytr_scaled, btr, ttr)
    ds_va = SeqDataset(Xva2, yva_scaled, bva, tva)
    ds_te = SeqDataset(Xte2, yte_scaled, bte, tte)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False)

    model = BiLSTMRegressor(
        n_features=Xtr2.shape[-1],
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        fc=args.fc
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    ensure_dir(args.outdir)

    print("=== BiLSTM Training (LSTM-style logs) ===")
    print(f"device: {str(device)}")
    print(f"target: {args.target}")
    print(f"target_mode: {args.target_mode}")
    print(f"lookback: {args.lookback} horizon: {args.horizon}")
    print(f"features: {Xtr2.shape[-1]}")
    print(f"train/val/test samples: {len(ds_tr)} {len(ds_va)} {len(ds_te)}")
    print(f"outdir: {args.outdir}")
    print(f"split ratios: {args.train_ratio} {args.val_ratio} {1.0-args.train_ratio-args.val_ratio}")
    print(f"seed: {args.seed}")
    print(f"drop_compressor_states: {args.drop_compressor_states} | include_target_lag: {args.include_target_lag}")

    best_epoch = -1
    best_val_rmse_scaled = float("inf")
    best_state = None
    bad = 0
    history = {"train_loss": [], "val_rmse_scaled": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for Xb, yb, _, _ in dl_tr:
            Xb = Xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(Xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses)) if losses else float("nan")

        val_metrics = eval_epoch(model, dl_va, device, y_scaler, args.target_mode)
        val_rmse_scaled = val_metrics["rmse_scaled"]

        history["train_loss"].append(train_loss)
        history["val_rmse_scaled"].append(val_rmse_scaled)

        improved = val_rmse_scaled < best_val_rmse_scaled - 1e-8
        if improved:
            best_val_rmse_scaled = val_rmse_scaled
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.6f} | val_rmse_scaled={val_rmse_scaled:.6f} | bad={bad}/{args.patience}")
        if bad >= args.patience:
            print("Early stopping.")
            break

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics
    tr_m = eval_epoch(model, dl_tr, device, y_scaler, args.target_mode)
    va_m = eval_epoch(model, dl_va, device, y_scaler, args.target_mode)
    te_m = eval_epoch(model, dl_te, device, y_scaler, args.target_mode)

    # Save artifacts
    best_pt = os.path.join(args.outdir, "model3_bilstm_best.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "x_scaler_mean": x_scaler.mean_.tolist(),
        "x_scaler_scale": x_scaler.scale_.tolist(),
        "y_scaler_mean": y_scaler.mean_.tolist(),
        "y_scaler_scale": y_scaler.scale_.tolist(),
        "feature_cols": feature_cols,
        "target": args.target,
        "target_mode": args.target_mode,
        "lookback": args.lookback,
        "horizon": args.horizon,
    }, best_pt)

    report = Report(
        model="BiLSTM",
        device=str(device),
        seed=args.seed,
        target=args.target,
        target_mode=args.target_mode,
        lookback=args.lookback,
        horizon=args.horizon,
        split={"train_ratio": args.train_ratio, "val_ratio": args.val_ratio, "test_ratio": 1.0 - args.train_ratio - args.val_ratio},
        n_features=Xtr2.shape[-1],
        n_samples={"train": len(ds_tr), "val": len(ds_va), "test": len(ds_te)},
        best_epoch=best_epoch,
        best_val_rmse_scaled=float(best_val_rmse_scaled),
        hparams={
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "hidden": args.hidden,
            "layers": args.layers,
            "dropout": args.dropout,
            "fc": args.fc,
            "drop_compressor_states": args.drop_compressor_states,
            "include_target_lag": args.include_target_lag,
        },
        metrics={
            "train": {"rmse": tr_m["rmse"], "mae": tr_m["mae"], "r2": tr_m["r2"]},
            "val":   {"rmse": va_m["rmse"], "mae": va_m["mae"], "r2": va_m["r2"]},
            "test":  {"rmse": te_m["rmse"], "mae": te_m["mae"], "r2": te_m["r2"]},
        }
    )

    report_path = os.path.join(args.outdir, "model3_bilstm_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2)

    # Save test predictions CSV (absolute values)
    # Re-run on test set to export per-row
    model.eval()
    rows = []
    with torch.no_grad():
        for Xb, yb_scaled, base_y, t_out_b in dl_te:
            Xb = Xb.to(device)
            pred_scaled = model(Xb).cpu().numpy()
            y_true = y_scaler.inverse_transform(yb_scaled.numpy()).reshape(-1)
            y_pred = y_scaler.inverse_transform(pred_scaled).reshape(-1)
            base_y = base_y.numpy().reshape(-1)

            if args.target_mode == "delta":
                y_true_abs = base_y + y_true
                y_pred_abs = base_y + y_pred
            else:
                y_true_abs = y_true
                y_pred_abs = y_pred

            for ti, yt, yp in zip(t_out_b, y_true_abs, y_pred_abs):
                rows.append({"time": str(ti), "y_true": float(yt), "y_pred": float(yp)})

    pred_csv = os.path.join(args.outdir, "model3_bilstm_test_predictions.csv")
    pd.DataFrame(rows).to_csv(pred_csv, index=False, encoding="utf-8-sig")

    # Plot forecast curve (first 300 points)
    n_plot = min(300, len(rows))
    yt = [r["y_true"] for r in rows[:n_plot]]
    yp = [r["y_pred"] for r in rows[:n_plot]]

    plt.figure()
    plt.plot(yt, label="True")
    plt.plot(yp, label="Pred")
    plt.title("BiLSTM Forecast (test set)")
    plt.xlabel("Index")
    plt.ylabel("Target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "model3_bilstm_forecast_curve.png"), dpi=200)
    plt.close()

    # Plot loss curve
    plt.figure()
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_rmse_scaled"], label="Val RMSE (scaled)")
    plt.title("BiLSTM Training Curves")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "model3_bilstm_loss_curve.png"), dpi=200)
    plt.close()

    print("=== BiLSTM Training Done ===")
    print(f"saved: {best_pt}")
    print(f"saved: {report_path}")
    print(f"saved: {os.path.join(args.outdir, 'model3_bilstm_forecast_curve.png')}")
    print(f"saved: {os.path.join(args.outdir, 'model3_bilstm_loss_curve.png')}")
    print(f"saved: {pred_csv}")
    print(f"test_mae: {te_m['mae']}")
    print(f"test_rmse: {te_m['rmse']}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--target", type=str, required=True)
    ap.add_argument("--target_mode", type=str, default="delta", choices=["abs", "delta"])
    ap.add_argument("--lookback", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=1)

    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--patience", type=int, default=20)

    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--fc", type=int, default=64)

    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--drop_compressor_states", action="store_true")
    ap.add_argument("--include_target_lag", action="store_true")

    ap.add_argument("--outdir", type=str, default="runs/model3_bilstm/baseline")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_main(args)
