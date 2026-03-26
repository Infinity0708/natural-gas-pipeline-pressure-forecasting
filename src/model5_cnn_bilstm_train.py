import os
import json
import math
import argparse
from dataclasses import asdict, dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def read_csv_auto(path: str) -> pd.DataFrame:
    # try utf-8, then gbk
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.read_csv(path, encoding="gbk")


def is_compressor_state_col(col: str) -> bool:
    # heuristic: compressor on/off columns
    keys = ["压缩机启停", "启停", "开停", "机组启停", "compressor", "onoff", "on_off"]
    c = str(col)
    return any(k in c for k in keys)


def build_feature_cols(df: pd.DataFrame,
                       time_col: str,
                       target_col: str,
                       drop_compressor_states: bool,
                       include_target_lag: bool) -> List[str]:
    numeric_cols = []
    for c in df.columns:
        if c in [time_col, target_col]:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

    if drop_compressor_states:
        numeric_cols = [c for c in numeric_cols if not is_compressor_state_col(c)]

    # target lag as an extra feature (P_t)
    if include_target_lag:
        lag_name = f"{target_col}__lag1"
        if lag_name not in df.columns:
            df[lag_name] = df[target_col].shift(1)
        if lag_name not in numeric_cols:
            numeric_cols.append(lag_name)

    return numeric_cols


def make_supervised(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    lookback: int,
    horizon: int,
    target_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X: [N, lookback, F]
      y: [N]
      t_out: [N] timestamps for y
    """
    feat = df[feature_cols].to_numpy(dtype=np.float32)
    y_abs = df[target_col].to_numpy(dtype=np.float32)
    t = df["time"].to_numpy()

    N = len(df)
    X_list, y_list, t_list = [], [], []
    # need index t0 such that:
    # X uses [i-lookback+1 ... i], y uses i+horizon
    for i in range(lookback - 1, N - horizon):
        x_seq = feat[i - lookback + 1:i + 1, :]
        y_future = y_abs[i + horizon]
        if target_mode == "abs":
            y = y_future
        elif target_mode == "delta":
            y = y_future - y_abs[i]  # delta relative to current time i
        else:
            raise ValueError("target_mode must be abs or delta")

        X_list.append(x_seq)
        y_list.append(y)
        t_list.append(t[i + horizon])

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.float32)
    t_out = np.asarray(t_list)
    return X, y, t_out


def chronological_split(n: int, train_ratio: float, val_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_train = max(n_train, 1)
    n_val = max(n_val, 1)
    n_test = n - n_train - n_val
    if n_test <= 0:
        # keep at least 1 sample in test
        n_test = 1
        if n_val > 1:
            n_val -= 1
        else:
            n_train -= 1

    idx = np.arange(n)
    tr = idx[:n_train]
    va = idx[n_train:n_train + n_val]
    te = idx[n_train + n_val:]
    return tr, va, te


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, t_out: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        # keep time as string to avoid dataloader collate error
        self.t = [str(v) for v in t_out]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.t[i]


# ----------------------------
# Model: CNN + BiLSTM
# ----------------------------
class CNNBiLSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        channels: int,
        kernel_size: int,
        lstm_hidden: int,
        lstm_layers: int,
        dropout: float,
        fc: int,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * lstm_hidden, fc),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc, 1),
        )

    def forward(self, x):
        # x: [B, T, F] -> conv expects [B, F, T]
        x = x.transpose(1, 2)
        z = self.conv(x)            # [B, C, T]
        z = z.transpose(1, 2)       # [B, T, C]
        out, _ = self.lstm(z)       # [B, T, 2H]
        last = out[:, -1, :]        # last time step
        y = self.head(last).squeeze(-1)
        return y


@dataclass
class RunConfig:
    csv: str
    target: str
    target_mode: str
    lookback: int
    horizon: int
    train_ratio: float
    val_ratio: float
    seed: int
    drop_compressor_states: bool
    include_target_lag: bool
    # model
    channels: int
    kernel_size: int
    lstm_hidden: int
    lstm_layers: int
    dropout: float
    fc: int
    # optim
    epochs: int
    lr: float
    batch_size: int
    patience: int
    weight_decay: float


def train_one_epoch(model, dl, optim, device):
    model.train()
    losses = []
    for Xb, yb, _tb in dl:
        Xb = Xb.to(device)
        yb = yb.to(device)
        pred = model(Xb)
        loss = torch.mean((pred - yb) ** 2)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def eval_rmse_scaled(model, dl, device):
    model.eval()
    ys, ps = [], []
    for Xb, yb, _tb in dl:
        Xb = Xb.to(device)
        pred = model(Xb).detach().cpu().numpy()
        ys.append(yb.numpy())
        ps.append(pred)
    y = np.concatenate(ys) if ys else np.array([])
    p = np.concatenate(ps) if ps else np.array([])
    if len(y) == 0:
        return float("nan")
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    return rmse


@torch.no_grad()
def predict_scaled(model, dl, device) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    ys, ps, ts = [], [], []
    for Xb, yb, tb in dl:
        Xb = Xb.to(device)
        pred = model(Xb).detach().cpu().numpy()
        ys.append(yb.numpy())
        ps.append(pred)
        ts.extend(list(tb))
    y = np.concatenate(ys) if ys else np.array([])
    p = np.concatenate(ps) if ps else np.array([])
    return y, p, ts


def inverse_to_abs(
    y_scaled: np.ndarray,
    p_scaled: np.ndarray,
    y_scaler: StandardScaler,
    target_mode: str,
    y_prev_abs: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert scaled y/p to absolute MPa series for reporting.
    - abs: inverse_transform directly
    - delta: inverse_transform delta, then add previous absolute target (P_t)
    """
    y_un = y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(-1)
    p_un = y_scaler.inverse_transform(p_scaled.reshape(-1, 1)).reshape(-1)

    if target_mode == "abs":
        return y_un, p_un

    if y_prev_abs is None:
        raise ValueError("y_prev_abs required for delta mode.")

    y_abs = y_prev_abs + y_un
    p_abs = y_prev_abs + p_un
    return y_abs, p_abs


def plot_forecast(y_true, y_pred, title, out_path):
    plt.figure()
    plt.plot(y_true, label="y_true")
    plt.plot(y_pred, label="y_pred")
    plt.title(title)
    plt.xlabel("test index")
    plt.ylabel("target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_loss(train_losses, val_rmse_scaled, title, out_path):
    plt.figure()
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_rmse_scaled, label="val_rmse_scaled")
    plt.title(title)
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def prepare_all(args) -> Tuple[RunConfig, Dict]:
    df = read_csv_auto(args.csv)
    df.columns = [str(c).strip() for c in df.columns]

    time_col = "时间" if "时间" in df.columns else df.columns[0]
    df["time"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    if args.target not in df.columns:
        raise ValueError(f"Target not found: {args.target}")

    # numeric fill (pandas>=2.0 no fillna(method=...))
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(0.0)

    feature_cols = build_feature_cols(
        df=df,
        time_col="time",
        target_col=args.target,
        drop_compressor_states=args.drop_compressor_states,
        include_target_lag=args.include_target_lag,
    )

    X, y, t_out = make_supervised(
        df=df,
        feature_cols=feature_cols,
        target_col=args.target,
        lookback=args.lookback,
        horizon=args.horizon,
        target_mode=args.target_mode,
    )

    n = X.shape[0]
    tr_idx, va_idx, te_idx = chronological_split(n, args.train_ratio, args.val_ratio)

    # scalers fit on TRAIN only
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    Xtr_2d = X[tr_idx].reshape(len(tr_idx), -1)
    Xva_2d = X[va_idx].reshape(len(va_idx), -1)
    Xte_2d = X[te_idx].reshape(len(te_idx), -1)

    Xtr_s = x_scaler.fit_transform(Xtr_2d).reshape(len(tr_idx), args.lookback, -1)
    Xva_s = x_scaler.transform(Xva_2d).reshape(len(va_idx), args.lookback, -1)
    Xte_s = x_scaler.transform(Xte_2d).reshape(len(te_idx), args.lookback, -1)

    ytr_s = y_scaler.fit_transform(y[tr_idx].reshape(-1, 1)).reshape(-1)
    yva_s = y_scaler.transform(y[va_idx].reshape(-1, 1)).reshape(-1)
    yte_s = y_scaler.transform(y[te_idx].reshape(-1, 1)).reshape(-1)

    # for delta mode: need P_t (previous abs target) for each sample's base time i
    # sample t_out corresponds to i+h; base is i.
    y_prev_abs = None
    if args.target_mode == "delta":
        # reconstruct base index i for each sample:
        # sample k corresponds to i = (lookback-1 + k)
        # thus previous abs is df[target][i]
        base_indices = np.arange(args.lookback - 1, len(df) - args.horizon)
        prev_abs_all = df[args.target].to_numpy(dtype=np.float32)[base_indices]
        y_prev_abs = prev_abs_all[te_idx]  # only for test reporting

    built = {
        "df": df,
        "feature_cols": feature_cols,
        "Xtr": Xtr_s,
        "Xva": Xva_s,
        "Xte": Xte_s,
        "ytr": ytr_s,
        "yva": yva_s,
        "yte": yte_s,
        "t_tr": t_out[tr_idx],
        "t_va": t_out[va_idx],
        "t_te": t_out[te_idx],
        "y_prev_abs_test": y_prev_abs,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "n_samples": {"train": int(len(tr_idx)), "val": int(len(va_idx)), "test": int(len(te_idx))},
        "n_features": int(X.shape[-1]),
    }

    cfg = RunConfig(
        csv=args.csv,
        target=args.target,
        target_mode=args.target_mode,
        lookback=args.lookback,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        drop_compressor_states=args.drop_compressor_states,
        include_target_lag=args.include_target_lag,
        channels=args.channels,
        kernel_size=args.kernel_size,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        fc=args.fc,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        weight_decay=args.weight_decay,
    )
    return cfg, built


def train_main(args):
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)
    device = pick_device()

    cfg, built = prepare_all(args)

    Xtr, ytr, ttr = built["Xtr"], built["ytr"], built["t_tr"]
    Xva, yva, tva = built["Xva"], built["yva"], built["t_va"]
    Xte, yte, tte = built["Xte"], built["yte"], built["t_te"]
    y_prev_abs_test = built["y_prev_abs_test"]
    x_scaler, y_scaler = built["x_scaler"], built["y_scaler"]
    feature_cols = built["feature_cols"]
    n_features = built["n_features"]
    n_samples = built["n_samples"]

    dl_tr = DataLoader(SeqDataset(Xtr, ytr, ttr), batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    dl_va = DataLoader(SeqDataset(Xva, yva, tva), batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    dl_te = DataLoader(SeqDataset(Xte, yte, tte), batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = CNNBiLSTM(
        n_features=n_features,
        channels=cfg.channels,
        kernel_size=cfg.kernel_size,
        lstm_hidden=cfg.lstm_hidden,
        lstm_layers=cfg.lstm_layers,
        dropout=cfg.dropout,
        fc=cfg.fc,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    print("=== CNN+BiLSTM Training (LSTM-style logs) ===")
    print(f"device: {device}")
    print(f"target: {cfg.target}")
    print(f"target_mode: {cfg.target_mode}")
    print(f"lookback: {cfg.lookback} horizon: {cfg.horizon}")
    print(f"features: {n_features}")
    print(f"train/val/test samples: {n_samples['train']} {n_samples['val']} {n_samples['test']}")
    print(f"outdir: {args.outdir}")
    print(f"split ratios: {cfg.train_ratio} {cfg.val_ratio} {1.0 - cfg.train_ratio - cfg.val_ratio}")
    print(f"seed: {cfg.seed}")
    print(f"drop_compressor_states: {cfg.drop_compressor_states} | include_target_lag: {cfg.include_target_lag}")

    best_val = float("inf")
    best_epoch = -1
    bad = 0
    train_losses, val_curve = [], []

    best_path = os.path.join(args.outdir, "model5_cnn_bilstm_best.pt")

    for ep in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(model, dl_tr, optim, device)
        val_rmse_s = eval_rmse_scaled(model, dl_va, device)

        train_losses.append(tr_loss)
        val_curve.append(val_rmse_s)

        if val_rmse_s < best_val - 1e-9:
            best_val = val_rmse_s
            best_epoch = ep
            bad = 0
            torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)}, best_path)
        else:
            bad += 1

        print(f"Epoch {ep:03d}/{cfg.epochs} | train_loss={tr_loss:.6f} | val_rmse_scaled={val_rmse_s:.6f} | bad={bad}/{cfg.patience}")

        if bad >= cfg.patience:
            print("Early stopping.")
            break

    # load best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    # test predict (scaled)
    y_s, p_s, t_list = predict_scaled(model, dl_te, device)

    # back to abs MPa for metrics + plots
    y_abs, p_abs = inverse_to_abs(y_s, p_s, y_scaler, cfg.target_mode, y_prev_abs_test)

    test_mae = float(mean_absolute_error(y_abs, p_abs))
    test_rmse = float(math.sqrt(mean_squared_error(y_abs, p_abs)))
    test_r2 = float(r2_score(y_abs, p_abs))

    # artifacts
    report = {
        "model": "CNN+BiLSTM",
        "target": cfg.target,
        "target_mode": cfg.target_mode,
        "lookback": cfg.lookback,
        "horizon": cfg.horizon,
        "n_features": n_features,
        "feature_cols": feature_cols,
        "n_samples": n_samples,
        "split": {"train_ratio": cfg.train_ratio, "val_ratio": cfg.val_ratio, "test_ratio": 1.0 - cfg.train_ratio - cfg.val_ratio},
        "seed": cfg.seed,
        "drop_compressor_states": cfg.drop_compressor_states,
        "include_target_lag": cfg.include_target_lag,
        "best_epoch": best_epoch,
        "best_val_rmse_scaled": float(best_val),
        "test_mae_abs": test_mae,
        "test_rmse_abs": test_rmse,
        "test_r2": test_r2,
        "hyperparams": asdict(cfg),
    }

    report_path = os.path.join(args.outdir, "model5_cnn_bilstm_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    pred_path = os.path.join(args.outdir, "model5_cnn_bilstm_test_predictions.csv")
    pd.DataFrame({"time": t_list, "y_true": y_abs, "y_pred": p_abs}).to_csv(pred_path, index=False, encoding="utf-8")

    forecast_png = os.path.join(args.outdir, "model5_cnn_bilstm_forecast_curve.png")
    plot_forecast(y_abs, p_abs, f"CNN+BiLSTM forecast | {cfg.target} | mode={cfg.target_mode} | h={cfg.horizon}", forecast_png)

    loss_png = os.path.join(args.outdir, "model5_cnn_bilstm_loss_curve.png")
    plot_loss(train_losses, val_curve, "CNN+BiLSTM training | val_rmse_scaled", loss_png)

    print("=== CNN+BiLSTM Training Done ===")
    print(f"saved: {best_path}")
    print(f"saved: {report_path}")
    print(f"saved: {forecast_png}")
    print(f"saved: {loss_png}")
    print(f"saved: {pred_path}")
    print(f"test_mae: {test_mae}")
    print(f"test_rmse: {test_rmse}")
    print(f"test_r2: {test_r2}")


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="raw/processed_data.csv")
    ap.add_argument("--target", default="出站压力-连木沁压气站")
    ap.add_argument("--target_mode", default="delta", choices=["abs", "delta"])
    ap.add_argument("--lookback", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=1)

    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--kernel_size", type=int, default=7)
    ap.add_argument("--lstm_hidden", type=int, default=64)
    ap.add_argument("--lstm_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--fc", type=int, default=64)

    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--drop_compressor_states", action="store_true")
    ap.add_argument("--include_target_lag", action="store_true")

    ap.add_argument("--outdir", default="runs/model5_cnn_bilstm/baseline")
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train_main(args)