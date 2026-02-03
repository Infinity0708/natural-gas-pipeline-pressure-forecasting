import os
import json
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import optuna


def read_csv_auto(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


def train_val_test_split_time(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    df_train = df.iloc[:n_train].copy().reset_index(drop=True)
    df_val = df.iloc[n_train:n_train + n_val].copy().reset_index(drop=True)
    df_test = df.iloc[n_train + n_val:].copy().reset_index(drop=True)
    return df_train, df_val, df_test


def make_supervised_sequences(df, feature_cols, target_col, lookback, horizon):
    X, y = [], []
    feat = df[feature_cols].to_numpy(dtype=np.float32)
    tgt = df[target_col].to_numpy(dtype=np.float32)
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
    if len(X) == 0:
        return None, None
    return np.stack(X, 0), np.asarray(y, dtype=np.float32)


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(-1)

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]


class CNN1DForecaster(nn.Module):
    def __init__(self, n_features, channels, kernel_size, dropout):
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
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (B,L,F)->(B,F,L)
        return self.net(x)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="raw/数据处理结果.csv")
    ap.add_argument("--target", default="出站压力-连木沁压气站")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="runs/model2_cnn1d/hpo")
    ap.add_argument("--drop_compressor_states", type=int, default=0)
    ap.add_argument("--max_features", type=int, default=0)
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

    for c in df.columns:
        if c in ["time", time_col]:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    numeric_cols = [c for c in df.columns if c not in ["time", time_col] and c != args.target]
    if args.drop_compressor_states == 1:
        numeric_cols = [c for c in numeric_cols if not str(c).startswith("压缩机启停-")]
    if args.max_features and args.max_features > 0:
        numeric_cols = numeric_cols[:args.max_features]

    df_tr, df_va, _ = train_val_test_split_time(df, 0.7, 0.15)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    Xtr = x_scaler.fit_transform(df_tr[numeric_cols].to_numpy(dtype=np.float32))
    Xva = x_scaler.transform(df_va[numeric_cols].to_numpy(dtype=np.float32))
    ytr = y_scaler.fit_transform(df_tr[[args.target]].to_numpy(dtype=np.float32)).squeeze()
    yva = y_scaler.transform(df_va[[args.target]].to_numpy(dtype=np.float32)).squeeze()

    df_tr_s = df_tr.copy()
    df_va_s = df_va.copy()
    df_tr_s[numeric_cols] = Xtr
    df_va_s[numeric_cols] = Xva
    df_tr_s[args.target] = ytr
    df_va_s[args.target] = yva

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial: optuna.Trial):
        lookback = trial.suggest_categorical("lookback", [12, 24, 48])
        channels = trial.suggest_categorical("channels", [32, 64, 128])
        kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        epochs = trial.suggest_categorical("epochs", [20, 30, 50])
        patience = trial.suggest_categorical("patience", [5, 8])

        Xtr_seq, ytr_seq = make_supervised_sequences(df_tr_s, numeric_cols, args.target, lookback, args.horizon)
        Xva_seq, yva_seq = make_supervised_sequences(df_va_s, numeric_cols, args.target, lookback, args.horizon)
        if Xtr_seq is None or Xva_seq is None:
            return 1e9

        train_loader = DataLoader(SeqDataset(Xtr_seq, ytr_seq), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(SeqDataset(Xva_seq, yva_seq), batch_size=batch_size, shuffle=False)

        model = CNN1DForecaster(Xtr_seq.shape[-1], channels, kernel_size, dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        best = float("inf")
        bad = 0

        for ep in range(1, epochs + 1):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

            # val
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    pred = model(xb).cpu().numpy().squeeze()
                    y_pred.append(pred)
                    y_true.append(yb.numpy().squeeze())
            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            val_rmse = rmse(y_true, y_pred)

            trial.report(val_rmse, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_rmse < best - 1e-6:
                best = val_rmse
                bad = 0
            else:
                bad += 1
            if bad >= patience:
                break

        return best

    storage = f"sqlite:///{os.path.join(args.outdir, 'study.db')}"
    study = optuna.create_study(direction="minimize", study_name="cnn1d_hpo", storage=storage, load_if_exists=True)

    if args.timeout and args.timeout > 0:
        study.optimize(objective, timeout=args.timeout, n_trials=args.trials)
    else:
        study.optimize(objective, n_trials=args.trials)

    best = study.best_trial
    best_path = os.path.join(args.outdir, "best_params.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump({"value": best.value, "params": best.params}, f, indent=2)

    print("=== CNN1D HPO Done ===")
    print(f"device: {device}")
    print(f"best_value (val_rmse_scaled): {best.value}")
    print(f"best_params saved to: {best_path}")
    print(f"study db: {os.path.join(args.outdir, 'study.db')}")


if __name__ == "__main__":
    main()
