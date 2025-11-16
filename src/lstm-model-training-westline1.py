import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------- 配置（只改这一段即可适配“西一线”） --------------------
# 把你的西一线 CSV 放到 raw/ 目录，例如 raw/x1.csv
ROOT = Path(__file__).resolve().parent.parent  # 项目根目录 …/xxc262
CONFIG = {
    "csv_path": str(ROOT / "raw" / "x1.csv"),  # ← 改成你的西一线CSV文件名
    "target_col": "X1_in",                     # ← 改成“西一线 本站进站压力”的真实列名
    "base_feature_cols": [                     # ← 改成你CSV里实际存在的列
        "X0_out",  # 上一站出站压力
        "X1_in",   # 本站进站压力
        "X1_out",  # 本站出站压力
        "Temp"     # 站内温度
    ],
    "sequence_length": 30,         # 窗口长度 L（用过去 30 步预测下一步）
    "lags": [1, 2, 3, 6, 12],      # 滞后阶数
    "add_time_features": True,     # 是否加 hour / dow
    "rolling_k": 6,                # 滚动均值窗口(0=不加)
    "train_ratio": 0.7, "val_ratio": 0.15,
    "batch_size_train": 128, "batch_size_val": 128, "batch_size_test": 256,
    "epochs": 50, "lr": 0.002, "patience": 6
}
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------------------------------------------------------

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).unsqueeze(-1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True)
        self.norm1 = nn.LayerNorm(64); self.drop1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.norm2 = nn.LayerNorm(32); self.drop2 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(32, 16); self.act = nn.ReLU(); self.drop3 = nn.Dropout(0.1)
        self.out = nn.Linear(16, 1)
    def forward(self, x):
        x, _ = self.lstm1(x); x = self.norm1(x); x = self.drop1(x)
        x, _ = self.lstm2(x); x = self.norm2(x); x = self.drop2(x)
        x = x[:, -1, :]
        x = self.fc1(x); x = self.act(x); x = self.drop3(x)
        return self.out(x)

def load_and_analyze_data(csv_path: str) -> pd.DataFrame:
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # 可选的时间列排序
    for cand in ["timestamp", "time", "datetime", "ts"]:
        if cand in df.columns:
            df[cand] = pd.to_datetime(df[cand])
            df = df.sort_values(cand).reset_index(drop=True)
            break
    df = df.drop_duplicates()
    # 简单插值：有时间列用 time 插值，否则线性
    df = df.interpolate(method="time") if any(c in df.columns for c in ["timestamp","time","datetime","ts"]) else df.interpolate()
    print(f"[Load] shape={df.shape} | columns={list(df.columns)}")
    return df

def build_features(df: pd.DataFrame, target_col: str, base_feature_cols: List[str],
                   lags: List[int], add_time_features: bool, rolling_k: int) -> Tuple[pd.DataFrame, List[str]]:
    df_feat = df.copy()

    # 时间特征
    if add_time_features:
        time_col = next((c for c in ["timestamp","time","datetime","ts"] if c in df_feat.columns), None)
        if time_col:
            df_feat["hour"] = df_feat[time_col].dt.hour
            df_feat["dow"]  = df_feat[time_col].dt.dayofweek

    # 滞后/差分/滚动
    for col in base_feature_cols:
        if col not in df_feat.columns:
            print(f"[Warn] missing base feature: {col}")
            continue
        for k in lags:
            df_feat[f"{col}_lag{k}"] = df_feat[col].shift(k)
        df_feat[f"{col}_diff1"] = df_feat[col].diff(1)
        if rolling_k and rolling_k > 0:
            df_feat[f"{col}_roll{rolling_k}"] = df_feat[col].rolling(rolling_k).mean()

    df_feat = df_feat.dropna().reset_index(drop=True)

    # 组装特征列：保留原始基础列 + 衍生列
    feature_cols = [c for c in base_feature_cols if c in df_feat.columns]
    feature_cols += [c for c in df_feat.columns if any(tag in c for tag in ["_lag","_diff","_roll","hour","dow"])]
    # 去重保序
    seen=set(); feature_cols=[x for x in feature_cols if not (x in seen or seen.add(x))]

    if target_col not in df_feat.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV.")

    print(f"[Features] target={target_col} | n_features(before window)={len(feature_cols)}")
    return df_feat, feature_cols

def time_split(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    N = len(df); n_train = int(N*train_ratio); n_val = int(N*val_ratio)
    return df.iloc[:n_train], df.iloc[n_train:n_train+n_val], df.iloc[n_train+n_val:]

def scale_and_window(train_df, val_df, test_df, feature_cols, target_col, L,
                     bs_train, bs_val, bs_test):
    feat_scaler = MinMaxScaler().fit(train_df[feature_cols].to_numpy())
    tgt_scaler  = MinMaxScaler().fit(train_df[[target_col]].to_numpy())

    def to_seq(df):
        X = feat_scaler.transform(df[feature_cols].to_numpy())
        y = tgt_scaler.transform(df[[target_col]].to_numpy()).ravel()
        Xs, ys = [], []
        for i in range(L, len(df)):
            Xs.append(X[i-L:i, :])
            ys.append(y[i])
        return np.stack(Xs), np.array(ys)

    Xtr, ytr = to_seq(train_df)
    Xva, yva = to_seq(val_df)
    Xte, yte = to_seq(test_df)

    train_loader = DataLoader(SequenceDataset(Xtr, ytr), batch_size=bs_train, shuffle=True)
    val_loader   = DataLoader(SequenceDataset(Xva, yva), batch_size=bs_val, shuffle=False)
    test_loader  = DataLoader(SequenceDataset(Xte, yte), batch_size=bs_test, shuffle=False)

    n_feat = Xtr.shape[2]
    print(f"[Windowed] train/val/test = {len(Xtr)}/{len(Xva)}/{len(Xte)} | L={L} | F={n_feat}")
    return train_loader, val_loader, test_loader, (Xte, yte), n_feat, feat_scaler, tgt_scaler

def train_model(model, train_loader, val_loader, epochs, lr, patience, device):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999), eps=1e-7)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.3, patience=5, min_lr=1e-6)
    loss_fn = nn.MSELoss()
    best_val = float('inf'); waited=0
    history = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}

    for ep in range(1, epochs+1):
        model.train(); trL=0.0; trM=0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward(); opt.step()
            trL += loss.item()*len(xb); trM += (pred.detach()-yb).abs().sum().item()
        trL /= len(train_loader.dataset); trM /= len(train_loader.dataset)

        model.eval(); vaL=0.0; vaM=0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                vaL += loss.item()*len(xb); vaM += (pred-yb).abs().sum().item()
        vaL /= len(val_loader.dataset); vaM /= len(val_loader.dataset)
        sched.step(vaL)
        history["loss"].append(trL); history["val_loss"].append(vaL)
        history["mae"].append(trM);  history["val_mae"].append(vaM)

        print(f"Epoch {ep:02d} | loss {trL:.6f} | val {vaL:.6f} | mae {trM:.6f} | val_mae {vaM:.6f}")

        if vaL < best_val - 1e-8:
            best_val, waited = vaL, 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            waited += 1
            if waited >= patience:
                print("Early stopping."); break
    return history

def evaluate_model(model, test_loader, tgt_scaler, device):
    model.eval(); preds=[]
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            yb = model(xb).cpu().numpy().flatten()
            preds.append(yb)
    y_pred_scaled = np.concatenate(preds, axis=0)
    return y_pred_scaled

def metrics_in_original_units(y_true_scaled, y_pred_scaled, tgt_scaler):
    y_true = tgt_scaler.inverse_transform(y_true_scaled.reshape(-1,1)).ravel()
    y_pred = tgt_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}, y_true, y_pred

if __name__ == "__main__":
    print("Using device:", DEVICE)
    # 1) 读数据
    df = load_and_analyze_data(CONFIG["rawX1.csv"])
    # 2) 构特征
    df_feat, feature_cols = build_features(
        df,
        target_col=CONFIG["target_col"],
        base_feature_cols=CONFIG["base_feature_cols"],
        lags=CONFIG["lags"],
        add_time_features=CONFIG["add_time_features"],
        rolling_k=CONFIG["rolling_k"]
    )
    # 3) 时间切分 + 无泄露标准化 + 滑窗
    tr, va, te = time_split(df_feat, CONFIG["train_ratio"], CONFIG["val_ratio"])
    train_loader, val_loader, test_loader, (Xte, yte), n_feat, feat_scaler, tgt_scaler = \
        scale_and_window(
            tr, va, te,
            feature_cols=feature_cols,
            target_col=CONFIG["target_col"],
            L=CONFIG["sequence_length"],
            bs_train=CONFIG["batch_size_train"],
            bs_val=CONFIG["batch_size_val"],
            bs_test=CONFIG["batch_size_test"]
        )
    # 4) 建模 + 训练
    model = LSTMRegressor(n_feat).to(DEVICE)
    print(model)
    history = train_model(model, train_loader, val_loader,
                          epochs=CONFIG["epochs"], lr=CONFIG["lr"],
                          patience=CONFIG["patience"], device=DEVICE)
    # 5) 测试集推理 + 反归一化评估
    y_pred_scaled = evaluate_model(model, test_loader, tgt_scaler, DEVICE)
    metrics, y_true_org, y_pred_org = metrics_in_original_units(yte, y_pred_scaled, tgt_scaler)
    print(f"[Test] RMSE={metrics['RMSE']:.4f} MPa | MAE={metrics['MAE']:.4f} MPa | R²={metrics['R2']:.4f}")
    # 6) 保存权重与元数据
    torch.save(model.state_dict(), "generic_lstm_best.pt")
    import pickle
    meta = dict(
        config=CONFIG, feature_cols=feature_cols, target_col=CONFIG["target_col"],
        sequence_length=CONFIG["sequence_length"],
        feature_scaler=feat_scaler, target_scaler=tgt_scaler,
        history=history, input_size=n_feat
    )
    with open("generic_lstm_best_metadata.pkl", "wb") as f:
        pickle.dump(meta, f)
    print("Saved: generic_lstm_best.pt + generic_lstm_best_metadata.pkl")
