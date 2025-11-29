import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from contextlib import nullcontext

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parents[1]
FILE_PATH = str(ROOT / "raw" / "raw.csv")
NODES = ["西二线", "西三线"]
EDGES = [("西二线", "西三线")]
NODE_FEATURE_MAP = {
    "西二线": {
        "in":   "西二线本站进站压力",
        "out":  "西二线上一站出站压力",
        "temp": "西二线上一站出站温度",
    },
    "西三线": {
        "in":   "西三线本站进站压力",
        "out":  "西三线上一站出站压力",
        "temp": "西三线本站进站温度",
    },
}

TARGET_NODE = "西三线"
TARGET_FEATURE = "in"

SEQUENCE_LENGTH = 240
TEMPORAL_KERNEL  = 5
HIDDEN           = 64
DROPOUT          = 0.2

BATCH_TRAIN, BATCH_VAL, BATCH_TEST = 256, 256, 512
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 12
ADD_TIME_FEATURES = True
ADD_LAGS = True
LAG_STEPS = [1, 5, 15]

# 数据集
class STSequenceDataset(Dataset):
    """X: (N, F_node, L, N_nodes), y: (N, 1)"""
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).unsqueeze(-1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def add_cyclical_time_features(df, time_col):
    """周期时间特征：hour/minute/dow 的 sin/cos"""
    ts = pd.to_datetime(df[time_col])
    hour = ts.dt.hour.values
    minute = ts.dt.minute.values
    dow = ts.dt.dayofweek.values

    df["_sin_hour"] = np.sin(2 * np.pi * hour / 24)
    df["_cos_hour"] = np.cos(2 * np.pi * hour / 24)
    df["_sin_min"]  = np.sin(2 * np.pi * (hour * 60 + minute) / 1440)
    df["_cos_min"]  = np.cos(2 * np.pi * (hour * 60 + minute) / 1440)
    df["_sin_dow"]  = np.sin(2 * np.pi * dow / 7)
    df["_cos_dow"]  = np.cos(2 * np.pi * dow / 7)
    return df, ["_sin_hour","_cos_hour","_sin_min","_cos_min","_sin_dow","_cos_dow"]

def add_lag_columns(df, col_names, lags):
    """对指定列增加滞后特征；严格因果，后续会 dropna 或切片对齐"""
    for c in col_names:
        for k in lags:
            df[f"{c}_lag{k}"] = df[c].shift(k)
    return df

#stgnn
def normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    I = torch.eye(A.size(0), device=A.device)
    A_tilde = A + I
    deg = torch.sum(A_tilde, dim=1)
    D_inv_sqrt = torch.diag(torch.pow(deg, -0.5))
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt

class TemporalConv(nn.Module):
    """时间卷积：对 (B, C, T, N) 在 T 维做卷积（Conv2d kernel=(k,1)）"""
    def __init__(self, c_in, c_out, k=3, dropout=0.1):
        super().__init__()
        self.k = k
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=(k,1), padding=(k-1,0))
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = self.conv(x)                 # (B,C_out,T+k-1,N)
        x = x[:, :, :-(self.k-1), :]     # 因果裁切，避免偷看未来
        x = self.act(x)
        return self.drop(x)

class GraphConv(nn.Module):
    """一阶GCN扩散：在节点维做 A_hat 乘法"""
    def __init__(self, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()
    def forward(self, x, A_hat):
        x = torch.einsum('bctn,nm->bctm', x, A_hat)   # (B,C,T,N) × (N,N)
        x = self.act(x)
        return self.drop(x)

class STGCNBlock(nn.Module):
    """STGCN 基本块：TempConv → GraphConv → TempConv"""
    def __init__(self, c_in, c_out, k_t=3, dropout=0.1):
        super().__init__()
        self.t1 = TemporalConv(c_in,  c_out, k=k_t, dropout=dropout)
        self.g  = GraphConv(dropout=dropout)
        self.t2 = TemporalConv(c_out, c_out, k=k_t, dropout=dropout)
    def forward(self, x, A_hat):
        x = self.t1(x)
        x = self.g(x, A_hat)
        x = self.t2(x)
        return x

class STGNNRegressor(nn.Module):
    """
    输入: (B, F_node, L, N)
    输出: (B, 1)  —— 目标站点下一时刻的标量预测
    """
    def __init__(self, f_in, hidden=32, k_t=3, dropout=0.1):
        super().__init__()
        self.block1 = STGCNBlock(f_in,   hidden, k_t=k_t, dropout=dropout)
        self.block2 = STGCNBlock(hidden, hidden, k_t=k_t, dropout=dropout)
        self.head = nn.Sequential(
            nn.Conv2d(hidden, 16, kernel_size=(1,1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(16, 1, kernel_size=(1,1))
        )
    def forward(self, x, A_hat, target_node_idx:int):
        x = self.block1(x, A_hat)
        x = self.block2(x, A_hat)
        x = x[:, :, -1:, :]                                  # 取最后时间步
        x = x[:, :, :, target_node_idx:target_node_idx+1]    # 取目标节点
        x = self.head(x)                                     # (B,1,1,1)
        return x.view(-1, 1)

class OptimizedTorchSTGNNForecast:
    def __init__(self, sequence_length=SEQUENCE_LENGTH, amp_enabled=False):
        self.sequence_length = sequence_length
        self.amp_enabled = amp_enabled and torch.cuda.is_available()
        self.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")
        print(f"Using device: {self.device}")
        self.model = None
        self.history = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}
        self.feature_scaler = None
        self.target_scaler = None
        self.A_hat = None
        self.target_node_idx = None
        self.nodes = NODES
        self.node_feature_map = NODE_FEATURE_MAP
        self.target_node = TARGET_NODE
        self.target_feature = TARGET_FEATURE

    # ---------- 1) 读取与体检 ----------
    def load_and_analyze_data(self, path):
        df = pd.read_csv(path)
        # 时间列解析与排序
        for c in ["timestamp","time","datetime","ts"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
                df = df.sort_values(c).reset_index(drop=True)
                break
        print(f"[Load] shape= {df.shape}  | cols= {list(df.columns)[:7]} ...")
        return df

    # ---------- 邻接 ----------
    def build_adjacency(self):
        idx = {n:i for i,n in enumerate(self.nodes)}
        N = len(self.nodes)
        A = np.zeros((N,N), dtype=np.float32)
        for u,v in EDGES:
            if u in idx and v in idx:
                A[idx[u], idx[v]] = 1.0
        A_t = torch.tensor(A, dtype=torch.float32, device=self.device)
        self.A_hat = normalize_adjacency(A_t)
        self.target_node_idx = idx[self.target_node]

    # ---------- 2) 预处理 ----------
    def prepare_data(self, df):
        """
        [CHANGED/NEW] 周期时间特征 + 滞后特征 + 更长窗口
        输出：DataLoaders + 测试集（用于评估反归一化）
        """
        # 时间列
        time_col = None
        for c in ["timestamp","time","datetime","ts"]:
            if c in df.columns: time_col = c; break
        if time_col is None and ADD_TIME_FEATURES:
            raise ValueError("未找到时间列，无法添加时间特征")

        #周期时间特征（加到所有节点）
        time_feats = []
        if ADD_TIME_FEATURES and time_col is not None:
            df, time_feats = add_cyclical_time_features(df, time_col)

        #滞后特征（对所有站点的 in/out）
        if ADD_LAGS:
            base_inout = []
            for n in self.nodes:
                fmap = self.node_feature_map[n]
                for key in ["in","out"]:
                    if key in fmap and fmap[key] in df.columns:
                        base_inout.append(fmap[key])
            df = add_lag_columns(df, base_inout, LAG_STEPS)
            max_lag = max(LAG_STEPS)
            df = df.iloc[max_lag:].reset_index(drop=True)  # 丢掉前 max_lag 行
        else:
            max_lag = 0

        # 组装每个节点特征
        node_arrays = []
        per_node_cols = []
        for n in self.nodes:
            fmap = self.node_feature_map[n]
            cols = []
            for key in ["in","out","temp"]:
                if key in fmap and fmap[key] in df.columns:
                    cols.append(fmap[key])
            if ADD_LAGS:
                for key in ["in","out"]:
                    if key in fmap and fmap[key] in df.columns:
                        base = fmap[key]
                        for k in LAG_STEPS:
                            lag_col = f"{base}_lag{k}"
                            if lag_col in df.columns:
                                cols.append(lag_col)
            cols.extend(time_feats)  # 时间特征复制到每个节点
            if len(cols) == 0:
                raise ValueError(f"节点 {n} 无有效特征列，请检查映射/lag")
            node_arrays.append(df[cols].to_numpy(dtype=np.float32))
            per_node_cols.append(cols)

        # 校验每节点特征维一致
        f_dims = [x.shape[1] for x in node_arrays]
        if len(set(f_dims)) != 1:
            raise ValueError(f"每个节点的特征数需一致：{f_dims}")
        F_node = f_dims[0]
        T = node_arrays[0].shape[0]

        # (T,N,F_node)
        X_all = np.stack(node_arrays, axis=1)

        # 目标列（已和 X 对齐）
        target_col = self.node_feature_map[self.target_node][self.target_feature]
        y_all = df[target_col].to_numpy(dtype=np.float32)

        # 按时间切分
        n_train = int(T * 0.7)
        n_val   = int(T * 0.15)
        tr, va, te = slice(0,n_train), slice(n_train, n_train+n_val), slice(n_train+n_val, T)

        # 无泄露标准化（fit on train only）
        feat_scaler = MinMaxScaler().fit(X_all[tr].reshape(n_train, -1))
        tgt_scaler  = MinMaxScaler().fit(y_all[tr].reshape(-1,1))
        self.feature_scaler = feat_scaler
        self.target_scaler  = tgt_scaler

        def scale_X(seg):
            Tseg = seg.shape[0]
            X2d = seg.reshape(Tseg, -1)
            X2d = feat_scaler.transform(X2d)
            return X2d.reshape(Tseg, len(self.nodes), F_node)

        Xtr_all = scale_X(X_all[tr])
        Xva_all = scale_X(X_all[va])
        Xte_all = scale_X(X_all[te])

        ytr = tgt_scaler.transform(y_all[tr].reshape(-1,1)).ravel()
        yva = tgt_scaler.transform(y_all[va].reshape(-1,1)).ravel()
        yte = tgt_scaler.transform(y_all[te].reshape(-1,1)).ravel()

        # 滑窗 → (S, F_node, L, N)
        def make_windows(X_all_seg, y_seg, L):
            Tseg = X_all_seg.shape[0]
            Xs, ys = [], []
            for i in range(L, Tseg):
                Xw = X_all_seg[i-L:i, :, :]      # (L,N,F)
                Xw = np.transpose(Xw, (2,0,1))   # (F,L,N)
                Xs.append(Xw); ys.append(y_seg[i])
            return np.stack(Xs), np.array(ys, dtype=np.float32)

        Xtr, ytr = make_windows(Xtr_all, ytr, self.sequence_length)
        Xva, yva = make_windows(Xva_all, yva, self.sequence_length)
        Xte, yte = make_windows(Xte_all, yte, self.sequence_length)

        print(f"训练/验证/测试样本: {len(Xtr)}/{len(Xva)}/{len(Xte)}; 输入维: F={Xtr.shape[1]}, L={Xtr.shape[2]}, N={Xtr.shape[3]}")

        train_loader = DataLoader(STSequenceDataset(Xtr, ytr), batch_size=BATCH_TRAIN, shuffle=True)
        val_loader   = DataLoader(STSequenceDataset(Xva, yva), batch_size=BATCH_VAL, shuffle=False)
        test_loader  = DataLoader(STSequenceDataset(Xte, yte), batch_size=BATCH_TEST, shuffle=False)

        # 邻接
        self.build_adjacency()

        return train_loader, val_loader, test_loader, Xte, yte, Xtr.shape[1]

    # ---------- 3) 构建模型 ----------
    def build_model(self, input_channels):
        self.model = STGNNRegressor(
            f_in=input_channels,
            hidden=HIDDEN,
            k_t=TEMPORAL_KERNEL,
            dropout=DROPOUT
        ).to(self.device)
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"模型参数总数: {total_params:,}")
        return self.model

    # ---------- 4) 训练 ----------
    def train(self, train_loader, val_loader, epochs=EPOCHS):
        model = self.model
        optimizer = Adam(model.parameters(), lr=LR, betas=(0.9,0.999), eps=1e-7,
                         weight_decay=WEIGHT_DECAY)   # [CHANGED] + L2
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=6, min_lr=1e-6)
        criterion = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        best_val = float('inf'); patience = PATIENCE; waited = 0
        autocast_ctx = torch.cuda.amp.autocast if self.amp_enabled else nullcontext

        for epoch in range(1, epochs+1):
            model.train(); trL=0.0; trM=0.0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with autocast_ctx():
                    pred = model(Xb, self.A_hat, self.target_node_idx)
                    loss = criterion(pred, yb)
                scaler.scale(loss).backward()

                # [NEW] 梯度裁剪，防极端梯度
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                scaler.step(optimizer); scaler.update()
                trL += loss.item() * len(Xb)
                trM += (pred.detach()-yb).abs().sum().item()
            trL /= len(train_loader.dataset); trM /= len(train_loader.dataset)

            model.eval(); vaL=0.0; vaM=0.0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(self.device), yb.to(self.device)
                    with autocast_ctx():
                        pred = model(Xb, self.A_hat, self.target_node_idx)
                        loss = criterion(pred, yb)
                    vaL += loss.item() * len(Xb)
                    vaM += (pred - yb).abs().sum().item()
            vaL /= len(val_loader.dataset); vaM /= len(val_loader.dataset)

            scheduler.step(vaL)
            self.history['loss'].append(trL); self.history['val_loss'].append(vaL)
            self.history['mae'].append(trM);  self.history['val_mae'].append(vaM)
            print(f"Epoch {epoch:02d} | loss {trL:.6f} | val {vaL:.6f} | mae {trM:.6f} | val_mae {vaM:.6f}")

            if vaL < best_val - 1e-8:
                best_val, waited = vaL, 0
                torch.save(model.state_dict(), 'stgnn_best23.pt')
                print("保存最佳模型: stgnn_best23.pt")
            else:
                waited += 1
                if waited >= patience:
                    print("Early stopping.")
                    break

        print("训练完成")
        return self.history

    # ---------- 5) 评估 ----------
    def evaluate(self, test_loader, X_test, y_test, input_channels):
        self.model.eval(); preds=[]
        with torch.no_grad():
            for Xb, _ in test_loader:
                Xb = Xb.to(self.device)
                yb = self.model(Xb, self.A_hat, self.target_node_idx).cpu().numpy().flatten()
                preds.append(yb)
        y_pred_scaled = np.concatenate(preds, axis=0)
        y_pred_org = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
        y_true_org = self.target_scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
        rmse = np.sqrt(mean_squared_error(y_true_org, y_pred_org))
        mae  = mean_absolute_error(y_true_org, y_pred_org)
        r2   = r2_score(y_true_org, y_pred_org)
        print(f"[Test] RMSE={rmse:.4f} MPa | MAE={mae:.4f} MPa | R²={r2:.4f}")
        return {'RMSE': rmse, 'MAE': mae, 'R²': r2}, y_true_org, y_pred_org

    # ---------- 6) 可视化 ----------
    def plot_test_comparison(self, y_true, y_pred, sample_size=500):
        n = len(y_true)
        idx = np.linspace(0, n-1, min(n, sample_size), dtype=int)
        yt, yp = y_true[idx], y_pred[idx]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes[0,0].plot(idx, yt, label='真实值', linewidth=1.5)
        axes[0,0].plot(idx, yp, label='预测值', linewidth=1.5)
        axes[0,0].set_title('测试集预测结果对比-时间序列'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

        axes[0,1].scatter(yt, yp, alpha=0.6, s=20)
        mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        axes[0,1].plot([mn,mx],[mn,mx],'r--', linewidth=2, label='理想预测线'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

        res = yp - yt
        axes[1,0].scatter(yp, res, alpha=0.6, s=20)
        axes[1,0].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[1,0].set_title('残差分析'); axes[1,0].grid(True, alpha=0.3)

        axes[1,1].hist(res, bins=50, alpha=0.7, edgecolor='black')
        axes[1,1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1,1].set_title('误差分布'); axes[1,1].grid(True, alpha=0.3)

        rmse = np.sqrt(np.mean(res**2)); mae = np.mean(np.abs(res)); r2 = 1 - np.sum(res**2)/np.sum((yt - np.mean(yt))**2)
        fig.text(0.02, 0.98, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}',
                 fontsize=12, va='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.tight_layout(); plt.show()

    def plot_training(self):
        if not self.history or not self.history['loss']:
            print("没有训练历史记录"); return
        h = self.history
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        ep = range(1, len(h['loss'])+1)
        axes[0].plot(ep, h['loss'], 'b-', label='训练损失'); axes[0].plot(ep, h['val_loss'], 'r-', label='验证损失')
        axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].plot(ep, h['mae'], 'b-', label='训练MAE'); axes[1].plot(ep, h['val_mae'], 'r-', label='验证MAE')
        axes[1].set_title('MAE'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

    # ---------- 7) 保存 ----------
    def save_model_with_metadata(self, filepath='stgnn_best.pt', metadata=None):
        torch.save(self.model.state_dict(), filepath)
        import pickle
        meta = {
            'sequence_length': self.sequence_length,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'history': self.history,
            'model_arch': {'hidden': HIDDEN, 'k_t': TEMPORAL_KERNEL},
            'training_metadata': metadata or {},
            'nodes': self.nodes, 'edges': EDGES,
            'target_node': self.target_node, 'target_feature': self.target_feature
        }
        with open(filepath.replace('.pt','_metadata.pkl'), 'wb') as f:
            pickle.dump(meta, f)
        print(f"模型与元数据已保存: {filepath}")

if __name__ == '__main__':
    print("STGNN— 周期时间特征 + 滞后特征 + 长窗口 + 正则/裁剪")

    forecaster = OptimizedTorchSTGNNForecast(sequence_length=SEQUENCE_LENGTH)

    df = forecaster.load_and_analyze_data(FILE_PATH)

    train_loader, val_loader, test_loader, X_test, y_test, in_ch = forecaster.prepare_data(df)

    forecaster.build_model(input_channels=in_ch)
    print(">>> sequence_length =", forecaster.sequence_length)
    print(">>> input_channels  =", in_ch)
    print(">>> target =", TARGET_NODE, "/", TARGET_FEATURE)

    print("\n开始训练 ...")
    forecaster.train(train_loader, val_loader, epochs=EPOCHS)

    print("\n评估 ...")
    metrics, y_true, y_pred = forecaster.evaluate(test_loader, X_test, y_test, in_ch)

    forecaster.plot_training()
    forecaster.plot_test_comparison(y_true, y_pred)

    forecaster.save_model_with_metadata('stgnn_best.pt',
                                        metadata={'epochs': len(forecaster.history['loss']),
                                                  'final_metrics': metrics,
                                                  'model': 'STGNN_enhanced'})
    print("\nDone.")