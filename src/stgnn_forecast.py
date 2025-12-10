import os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent  # 项目根目录 …/xxc262
CONFIG = {
    "csv_path": str(ROOT / "raw" / "raw.csv"),

    # 图里的节点（和下面的映射一一对应）
    "nodes": ["西二线", "西三线"],

    # 每个节点的列名映射 —— 与 raw.csv 完全一致（逐字匹配）
    "node_feature_map": {
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
    },

    # 要预测谁
    "target_node": "西三线",
    "target_feature": "in",   # 进站压力

    # 站点关系（只想单向就改成 [("西二线","西三线")]）
    "edges": [("西二线", "西三线"), ("西三线", "西二线")],

    # 其他保持不变即可
    "sequence_length": 30,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "batch_size_train": 128, "batch_size_val": 128, "batch_size_test": 256,
    "epochs": 50, "lr": 0.002, "patience": 6,
    "add_time_features": True,          # 你有 timestamp 列，开着就行
    "temporal_kernel": 3, "dropout": 0.1,
}

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']; plt.rcParams['axes.unicode_minus'] = False
# -----------------------------------------------------------------------------


# --------------------------- 数据集与滑窗 ---------------------------

class STSeqDataset(Dataset):
    """
    X: (N_samples, F_node, L, N_nodes)
    y: (N_samples,)  预测目标节点的下一时刻目标特征
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).unsqueeze(-1)

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# --------------------------- STGCN 模型 ---------------------------

def normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    """
    Kipf & Welling 一阶 GCN 规范化: A_hat = D^{-1/2} (A + I) D^{-1/2}
    A: (N_nodes, N_nodes)
    """
    I = torch.eye(A.size(0), device=A.device)
    A_tilde = A + I
    deg = torch.sum(A_tilde, dim=1)
    D_inv_sqrt = torch.diag(torch.pow(deg, -0.5))
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt


class TemporalConv(nn.Module):
    """
    Temporal 1D conv 实现在 (B, C, T, N) 的张量上：对时间维做卷积
    这里用 Conv2d(kernel=(k,1)) 等价于每个节点独立的 Temporal Conv
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                              padding=(padding, 0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C_in, T, N)
        x = self.conv(x)  # (B, C_out, T', N)  这里用 causal-like padding 简化（右裁剪）
        x = x[:, :, :- (self.conv.kernel_size[0]-1), :]  # 裁掉未来的信息
        x = self.relu(x)
        return self.dropout(x)


class GraphConv(nn.Module):
    """
    一阶近似 GCN：X' = X * A_hat
    对 (B, C, T, N) 在节点维乘 A_hat
    """
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, A_hat):
        # x: (B, C, T, N), A_hat: (N, N)
        x = torch.einsum('bctn,nm->bctm', x, A_hat)  # 节点维做图扩散
        x = self.relu(x)
        return self.dropout(x)


class STGCNBlock(nn.Module):
    """
    经典 STGCN Block：TempConv -> GraphConv -> TempConv
    """
    def __init__(self, c_in, c_out, k_t=3, dropout=0.1):
        super().__init__()
        self.temp1 = TemporalConv(c_in,  c_out, kernel_size=k_t, dropout=dropout)
        self.gcn   = GraphConv(dropout=dropout)
        self.temp2 = TemporalConv(c_out, c_out, kernel_size=k_t, dropout=dropout)

    def forward(self, x, A_hat):
        x = self.temp1(x)             # (B, c_out, T1, N)
        x = self.gcn(x, A_hat)        # (B, c_out, T1, N)
        x = self.temp2(x)             # (B, c_out, T2, N)
        return x


class STGCNRegressor(nn.Module):
    """
    输入: (B, F_node, L, N_nodes)
    输出: (B, 1)  预测目标节点的下一时刻值
    """
    def __init__(self, f_in, hidden=32, k_t=3, dropout=0.1):
        super().__init__()
        # 两个 STGCN Block
        self.block1 = STGCNBlock(f_in,   hidden, k_t=k_t, dropout=dropout)
        self.block2 = STGCNBlock(hidden, hidden, k_t=k_t, dropout=dropout)
        # 聚合：最后时间步，目标节点通道压缩
        self.head = nn.Sequential(
            nn.Conv2d(hidden, 16, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(16, 1, kernel_size=(1, 1))  # 输出通道为1
        )

    def forward(self, x, A_hat, target_node_index: int):
        # x: (B, F_node, L, N)
        x = self.block1(x, A_hat)   # (B, hidden, T1, N)
        x = self.block2(x, A_hat)   # (B, hidden, T2, N)

        # 取最后时间步（因果）
        x = x[:, :, -1:, :]         # (B, hidden, 1, N)

        # 取目标节点
        x = x[:, :, :, target_node_index:target_node_index+1]  # (B, hidden, 1, 1)

        # 头部回归
        x = self.head(x)            # (B, 1, 1, 1)
        return x.view(-1, 1)


# --------------------------- 数据处理流水线 ---------------------------

def build_adjacency(nodes: List[str], edges: List[Tuple[str, str]]) -> np.ndarray:
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    A = np.zeros((N, N), dtype=np.float32)
    for u, v in edges:
        if u in idx and v in idx:
            A[idx[u], idx[v]] = 1.0
    return A


def load_and_basic_check(csv_path: str) -> pd.DataFrame:
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # 可选：若有时间列，排序并派生时间特征
    time_col = None
    for c in ["timestamp", "time", "datetime", "ts"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
            df = df.sort_values(c).reset_index(drop=True)
            time_col = c
            break
    print("[Load] shape=", df.shape, "| cols=", list(df.columns))
    return df


def assemble_node_feature_matrix(df: pd.DataFrame,
                                 nodes: List[str],
                                 node_feature_map: Dict[str, Dict[str, str]],
                                 add_time_features: bool) -> Tuple[np.ndarray, List[str]]:
    """
    返回：X_all (T, N_nodes, F_node) 以及 feature_names
    每个节点统一使用 [in, out, temp]（若列缺失将报错或跳过）
    可选拼接时间特征（对所有节点复制一份）
    """
    N = len(nodes)
    # 基础每节点特征顺序
    base_keys = ["in", "out", "temp"]
    base_cols_per_node = []
    for n in nodes:
        fmap = node_feature_map[n]
        cols = [fmap[k] for k in base_keys if k in fmap and fmap[k] in df.columns]
        if len(cols) < 1:
            raise ValueError(f"Node {n} has no valid feature columns; check mapping.")
        base_cols_per_node.append(cols)

    # 先拼基础特征
    node_feats = []
    for n, cols in zip(nodes, base_cols_per_node):
        node_feats.append(df[cols].to_numpy())  # (T, F_node_base_n)
    # 对齐每个节点的特征维度：以最多列为基准，不足的补齐（简单起见这里直接要求一致）
    f_dims = [arr.shape[1] for arr in node_feats]
    if len(set(f_dims)) != 1:
        raise ValueError(f"Each node must have same #features. Got {f_dims}")
    T, F_node_base = node_feats[0].shape
    X_base = np.stack(node_feats, axis=1)  # (T, N_nodes, F_node_base)

    feature_names = base_keys[:F_node_base]

    # 时间特征（可选）：hour, dow
    if add_time_features:
        hour = df["timestamp"].dt.hour.to_numpy() if "timestamp" in df.columns else \
               df["time"].dt.hour.to_numpy() if "time" in df.columns else \
               df["datetime"].dt.hour.to_numpy() if "datetime" in df.columns else \
               df["ts"].dt.hour.to_numpy() if "ts" in df.columns else None
        dow = df["timestamp"].dt.dayofweek.to_numpy() if "timestamp" in df.columns else \
              df["time"].dt.dayofweek.to_numpy() if "time" in df.columns else \
              df["datetime"].dt.dayofweek.to_numpy() if "datetime" in df.columns else \
              df["ts"].dt.dayofweek.to_numpy() if "ts" in df.columns else None
        if hour is not None and dow is not None:
            # 将时间特征复制到每个节点（广播）
            time_feat = np.stack([hour, dow], axis=1)  # (T, 2)
            time_feat = np.expand_dims(time_feat, axis=1)  # (T,1,2)
            time_feat = np.repeat(time_feat, repeats=X_base.shape[1], axis=1)  # (T,N,2)
            X_all = np.concatenate([X_base, time_feat], axis=2)  # (T,N,F_node_base+2)
            feature_names = feature_names + ["hour", "dow"]
        else:
            X_all = X_base
    else:
        X_all = X_base

    return X_all, feature_names


def make_sliding_windows_stgnn(X_all: np.ndarray,
                               target_seq: np.ndarray,
                               L: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    X_all: (T, N_nodes, F_node_total)
    target_seq: (T,)  目标节点的目标特征时间序列
    返回：
    X: (N_samples, F_node_total, L, N_nodes)  -> 供 STGCN 输入
    y: (N_samples,)
    """
    T, N, F = X_all.shape
    Xs, ys = [], []
    for i in range(L, T):
        X_win = X_all[i-L:i, :, :]                 # (L, N, F)
        X_win = np.transpose(X_win, (2, 0, 1))     # (F, L, N)
        Xs.append(X_win)
        ys.append(target_seq[i])                   # 预测当前步
    X = np.stack(Xs, axis=0)                       # (S, F, L, N)
    y = np.array(ys, dtype=np.float32)             # (S,)
    return X, y


# --------------------------- 训练 / 评估 / 可视化 ---------------------------

def train_one(model, train_loader, val_loader, A_hat, epochs, lr, patience, device):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.999), eps=1e-7)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.3, patience=5, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    best_val = float('inf'); waited = 0
    hist = {"loss":[], "val_loss":[], "mae":[], "val_mae":[]}

    for ep in range(1, epochs+1):
        model.train(); trL=0.0; trM=0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb, A_hat, target_node_index=TARGET_IDX)  # (B,1)
            loss = loss_fn(pred, yb)
            loss.backward(); opt.step()
            trL += loss.item()*len(xb); trM += (pred.detach()-yb).abs().sum().item()
        trL /= len(train_loader.dataset); trM /= len(train_loader.dataset)

        model.eval(); vaL=0.0; vaM=0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb, A_hat, target_node_index=TARGET_IDX)
                loss = loss_fn(pred, yb)
                vaL += loss.item()*len(xb); vaM += (pred-yb).abs().sum().item()
        vaL /= len(val_loader.dataset); vaM /= len(val_loader.dataset)
        sched.step(vaL)

        hist["loss"].append(trL); hist["val_loss"].append(vaL)
        hist["mae"].append(trM);  hist["val_mae"].append(vaM)
        print(f"Epoch {ep:02d} | loss {trL:.6f} | val {vaL:.6f} | mae {trM:.6f} | val_mae {vaM:.6f}")

        if vaL < best_val - 1e-8:
            best_val, waited = vaL, 0
            torch.save(model.state_dict(), "stgnn_best.pt")
        else:
            waited += 1
            if waited >= patience:
                print("Early stopping."); break
    return hist


def predict_scaled(model, test_loader, A_hat, device):
    model.eval(); preds=[]
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            yb = model(xb, A_hat, target_node_index=TARGET_IDX).cpu().numpy().flatten()
            preds.append(yb)
    return np.concatenate(preds, axis=0)


def plot_test(y_true, y_pred, title="STGNN Test Diagnostics"):
    # 子集可视化
    n = len(y_true)
    idx = np.linspace(0, n-1, min(n, 500), dtype=int)
    yt, yp = y_true[idx], y_pred[idx]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes[0,0].plot(idx, yt, label='真实值'); axes[0,0].plot(idx, yp, label='预测值'); axes[0,0].legend(); axes[0,0].set_title('时间序列对比'); axes[0,0].grid(True, alpha=0.3)
    axes[0,1].scatter(yt, yp, s=12, alpha=0.6); mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    axes[0,1].plot([mn,mx],[mn,mx],'r--'); axes[0,1].set_title('实-预散点'); axes[0,1].grid(True, alpha=0.3)
    res = yp - yt
    axes[1,0].scatter(yp, res, s=12, alpha=0.6); axes[1,0].axhline(0, color='r', ls='--'); axes[1,0].set_title('残差分析'); axes[1,0].grid(True, alpha=0.3)
    axes[1,1].hist(res, bins=50, alpha=0.7, edgecolor='k'); axes[1,1].axvline(0, color='r', ls='--'); axes[1,1].set_title('残差分布'); axes[1,1].grid(True, alpha=0.3)

    rmse = float(np.sqrt(np.mean(res**2))); mae = float(np.mean(np.abs(res))); r2 = 1 - np.sum(res**2)/np.sum((yt-np.mean(yt))**2)
    fig.text(0.02, 0.98, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}', fontsize=12, va='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(); plt.show()


# --------------------------- 主流程 ---------------------------

if __name__ == "__main__":
    print("Using device:", DEVICE)

    # 1) 读数据 & 体检
    df = load_and_basic_check(CONFIG["csv_path"])

    # 2) 组装节点特征矩阵 (T, N_nodes, F_node)
    X_all, feature_names = assemble_node_feature_matrix(
        df,
        nodes=CONFIG["nodes"],
        node_feature_map=CONFIG["node_feature_map"],
        add_time_features=CONFIG["add_time_features"]
    )
    T, N_nodes, F_node = X_all.shape
    print(f"[Assemble] T={T}, N_nodes={N_nodes}, F_node={F_node} | feats={feature_names}")

    # 3) 目标序列（目标节点/特征）
    target_col = CONFIG["node_feature_map"][CONFIG["target_node"]][CONFIG["target_feature"]]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV.")
    target_seq_raw = df[target_col].to_numpy().astype(np.float32)  # (T,)

    # 4) 构建邻接并规范化
    A_np = build_adjacency(CONFIG["nodes"], CONFIG["edges"])
    A_t = torch.tensor(A_np, dtype=torch.float32, device=DEVICE)
    A_hat = normalize_adjacency(A_t)   # (N,N)
    TARGET_IDX = CONFIG["nodes"].index(CONFIG["target_node"])

    # 5) 时间切分（按比例）
    n_train = int(T * CONFIG["train_ratio"])
    n_val   = int(T * CONFIG["val_ratio"])
    train_slice = slice(0, n_train)
    val_slice   = slice(n_train, n_train + n_val)
    test_slice  = slice(n_train + n_val, T)

    # 6) 无泄露标准化：特征/目标各自独立 scaler，并且只在 train 上 fit
    feat_scaler = MinMaxScaler().fit(X_all[train_slice].reshape(n_train, -1))
    tgt_scaler  = MinMaxScaler().fit(target_seq_raw[train_slice].reshape(-1, 1))

    def scale_X(Xseg):
        Tseg = Xseg.shape[0]
        X2d = Xseg.reshape(Tseg, -1)
        X2d = feat_scaler.transform(X2d)
        return X2d.reshape(Tseg, N_nodes, F_node)

    X_train_all = scale_X(X_all[train_slice])
    X_val_all   = scale_X(X_all[val_slice])
    X_test_all  = scale_X(X_all[test_slice])

    y_train = tgt_scaler.transform(target_seq_raw[train_slice].reshape(-1,1)).ravel()
    y_val   = tgt_scaler.transform(target_seq_raw[val_slice].reshape(-1,1)).ravel()
    y_test  = tgt_scaler.transform(target_seq_raw[test_slice].reshape(-1,1)).ravel()

    # 7) 滑窗到 STGNN 输入：(S, F_node_total, L, N_nodes)
    L = CONFIG["sequence_length"]
    Xtr, ytr = make_sliding_windows_stgnn(X_train_all, y_train, L)
    Xva, yva = make_sliding_windows_stgnn(X_val_all,   y_val,   L)
    Xte, yte = make_sliding_windows_stgnn(X_test_all,  y_test,  L)
    print(f"[Window] train/val/test = {len(Xtr)}/{len(Xva)}/{len(Xte)} | input=(F={Xtr.shape[1]}, L={Xtr.shape[2]}, N={Xtr.shape[3]})")

    # 8) DataLoader
    train_loader = DataLoader(STSeqDataset(Xtr, ytr), batch_size=CONFIG["batch_size_train"], shuffle=True)
    val_loader   = DataLoader(STSeqDataset(Xva, yva), batch_size=CONFIG["batch_size_val"],   shuffle=False)
    test_loader  = DataLoader(STSeqDataset(Xte, yte), batch_size=CONFIG["batch_size_test"],  shuffle=False)

    # 9) 构建模型（输入通道=每节点特征数）
    in_channels = Xtr.shape[1]  # F_node_total
    model = STGCNRegressor(f_in=in_channels,
                           hidden=32,
                           k_t=CONFIG["temporal_kernel"],
                           dropout=CONFIG["dropout"]).to(DEVICE)
    print(model)

    # 10) 训练
    history = train_one(model, train_loader, val_loader, A_hat, epochs=CONFIG["epochs"],
                        lr=CONFIG["lr"], patience=CONFIG["patience"], device=DEVICE)

    # 11) 测试集推理 + 反归一化评估（单位回到 MPa）
    y_pred_scaled = predict_scaled(model, test_loader, A_hat, DEVICE)
    y_true_org = tgt_scaler.inverse_transform(yte.reshape(-1,1)).ravel()
    y_pred_org = tgt_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
    rmse = float(np.sqrt(mean_squared_error(y_true_org, y_pred_org)))
    mae  = float(mean_absolute_error(y_true_org, y_pred_org))
    r2   = float(r2_score(y_true_org, y_pred_org))
    print(f"[Test] RMSE={rmse:.4f} MPa | MAE={mae:.4f} MPa | R²={r2:.4f}")

    # 12) 训练曲线（可选）
    if history["loss"]:
        ep = range(1, len(history["loss"])+1)
        fig, axes = plt.subplots(1,2, figsize=(14,5))
        axes[0].plot(ep, history["loss"], label='训练损失'); axes[0].plot(ep, history["val_loss"], label='验证损失'); axes[0].legend(); axes[0].grid(True, alpha=0.3); axes[0].set_title("Loss")
        axes[1].plot(ep, history["mae"], label='训练MAE'); axes[1].plot(ep, history["val_mae"], label='验证MAE'); axes[1].legend(); axes[1].grid(True, alpha=0.3); axes[1].set_title("MAE")
        plt.tight_layout(); plt.show()

    # 13) 测试诊断图
    plot_test(y_true_org, y_pred_org, title="STGNN Test Diagnostics")

    # 14) 保存权重与元数据
    torch.save(model.state_dict(), "stgnn_best.pt")
    import pickle
    meta = dict(
        config=CONFIG, nodes=CONFIG["nodes"], feature_names=feature_names,
        A=A_np, sequence_length=L, input_channels=in_channels,
        feat_scaler=feat_scaler, target_scaler=tgt_scaler, history=history,
        target_node=CONFIG["target_node"], target_feature=CONFIG["target_feature"]
    )
    with open("stgnn_best_metadata.pkl", "wb") as f:
        pickle.dump(meta, f)
    print("Saved: stgnn_best.pt + stgnn_best_metadata.pkl")
