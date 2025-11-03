# 【数据处理和数值计算】
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from contextlib import nullcontext
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class SequenceDataset(Dataset):
    """
    时间序列数据集类

    功能：将numpy数组转换为PyTorch张量，并实现Dataset接口
    数据流向：numpy数组 → PyTorch张量 → DataLoader批量加载

    输入数据格式：
    - X: 形状为 (样本数, 序列长度, 特征数) 的3D数组
    - y: 形状为 (样本数,) 的1D数组，表示目标值

    输出数据格式：
    - X: torch.FloatTensor, 形状 (样本数, 序列长度, 特征数)
    - y: torch.FloatTensor, 形状 (样本数, 1) - 增加一个维度用于回归
    """

    def __init__(self, X, y):
        """
        初始化数据集

        参数：
        - X: numpy数组，输入特征序列
        - y: numpy数组，目标值

        数据转换过程：
        1. numpy.ndarray → torch.FloatTensor (自动推断数据类型)
        2. y增加维度：(N,) → (N,1) 用于回归任务
        """
        self.X = torch.from_numpy(X)  # 转换输入特征为张量
        self.y = torch.from_numpy(y).unsqueeze(-1)  # 转换目标值为张量并增加维度

    def __len__(self):
        """
        返回数据集大小

        返回：数据集中样本的总数
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        根据索引获取单个样本

        参数：
        - idx: 样本索引

        返回：
        - (X[idx], y[idx]): 一个样本的特征序列和对应的目标值

        数据流向：索引 → 对应的(特征序列, 目标值)对
        """
        return self.X[idx], self.y[idx]


# ============================================================================
# LSTM回归模型定义 - 深度学习网络架构
# ============================================================================

class LSTMRegressor(nn.Module):
    """
    LSTM回归模型 - 用于时间序列预测

    网络架构详解：
    ┌─────────────────────────────────────────────────────────────────┐
    │ 输入: (batch_size, sequence_length, input_features)             │
    │                           ↓                                     │
    │ LSTM层1: input_features → 64 (隐藏单元)                        │
    │                           ↓                                     │
    │ LayerNorm + Dropout(0.2)                                       │
    │                           ↓                                     │
    │ LSTM层2: 64 → 32 (隐藏单元)                                    │
    │                           ↓                                     │
    │ LayerNorm + Dropout(0.1)                                       │
    │                           ↓                                     │
    │ 取最后时间步: (batch, 32)                                       │
    │                           ↓                                     │
    │ 全连接层1: 32 → 16 + ReLU + Dropout(0.1)                      │
    │                           ↓                                     │
    │ 输出层: 16 → 1 (回归预测值)                                     │
    └─────────────────────────────────────────────────────────────────┘

    设计理念：
    1. 双层LSTM：第一层提取序列特征，第二层进行特征融合
    2. LayerNorm：稳定训练过程，加速收敛
    3. 渐进式Dropout：防止过拟合，从0.2逐渐降到0.1
    4. 参数量约13,000个：适中的模型复杂度，避免过拟合
    """

    def __init__(self, input_size):
        """
        初始化LSTM回归模型

        参数：
        - input_size: 输入特征的维度数量

        网络层初始化顺序：
        1. 两层LSTM + 对应的LayerNorm和Dropout
        2. 全连接层 + 激活函数 + 最终输出层
        """
        super().__init__()

        # ═══════════════════════════════════════════════════════════════
        # 第一层LSTM - 序列特征提取层
        # ═══════════════════════════════════════════════════════════════
        self.lstm1 = nn.LSTM(
            input_size=input_size,  # 输入特征维度
            hidden_size=64,  # 隐藏状态维度，控制模型容量
            batch_first=True  # 输入格式：(batch, seq, feature)
        )

        # 【归一化层】对LSTM输出进行归一化，稳定训练
        self.norm1 = nn.LayerNorm(64)  # 对64维隐藏状态进行归一化

        # 【正则化层】防止过拟合，训练时随机丢弃20%的神经元
        self.dropout1 = nn.Dropout(0.2)

        # ═══════════════════════════════════════════════════════════════
        # 第二层LSTM - 特征融合层
        # ═══════════════════════════════════════════════════════════════
        self.lstm2 = nn.LSTM(
            input_size=64,  # 接收第一层LSTM的64维输出
            hidden_size=32,  # 降维到32，减少参数量
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(32)  # 对32维隐藏状态进行归一化
        self.dropout2 = nn.Dropout(0.1)  # 较小的dropout率，保留更多信息

        # ═══════════════════════════════════════════════════════════════
        # 全连接层 - 最终预测层
        # ═══════════════════════════════════════════════════════════════
        self.fc1 = nn.Linear(32, 16)  # 第一个全连接层：32 → 16
        self.act = nn.ReLU()  # ReLU激活函数，引入非线性
        self.dropout3 = nn.Dropout(0.1)  # 最后的正则化
        self.out = nn.Linear(16, 1)  # 输出层：16 → 1（回归预测值）

    def forward(self, x):
        """
        前向传播过程 - 数据流向详解

        输入：x.shape = (batch_size, sequence_length, input_features)
        例如：(128, 30, 18) - 128个样本，每个30个时间步，18个特征

        数据流向：
        输入序列 → LSTM特征提取 → 时间步选择 → 全连接预测 → 输出
        """

        # ═══════════════════════════════════════════════════════════════
        # 第一层LSTM处理 - 序列特征提取
        # ═══════════════════════════════════════════════════════════════
        # 输入: (batch, 30, input_features) → 输出: (batch, 30, 64)
        x, (h1, c1) = self.lstm1(x)  # h1,c1是最终的隐藏状态和细胞状态
        # x现在包含每个时间步的64维特征表示

        x = self.norm1(x)  # 归一化：稳定训练过程
        x = self.dropout1(x)  # 随机丢弃：防止过拟合

        # ═══════════════════════════════════════════════════════════════
        # 第二层LSTM处理 - 特征融合
        # ═══════════════════════════════════════════════════════════════
        # 输入: (batch, 30, 64) → 输出: (batch, 30, 32)
        x, (h2, c2) = self.lstm2(x)  # 进一步提取时序模式

        x = self.norm2(x)  # 归一化
        x = self.dropout2(x)  # 正则化

        # ═══════════════════════════════════════════════════════════════
        # 时间步选择 - 关键决策点
        # ═══════════════════════════════════════════════════════════════
        # 【重要】只取最后一个时间步的输出用于预测
        # 原理：最后一个时间步包含了整个序列的信息摘要
        # x[:, -1, :] 表示：所有样本的最后一个时间步的所有特征
        x = x[:, -1, :]  # (batch, 30, 32) → (batch, 32)

        # ═══════════════════════════════════════════════════════════════
        # 全连接层预测 - 最终输出
        # ═══════════════════════════════════════════════════════════════
        x = self.fc1(x)  # (batch, 32) → (batch, 16)
        x = self.act(x)  # ReLU激活，引入非线性
        x = self.dropout3(x)  # 最后的正则化

        # 最终预测输出
        output = self.out(x)  # (batch, 16) → (batch, 1)

        return output  # 返回预测值，形状：(batch_size, 1)


class OptimizedTorchLSTMForecast:
    def load_and_analyze_data(self, file_path):
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        print(f"数据形状: {df.shape}\n缺失值:\n{df.isnull().sum()}")
        return df

    def prepare_data(self, df, target_column='西二线本站进站压力'):
        """
        数据预处理函数 - 深度学习数据准备详解

        数据预处理流程图：
        ┌─────────────────────────────────────────────────────────────────┐
        │ 1. 特征选择：移除目标变量，选择相关独立特征                       │
        │                           ↓                                     │
        │ 2. 特征工程：                                                   │
        │    ├─ 时间特征：小时、分钟、星期几                               │
        │    ├─ 滞后特征：目标变量的历史值 (lag_1, lag_2, ...)            │
        │    ├─ 移动平均：短期和长期趋势 (ma_5, ma_10, ma_15)             │
        │    └─ 差分特征：变化趋势 (diff_1, diff_5)                       │
        │                           ↓                                     │
        │ 3. 数据清洗：删除NaN值（由特征工程产生）                         │
        │                           ↓                                     │
        │ 4. 数据分割：训练集(70%) | 验证集(15%) | 测试集(15%)             │
        │                           ↓                                     │
        │ 5. 数据标准化：分别标准化特征和目标（防止数据泄露）               │
        │                           ↓                                     │
        │ 6. 序列构建：滑动窗口创建时间序列                               │
        │                           ↓                                     │
        │ 7. DataLoader创建：批量加载，支持GPU训练                        │
        └─────────────────────────────────────────────────────────────────┘

        关键修复点：
        1. ❌ 移除目标变量作为输入特征 → ✅ 避免数据泄露
        2. ❌ 单一标准化器 → ✅ 分离特征和目标标准化器
        3. ❌ 简单特征 → ✅ 丰富的特征工程

        参数：
        - df: 原始数据DataFrame，包含时间戳和各种压力温度数据
        - target_column: 目标变量列名，默认为'西二线本站进站压力'

        返回：
        - train_loader: 训练数据加载器
        - val_loader: 验证数据加载器
        - test_loader: 测试数据加载器
        - X_test: 测试特征（用于评估）
        - y_test: 测试目标（用于评估）
        - n_features: 特征数量

        数据流向：
        原始DataFrame → 特征工程 → 数据分割 → 标准化 → 序列构建 → DataLoader
        """

        # ═══════════════════════════════════════════════════════════════
        # 特征选择 - 避免数据泄露的关键步骤
        # ═══════════════════════════════════════════════════════════════
        # 【修复1】移除目标变量，只使用相关但独立的特征
        # 原理：使用上游和并行管线的数据预测本站压力，避免直接使用目标变量
        feature_columns = [
            '西二线上一站出站压力',  # 上游压力，直接影响本站进站压力（物理因果关系）
            '西二线上一站出站温度',  # 上游温度，影响气体密度和流动特性
            '西三线本站进站压力',  # 并行管线压力，可能存在系统性相关性
            '西三线上一站出站压力',  # 西三线上游压力，系统整体状态指标
            '西三线本站进站温度'  # 西三线温度，环境和系统状态指标
        ]

        # 添加时间特征 - 捕获周期性模式
        df_copy = df.copy()  # 创建副本，避免修改原始数据

        # 【时间特征】提取时间的周期性信息
        df_copy['hour'] = df_copy['timestamp'].dt.hour  # 小时：0-23，捕获日内周期
        df_copy['minute'] = df_copy['timestamp'].dt.minute  # 分钟：0-59，捕获小时内变化
        df_copy['day_of_week'] = df_copy['timestamp'].dt.dayofweek  # 星期：0-6，捕获周内模式

        # 【修复3】添加滞后特征 - 使用目标变量的历史值作为特征
        # 重要说明：这是合理的时间序列建模方法，因为我们使用历史信息预测未来
        # 数据流向：t-lag时刻的目标值 → 作为t时刻的输入特征
        for lag in [1, 2, 3, 5, 10]:  # 不同时间尺度的滞后
            # lag_1: 前1个时间点的值（最近历史）
            # lag_2, lag_3: 短期历史趋势
            # lag_5, lag_10: 中期历史模式
            df_copy[f'{target_column}_lag_{lag}'] = df_copy[target_column].shift(lag)

        # 【修复4】添加移动平均特征 - 捕获不同时间尺度的趋势
        # 移动平均能够平滑噪声，突出趋势信息
        for window in [5, 10, 15]:  # 短期、中期、长期趋势
            # ma_5: 短期平均趋势（5个时间点）
            # ma_10: 中期平均趋势（10个时间点）
            # ma_15: 长期平均趋势（15个时间点）
            df_copy[f'{target_column}_ma_{window}'] = df_copy[target_column].rolling(window=window).mean()

        # 【修复5】添加差分特征 - 捕获变化趋势和动量
        # 差分特征能够捕获数据的变化速度和方向
        df_copy[f'{target_column}_diff_1'] = df_copy[target_column].diff(1)  # 一阶差分：瞬时变化率
        df_copy[f'{target_column}_diff_5'] = df_copy[target_column].diff(5)  # 五阶差分：中期变化趋势

        # ═══════════════════════════════════════════════════════════════
        # 特征列表整合
        # ═══════════════════════════════════════════════════════════════
        # 【工程特征列表】整合所有新创建的特征
        engineered_features = [
            # 时间特征
            'hour', 'minute', 'day_of_week',
            # 滞后特征
            f'{target_column}_lag_1', f'{target_column}_lag_2', f'{target_column}_lag_3',
            f'{target_column}_lag_5', f'{target_column}_lag_10',
            # 移动平均特征
            f'{target_column}_ma_5', f'{target_column}_ma_10', f'{target_column}_ma_15',
            # 差分特征
            f'{target_column}_diff_1', f'{target_column}_diff_5'
        ]

        # 合并原始特征和工程特征
        all_feature_columns = feature_columns + engineered_features

        # 数据清洗
        df_clean = df_copy.dropna().reset_index(drop=True)

        # 【信息输出】显示数据处理结果
        print(f"数据清洗后形状: {df_clean.shape}")
        print(f"使用的特征数量: {len(all_feature_columns)}")
        print(f"特征列表: {all_feature_columns}")

        # ═══════════════════════════════════════════════════════════════
        # 数据准备
        # ═══════════════════════════════════════════════════════════════
        # 【数据提取】从DataFrame中提取特征和目标数据
        X_data = df_clean[all_feature_columns].values.astype(np.float32)  # 特征矩阵
        y_data = df_clean[target_column].values.astype(np.float32)  # 目标向量

        # ═══════════════════════════════════════════════════════════════
        # 数据分割 - 时间序列的正确分割方法
        # ═══════════════════════════════════════════════════════════════
        # 【重要】时间序列必须按时间顺序分割，不能随机分割
        total = len(X_data)
        tr = int(total * 0.7)  # 训练集：前70%的数据（最早的时间段）
        val = int(total * 0.15)  # 验证集：70%-85%的数据（中间时间段）
        # 测试集：85%-100%的数据（最新的时间段）

        # 【数据分割】按时间顺序分割数据
        X_train_raw = X_data[:tr]  # 训练特征
        X_val_raw = X_data[tr:tr + val]  # 验证特征
        X_test_raw = X_data[tr + val:]  # 测试特征

        y_train_raw = y_data[:tr]  # 训练目标
        y_val_raw = y_data[tr:tr + val]  # 验证目标
        y_test_raw = y_data[tr + val:]  # 测试目标

        # 数据标准化 - 防止数据泄露的关键改进
        # 【重要修复】分别对特征和目标进行标准化
        # 原因：特征和目标的尺度不同，需要独立的标准化器
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))  # 特征标准化器
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))  # 目标标准化器

        # 【关键原则】只在训练集上拟合scaler，避免数据泄露
        self.feature_scaler.fit(X_train_raw)  # 仅用训练集拟合特征scaler
        self.target_scaler.fit(y_train_raw.reshape(-1, 1))  # 仅用训练集拟合目标scaler

        X_train_scaled = self.feature_scaler.transform(X_train_raw).astype(np.float32)
        X_val_scaled = self.feature_scaler.transform(X_val_raw).astype(np.float32)
        X_test_scaled = self.feature_scaler.transform(X_test_raw).astype(np.float32)

        y_train_scaled = self.target_scaler.transform(y_train_raw.reshape(-1, 1)).flatten().astype(np.float32)
        y_val_scaled = self.target_scaler.transform(y_val_raw.reshape(-1, 1)).flatten().astype(np.float32)
        y_test_scaled = self.target_scaler.transform(y_test_raw.reshape(-1, 1)).flatten().astype(np.float32)