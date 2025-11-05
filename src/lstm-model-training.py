import numpy as np  # 数值计算库，用于数组操作、数学运算
import pandas as pd  # 数据分析库，用于读取CSV、数据预处理、时间序列处理

import matplotlib.pyplot as plt  # 绘图库，用于绘制训练曲线、预测结果对比图
import seaborn as sns  # 统计可视化库，基于matplotlib，提供更美观的图表

from sklearn.preprocessing import MinMaxScaler  # 数据标准化，将特征缩放到[0,1]区间
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # 回归评估指标

from pathlib import Path
import os

# 【深度学习框架 - PyTorch】
import torch  # PyTorch主模块，提供张量操作、自动微分
import torch.nn as nn  # 神经网络模块，包含各种层定义（LSTM、Linear等）
from torch.utils.data import Dataset, DataLoader  # 数据加载器，用于批量处理训练数据
from torch.optim import Adam  # Adam优化器，用于参数更新
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 学习率调度器，根据验证损失自动调整学习率

# 【上下文管理】
from contextlib import nullcontext  # 空上下文管理器，用于条件性启用混合精度训练

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 设置中文字体
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
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}
        self.device, self.amp_enabled = self._setup_device_amp()

    def _setup_device_amp(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        amp_enabled = use_cuda
        print(f"设备: {device}, 混合精度: {'是' if amp_enabled else '否'}")
        return device, amp_enabled

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

        # ═══════════════════════════════════════════════════════════════
        # 特征工程 - 增强模型预测能力
        # ═══════════════════════════════════════════════════════════════
        # 【修复2】添加时间特征 - 捕获周期性模式
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

        all_feature_columns = feature_columns + engineered_features
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

        # ═══════════════════════════════════════════════════════════════
        # 数据标准化 - 防止数据泄露的关键改进
        # ═══════════════════════════════════════════════════════════════
        # 【重要修复】分别对特征和目标进行标准化
        # 原因：特征和目标的尺度不同，需要独立的标准化器
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))  # 特征标准化器
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))  # 目标标准化器

        # 【关键原则】只在训练集上拟合scaler，避免数据泄露
        # 测试集的信息不能用于训练过程的任何环节
        self.feature_scaler.fit(X_train_raw)  # 仅用训练集拟合特征scaler
        self.target_scaler.fit(y_train_raw.reshape(-1, 1))  # 仅用训练集拟合目标scaler

        # 【数据标准化】应用已拟合的scaler到所有数据集
        X_train_scaled = self.feature_scaler.transform(X_train_raw).astype(np.float32)
        X_val_scaled = self.feature_scaler.transform(X_val_raw).astype(np.float32)
        X_test_scaled = self.feature_scaler.transform(X_test_raw).astype(np.float32)

        y_train_scaled = self.target_scaler.transform(y_train_raw.reshape(-1, 1)).flatten().astype(np.float32)
        y_val_scaled = self.target_scaler.transform(y_val_raw.reshape(-1, 1)).flatten().astype(np.float32)
        y_test_scaled = self.target_scaler.transform(y_test_raw.reshape(-1, 1)).flatten().astype(np.float32)

        def build_sequences(X_arr, y_arr):
            """
            时间序列构建函数 - 滑动窗口方法详解

            序列构建原理图：
            ┌─────────────────────────────────────────────────────────────────┐
            │ 时间轴：  t-30  t-29  ...  t-2   t-1   t     t+1   t+2  ...     │
            │                                                                 │
            │ 第1个序列：                                                      │
            │   输入X: [t-30, t-29, ..., t-2, t-1]  (30个时间步的所有特征)    │
            │   输出y:                            t   (第t个时间步的目标值)    │
            │                                                                 │
            │ 第2个序列：                                                      │
            │   输入X:       [t-29, t-28, ..., t-1, t]   (滑动窗口向前移动)   │
            │   输出y:                                 t+1 (第t+1个时间步)     │
            │                                                                 │
            │ 第3个序列：                                                      │
            │   输入X:             [t-28, ..., t, t+1]   (继续滑动)           │
            │   输出y:                              t+2   (第t+2个时间步)      │
            │                                                                 │
            │ ...以此类推，直到数据末尾                                        │
            └─────────────────────────────────────────────────────────────────┘

            关键设计原则：
            1. 🔍 严格时间顺序：输入序列的时间范围 < 输出目标的时间点
            2. 🚫 避免数据泄露：输入特征不包含未来信息
            3. 📊 滑动窗口：每次移动一个时间步，最大化数据利用
            4. 🎯 监督学习：每个输入序列对应一个明确的目标值

            数据流向详解：
            原始数据 → 滑动窗口切片 → 3D张量(样本数, 序列长度, 特征数) → LSTM输入

            参数：
            - X_arr: 标准化后的特征数组，形状 (时间步数, 特征数)
            - y_arr: 标准化后的目标数组，形状 (时间步数,)

            返回：
            - Xs: 输入序列数组，形状 (样本数, sequence_length, 特征数)
            - ys: 目标值数组，形状 (样本数,)

            示例（假设sequence_length=3）：
            输入数据：X_arr.shape=(100, 5), y_arr.shape=(100,)
            输出数据：Xs.shape=(97, 3, 5), ys.shape=(97,)
            解释：从第3个时间步开始，每个样本使用前3个时间步预测当前时间步
            """

            # ═══════════════════════════════════════════════════════════════
            # 序列构建核心逻辑
            # ═══════════════════════════════════════════════════════════════
            Xs, ys = [], []  # 初始化输入序列列表和目标值列表

            # 【关键循环】从第sequence_length个时间步开始遍历
            # 原因：需要前sequence_length个时间步作为输入序列
            for i in range(self.sequence_length, len(X_arr)):
                # ───────────────────────────────────────────────────────────
                # 输入序列构建
                # ───────────────────────────────────────────────────────────
                # 【输入X】提取从 i-sequence_length 到 i-1 的所有特征
                # 时间范围：[i-30, i-29, ..., i-2, i-1]（共30个时间步）
                # 形状：(sequence_length, n_features) = (30, n_features)
                input_sequence = X_arr[i - self.sequence_length:i]
                Xs.append(input_sequence)  # 前30个时间点的特征

                # ───────────────────────────────────────────────────────────
                # 目标值提取
                # ───────────────────────────────────────────────────────────
                # 【输出y】提取第i个时间步的目标值
                # 时间点：第i个时间步（当前时间步）
                # 形状：标量值
                target_value = y_arr[i]
                ys.append(target_value)  # 当前时间点的目标值

            # ═══════════════════════════════════════════════════════════════
            # 数据类型转换和形状整理
            # ═══════════════════════════════════════════════════════════════
            # 【转换为numpy数组】确保数据类型一致性和内存效率
            Xs_array = np.array(Xs, dtype=np.float32)  # 形状：(样本数, sequence_length, 特征数)
            ys_array = np.array(ys, dtype=np.float32)  # 形状：(样本数,)

            # 【信息输出】显示序列构建结果（仅在第一次调用时输出）
            if len(Xs_array) > 0:
                print(f"    序列构建: 输入{Xs_array.shape} → 输出{ys_array.shape}")
                print(f"    时间窗口: {self.sequence_length}步 → 1步预测")

            return Xs_array, ys_array

        X_train, y_train = build_sequences(X_train_scaled, y_train_scaled)
        X_val, y_val = build_sequences(X_val_scaled, y_val_scaled)
        X_test, y_test = build_sequences(X_test_scaled, y_test_scaled)

        print(f"训练/验证/测试样本: {len(X_train)}/{len(X_val)}/{len(X_test)}; 特征数: {X_train.shape[2]}")

        train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=128, shuffle=True)
        val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=128, shuffle=False)
        test_loader = DataLoader(SequenceDataset(X_test, y_test), batch_size=256, shuffle=False)

        return train_loader, val_loader, test_loader, X_test, y_test, len(all_feature_columns)

    def build_model(self, input_size):
        self.model = LSTMRegressor(input_size).to(self.device)
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"模型参数总数: {total_params:,}")
        return self.model

    def train(self, train_loader, val_loader, epochs=30):
        """
        模型训练函数 - 深度学习训练循环详解

        训练流程图：
        ┌─────────────────────────────────────────────────────────────────┐
        │ 初始化优化器、调度器、损失函数                                    │
        │                           ↓                                     │
        │ For each epoch:                                                 │
        │   ├─ 训练阶段 (model.train())                                   │
        │   │   ├─ For each batch:                                        │
        │   │   │   ├─ 前向传播 → 计算损失                                │
        │   │   │   ├─ 反向传播 → 计算梯度                                │
        │   │   │   └─ 参数更新 → 清零梯度                                │
        │   │   └─ 计算平均训练损失和MAE                                   │
        │   │                                                             │
        │   ├─ 验证阶段 (model.eval())                                    │
        │   │   ├─ 禁用梯度计算 (torch.no_grad())                         │
        │   │   ├─ For each batch: 前向传播 → 累积验证损失                 │
        │   │   └─ 计算平均验证损失和MAE                                   │
        │   │                                                             │
        │   ├─ 学习率调度 (基于验证损失)                                   │
        │   ├─ 记录训练历史                                               │
        │   ├─ 模型保存 (如果验证损失改善)                                 │
        │   └─ 早停检查 (如果验证损失不再改善)                             │
        └─────────────────────────────────────────────────────────────────┘

        参数：
        - train_loader: 训练数据加载器
        - val_loader: 验证数据加载器
        - epochs: 训练轮数

        返回：
        - self.history: 包含训练历史的字典
        """

        # ═══════════════════════════════════════════════════════════════
        # 训练组件初始化
        # ═══════════════════════════════════════════════════════════════
        model = self.model

        # 【优化器】Adam优化器，自适应学习率
        optimizer = Adam(
            model.parameters(),  # 模型参数
            lr=0.002,  # 初始学习率
            betas=(0.9, 0.999),  # 动量参数：β1控制梯度，β2控制梯度平方
            eps=1e-7  # 数值稳定性参数
        )

        # 【学习率调度器】根据验证损失自动调整学习率
        scheduler = ReduceLROnPlateau(
            optimizer,  # 要调度的优化器
            mode='min',  # 监控指标越小越好
            factor=0.3,  # 学习率衰减因子：lr = lr * 0.3
            patience=5,  # 容忍验证损失不改善的轮数
            min_lr=1e-6  # 最小学习率
        )

        # 【损失函数】均方误差损失，适用于回归任务
        criterion = nn.MSELoss()

        # 【混合精度训练】加速训练并节省显存
        scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        # 【早停机制】防止过拟合
        best_val = float('inf')  # 记录最佳验证损失
        patience, waited = 6, 0  # 早停容忍度和等待计数器

        # 【混合精度上下文】条件性启用自动混合精度
        autocast_ctx = torch.cuda.amp.autocast if self.amp_enabled else nullcontext

        # ═══════════════════════════════════════════════════════════════
        # 主训练循环
        # ═══════════════════════════════════════════════════════════════
        for epoch in range(1, epochs + 1):

            # ───────────────────────────────────────────────────────────
            # 训练阶段 - 模型参数更新
            # ───────────────────────────────────────────────────────────
            model.train()  # 启用训练模式：dropout生效，BatchNorm更新统计量
            train_loss, train_mae = 0.0, 0.0  # 初始化训练指标累积器

            # 遍历训练批次
            for Xb, yb in train_loader:
                # 【数据转移】将数据移动到GPU（如果可用）
                Xb, yb = Xb.to(self.device), yb.to(self.device)

                # 【梯度清零】防止梯度在批次间累积
                # set_to_none=True 比 zero_grad() 更高效
                optimizer.zero_grad(set_to_none=True)

                # 【前向传播】计算预测值和损失
                with autocast_ctx():  # 混合精度上下文
                    pred = model(Xb)  # 模型预测：(batch_size, 1)
                    loss = criterion(pred, yb)  # 计算MSE损失

                # 【反向传播】计算梯度并更新参数
                scaler.scale(loss).backward()  # 缩放损失防止梯度下溢
                scaler.step(optimizer)  # 更新模型参数
                scaler.update()  # 更新缩放因子

                # 【指标累积】用于计算epoch平均值
                train_loss += loss.item() * len(Xb)  # 加权损失累积
                train_mae += (pred.detach() - yb).abs().sum().item()  # MAE累积

            # 【训练指标计算】计算epoch平均指标
            train_loss /= len(train_loader.dataset)  # 平均训练损失
            train_mae /= len(train_loader.dataset)  # 平均训练MAE

            # ───────────────────────────────────────────────────────────
            # 验证阶段 - 模型性能评估
            # ───────────────────────────────────────────────────────────
            model.eval()  # 启用评估模式：dropout失效，BatchNorm使用固定统计量
            val_loss, val_mae = 0.0, 0.0  # 初始化验证指标累积器

            # 禁用梯度计算，节省内存和计算
            with torch.no_grad():
                for Xb, yb in val_loader:
                    # 数据转移到设备
                    Xb, yb = Xb.to(self.device), yb.to(self.device)

                    # 前向传播（仅推理，无梯度）
                    with autocast_ctx():
                        pred = model(Xb)
                        loss = criterion(pred, yb)

                    # 累积验证指标
                    val_loss += loss.item() * len(Xb)
                    val_mae += (pred - yb).abs().sum().item()

            # 【验证指标计算】
            val_loss /= len(val_loader.dataset)
            val_mae /= len(val_loader.dataset)

            # ───────────────────────────────────────────────────────────
            # 训练策略和监控
            # ───────────────────────────────────────────────────────────

            # 【学习率调度】根据验证损失调整学习率
            # 如果验证损失plateau，自动降低学习率
            scheduler.step(val_loss)

            # 【训练历史记录】保存每个epoch的指标用于分析
            self.history['loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)

            # 【训练进度输出】实时监控训练状态
            print(
                f"Epoch {epoch}/{epochs} - loss: {train_loss:.6f} - mae: {train_mae:.6f} - val_loss: {val_loss:.6f} - val_mae: {val_mae:.6f}")

            # ───────────────────────────────────────────────────────────
            # 模型保存和早停机制
            # ───────────────────────────────────────────────────────────

            # 【最佳模型保存】当验证损失改善时保存模型
            if val_loss < best_val - 1e-8:  # 1e-8是最小改善阈值，避免数值误差
                best_val, waited = val_loss, 0  # 更新最佳损失，重置等待计数
                torch.save(model.state_dict(), 'best_model.pt')  # 保存模型权重
                print("保存最佳模型: best_model.pt")
            else:
                # 【早停逻辑】验证损失未改善时增加等待计数
                waited += 1
                if waited >= patience:
                    print("早停触发，停止训练")
                    break  # 跳出训练循环

        print("训练完成")
        return self.history  # 返回训练历史用于分析

    def evaluate(self, test_loader, X_test, y_test, n_features):
        """
        模型评估函数 - 深度学习模型性能评估详解

        评估流程：
        ┌─────────────────────────────────────────────────────────────────┐
        │ 1. 设置模型为评估模式 (model.eval())                             │
        │                           ↓                                     │
        │ 2. 禁用梯度计算 (torch.no_grad())                               │
        │                           ↓                                     │
        │ 3. 批量预测：For each batch in test_loader                      │
        │    ├─ 数据转移到GPU                                             │
        │    ├─ 模型前向传播获得预测值                                     │
        │    └─ 收集所有预测结果                                           │
        │                           ↓                                     │
        │ 4. 数据反归一化：将标准化的预测值和真实值还原到原始尺度           │
        │                           ↓                                     │
        │ 5. 计算评估指标：RMSE、MAE、R²                                   │
        │                           ↓                                     │
        │ 6. 返回评估结果和原始尺度的预测值、真实值                         │
        └─────────────────────────────────────────────────────────────────┘

        参数：
        - test_loader: 测试数据加载器，包含标准化的测试序列
        - X_test: 测试特征数据（用于验证，实际未使用）
        - y_test: 测试目标值（标准化后的）
        - n_features: 特征数量（用于验证，实际未使用）

        返回：
        - metrics: 包含RMSE、MAE、R²的字典
        - y_true_org: 原始尺度的真实值
        - y_pred_org: 原始尺度的预测值

        数据流向：
        测试序列 → 模型预测 → 标准化预测值 → 反归一化 → 原始尺度预测值 → 评估指标
        """

        # ═══════════════════════════════════════════════════════════════
        # 模型评估模式设置
        # ═══════════════════════════════════════════════════════════════
        self.model.eval()  # 设置为评估模式：禁用dropout，BatchNorm使用固定统计量
        preds = []  # 初始化预测结果收集器

        # ═══════════════════════════════════════════════════════════════
        # 批量预测过程
        # ═══════════════════════════════════════════════════════════════
        # 【预测过程】在测试集上进行批量预测，不计算梯度以节省内存
        with torch.no_grad():  # 禁用梯度计算，仅进行前向传播
            for Xb, _ in test_loader:  # 遍历测试批次，忽略标签（因为是预测阶段）
                # 【数据转移】将输入序列转移到计算设备（GPU/CPU）
                Xb = Xb.to(self.device)

                # 【模型预测】
                # 输入：(batch_size, sequence_length, n_features)
                # 输出：(batch_size, 1) → 展平为 (batch_size,)
                yb = self.model(Xb).cpu().numpy().flatten()

                # 【预测收集】将当前批次的预测结果添加到列表中
                preds.append(yb)

        # 【预测合并】将所有批次的预测结果合并为一个数组
        # 形状：(total_test_samples,)
        y_pred_scaled = np.concatenate(preds, axis=0)

        # ═══════════════════════════════════════════════════════════════
        # 数据反归一化 - 关键修复点
        # ═══════════════════════════════════════════════════════════════
        # 【修复】使用专门的目标scaler进行反归一化
        # 这是修复数据泄露后的关键改进：分离特征和目标的标准化器

        # 【预测值反归一化】
        # 输入：标准化的预测值 (n_samples,) → 重塑为 (n_samples, 1)
        # 输出：原始尺度的预测值 (n_samples,)
        y_pred_org = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        # 【真实值反归一化】
        # 输入：标准化的真实值 (n_samples,) → 重塑为 (n_samples, 1)
        # 输出：原始尺度的真实值 (n_samples,)
        y_true_org = self.target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # ═══════════════════════════════════════════════════════════════
        # 评估指标计算
        # ═══════════════════════════════════════════════════════════════
        # 【评估指标计算】基于原始尺度的数据计算性能指标

        # 【RMSE】均方根误差 - 衡量预测值与真实值的平均偏差程度
        # 公式：√(Σ(y_true - y_pred)² / n)
        # 单位：与目标变量相同（MPa）
        rmse = np.sqrt(mean_squared_error(y_true_org, y_pred_org))

        # 【MAE】平均绝对误差 - 衡量预测误差的平均绝对值
        # 公式：Σ|y_true - y_pred| / n
        # 单位：与目标变量相同（MPa）
        mae = mean_absolute_error(y_true_org, y_pred_org)

        # 【R²】决定系数 - 衡量模型解释目标变量变异的比例
        # 公式：1 - SS_res/SS_tot，其中SS_res=Σ(y_true-y_pred)²，SS_tot=Σ(y_true-y_mean)²
        # 范围：(-∞, 1]，越接近1表示模型拟合越好
        r2 = r2_score(y_true_org, y_pred_org)

        # 【结果输出】打印评估指标，便于实时监控模型性能
        print(f"RMSE: {rmse:.4f} MPa, MAE: {mae:.4f} MPa, R²: {r2:.4f}")

        # 【返回结果】
        # 1. metrics: 评估指标字典，用于程序化处理
        # 2. y_true_org: 原始尺度真实值，用于可视化分析
        # 3. y_pred_org: 原始尺度预测值，用于可视化分析
        return {'RMSE': rmse, 'MAE': mae, 'R²': r2}, y_true_org, y_pred_org

    def plot_test_comparison(self, y_true, y_pred, sample_size=500):
        if len(y_true) > sample_size:
            idx = np.linspace(0, len(y_true) - 1, sample_size, dtype=int)
            yt, yp = y_true[idx], y_pred[idx]
        else:
            yt, yp = y_true, y_pred
            idx = np.arange(len(yt))
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes[0, 0].plot(idx, yt, label='真实值', color='blue', linewidth=1.5)
        axes[0, 0].plot(idx, yp, label='预测值', color='red', linewidth=1.5)
        axes[0, 0].set_title('测试集预测结果对比-时间序列');
        axes[0, 0].legend();
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].scatter(yt, yp, alpha=0.6, s=20, color='green')
        mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        axes[0, 1].plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='理想预测线');
        axes[0, 1].legend();
        axes[0, 1].grid(True, alpha=0.3)
        res = yp - yt
        axes[1, 0].scatter(yp, res, alpha=0.6, s=20, color='purple');
        axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_title('残差分析');
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].hist(res, bins=50, alpha=0.7, color='orange', edgecolor='black');
        axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_title('误差分布');
        axes[1, 1].grid(True, alpha=0.3)
        rmse = np.sqrt(np.mean(res ** 2));
        mae = np.mean(np.abs(res));
        r2 = 1 - np.sum(res ** 2) / np.sum((yt - np.mean(yt)) ** 2)
        fig.text(0.02, 0.98, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}', fontsize=12, va='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.tight_layout();
        plt.show()

    def plot_training(self):
        if not self.history or not self.history['loss']:
            print("没有训练历史记录")
            return
        h = self.history
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        ep = range(1, len(h['loss']) + 1)
        axes[0].plot(ep, h['loss'], 'b-', label='训练损失')
        axes[0].plot(ep, h['val_loss'], 'r-', label='验证损失')
        axes[0].set_title('损失变化');
        axes[0].legend();
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(ep, h['mae'], 'b-', label='训练MAE')
        axes[1].plot(ep, h['val_mae'], 'r-', label='验证MAE')
        axes[1].set_title('MAE变化');
        axes[1].legend();
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout();
        plt.show()

    def save_model_with_metadata(self, filepath='optimized_lstm_torch.pt', metadata=None):
        torch.save(self.model.state_dict(), filepath)
        import pickle
        meta = {
            'sequence_length': self.sequence_length,
            'feature_scaler': getattr(self, 'feature_scaler', None),
            'target_scaler': getattr(self, 'target_scaler', None),
            'scaler': getattr(self, 'scaler', None),  # 保持向后兼容
            'history': self.history,
            'model_arch': {'input_size': self.model.lstm1.input_size},
            'training_metadata': metadata or {}
        }
        with open(filepath.replace('.pt', '_metadata.pkl'), 'wb') as f:
            pickle.dump(meta, f)
        print(f"模型与元数据已保存: {filepath}")

    def load_model_with_metadata(self, filepath='optimized_lstm_torch.pt'):
        try:
            import pickle
            with open(filepath.replace('.pt', '_metadata.pkl'), 'rb') as f:
                meta = pickle.load(f)
            self.sequence_length = meta.get('sequence_length', self.sequence_length)

            # 加载新的scaler或回退到旧的scaler
            self.feature_scaler = meta.get('feature_scaler', None)
            self.target_scaler = meta.get('target_scaler', None)
            if self.feature_scaler is None or self.target_scaler is None:
                self.scaler = meta.get('scaler', MinMaxScaler())

            input_size = meta.get('model_arch', {}).get('input_size', 4)
            self.build_model(input_size)
            self.model.load_state_dict(torch.load(filepath, map_location=self.device))
            print("模型与元数据加载成功")
            self.history = meta.get('history', self.history)
            return True
        except Exception as e:
            print(f"加载失败: {e}")
            return False

    def continue_training(self, train_loader, val_loader, additional_epochs=10, new_learning_rate=None):
        if self.model is None:
            print("请先构建或加载模型")
            return None
        optimizer = Adam(self.model.parameters(), lr=new_learning_rate or 0.002)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, min_lr=1e-6)
        criterion = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        autocast_ctx = torch.cuda.amp.autocast if self.amp_enabled else nullcontext
        best_val = min(self.history['val_loss']) if self.history['val_loss'] else float('inf')
        waited, patience = 0, 8
        for epoch in range(1, additional_epochs + 1):
            self.model.train();
            tl, tm = 0.0, 0.0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with autocast_ctx():
                    pred = self.model(Xb)
                    loss = criterion(pred, yb)
                scaler.scale(loss).backward();
                scaler.step(optimizer);
                scaler.update()
                tl += loss.item() * len(Xb);
                tm += (pred.detach() - yb).abs().sum().item()
            tl /= len(train_loader.dataset);
            tm /= len(train_loader.dataset)
            self.model.eval();
            vl, vm = 0.0, 0.0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(self.device), yb.to(self.device)
                    with autocast_ctx():
                        pred = self.model(Xb)
                        loss = criterion(pred, yb)
                    vl += loss.item() * len(Xb);
                    vm += (pred - yb).abs().sum().item()
            vl /= len(val_loader.dataset);
            vm /= len(val_loader.dataset)
            scheduler.step(vl)
            self.history['loss'].append(tl);
            self.history['val_loss'].append(vl)
            self.history['mae'].append(tm);
            self.history['val_mae'].append(vm)
            print(f"Continue Epoch - loss: {tl:.6f} - val_loss: {vl:.6f}")
            if vl < best_val - 1e-8:
                best_val, waited = vl, 0
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                waited += 1
                if waited >= patience:
                    print("早停触发，停止继续训练")
                    break
        return self.history


if __name__ == '__main__':
    """
    主函数 - 修复版本

    主要修复：
    1. 移除了数据泄露问题
    2. 使用更合理的特征工程
    3. 分离特征和目标的标准化
    """

    print("=" * 60)
    print("LSTM时间序列预测模型 - 修复版本")
    print("主要修复：移除数据泄露，重新设计特征工程")
    print("=" * 60)

    # 【数据文件路径】
    BASE_DIR = Path(__file__).resolve().parent  # …/xxc262/src
    ROOT_DIR = BASE_DIR.parent  # …/xxc262
    file_path = (ROOT_DIR / "raw" / "raw.csv").resolve()

    print("cwd =", os.getcwd())
    print("Using data file:", file_path)

    # 【模型初始化】
    # sequence_length=30：使用前30个时间点预测下一个时间点
    forecaster = OptimizedTorchLSTMForecast(sequence_length=30)

    # 【数据加载和分析】
    # 加载CSV文件，进行基本的数据清洗和分析
    df = forecaster.load_and_analyze_data(file_path)

    # 【数据预处理 - 已修复数据泄露问题】
    # 这一步创建了训练、验证、测试数据加载器
    train_loader, val_loader, test_loader, X_test, y_test, n_feat = forecaster.prepare_data(df)

    # 【模型构建】
    forecaster.build_model(input_size=n_feat)

    # 【模型训练】
    print("\n开始训练修复后的模型...")
    forecaster.train(train_loader, val_loader, epochs=50)  # 增加训练轮数

    # 【模型评估】
    print("\n评估修复后的模型性能...")
    metrics, y_true, y_pred = forecaster.evaluate(test_loader, X_test, y_test, n_feat)

    # 【结果可视化】
    # 绘制训练历史和预测结果对比图
    forecaster.plot_training()
    forecaster.plot_test_comparison(y_true, y_pred)

    # 【模型保存】
    # 保存训练好的模型和相关元数据
    forecaster.save_model_with_metadata('lstm_fixed_torch.pt',
                                        metadata={
                                            'epochs': len(forecaster.history['loss']),
                                            'final_metrics': metrics,
                                            'data_leakage_fixed': True,
                                            'feature_engineering': 'enhanced'
                                        })

    print("\n" + "=" * 60)
    print("训练完成！修复后的模型性能:")
    print(f"RMSE: {metrics['RMSE']:.4f} MPa")
    print(f"MAE: {metrics['MAE']:.4f} MPa")
    print(f"R²: {metrics['R²']:.4f}")
    print("=" * 60)