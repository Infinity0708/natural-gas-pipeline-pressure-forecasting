# 【数据处理和数值计算】
import numpy as np
import pandas as pd

# 【机器学习预处理和评估】
from sklearn.preprocessing import MinMaxScaler  # 数据标准化，将特征缩放到[0,1]区间
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # 回归评估指标


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