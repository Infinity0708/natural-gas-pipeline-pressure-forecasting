import pandas as pd
import os
import glob
import re

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "raw")
PROCESS_DIR = RAW_DATA_DIR

RAW_DIRS = [RAW_DATA_DIR]

POINT_LIST_FILE = os.path.join(PROCESS_DIR, "需处理点位表.csv")
OUTPUT_FILE = os.path.join(PROCESS_DIR, "数据处理结果.csv")


def load_file_map(directories):
    """
    扫描文件夹，建立 点位号 -> 文件路径 的映射。
    假设文件名以 "{点位号}.csv" 结尾。
    """
    print("正在扫描原始数据文件...")
    file_map = {}
    for d in directories:
        if not os.path.exists(d):
            print(f"警告: 文件夹不存在 {d}")
            continue
        
        # 获取所有csv文件
        files = glob.glob(os.path.join(d, "*.csv"))
        for f in files:
            # 文件名示例: 西三线 乌苏压气站 1#压缩机入口压力_WinCC_OA.XSX00D003_UCP1_79.In.values.value.csv
            # 或者直接是: WinCC_OA.XSX00D003_UCP1_79.In.values.value.csv
            # 我们假设点位号包含在文件名中，且通常在最后（去扩展名）
            basename = os.path.basename(f)
            name_no_ext = os.path.splitext(basename)[0]
            
            # 尝试提取点位号。策略：点位号通常包含 'WinCC_OA'
            # 如果文件名包含 WinCC_OA，则取从 WinCC_OA 开始的部分作为点位号
            if 'WinCC_OA' in name_no_ext:
                idx = name_no_ext.find('WinCC_OA')
                tag = name_no_ext[idx:]
                file_map[tag] = f
            else:
                # 如果不包含，则整个文件名作为key（或者忽略，视情况而定）
                # 根据题目描述，点位号如 WinCC_OA.XSX00D007...
                pass
    print(f"扫描完成，找到 {len(file_map)} 个相关文件。")
    return file_map

def read_data_series(file_path):
    """
    读取单个CSV文件，返回以时间为索引的Series。
    """
    try:
        # 尝试使用utf-8读取，如果失败尝试gbk
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='gbk')
            
        # 假设列名为: 时间戳, 数值, 质量码
        # 清理列名空格
        df.columns = [c.strip() for c in df.columns]
        
        if '时间戳' not in df.columns or '数值' not in df.columns:
            print(f"警告: 文件 {os.path.basename(file_path)} 列名不符合预期: {df.columns}")
            return None
            
        df['时间戳'] = pd.to_datetime(df['时间戳'])
        df.set_index('时间戳', inplace=True)
        # 转换数值列为float，处理可能的非数字字符
        df['数值'] = pd.to_numeric(df['数值'], errors='coerce')
        
        return df['数值']
    except Exception as e:
        print(f"读取文件失败 {file_path}: {e}")
        return None

def process_data():
    # 1. 加载点位表
    if not os.path.exists(POINT_LIST_FILE):
        print(f"错误: 找不到点位表文件 {POINT_LIST_FILE}")
        return

    try:
        points_df = pd.read_csv(POINT_LIST_FILE, encoding='utf-8')
    except:
        points_df = pd.read_csv(POINT_LIST_FILE, encoding='gbk')
    
    # 清理列名
    points_df.columns = [c.strip() for c in points_df.columns]
    print(f"点位表加载成功，共 {len(points_df)} 行。")

    # 2. 建立文件映射
    file_map = load_file_map(RAW_DIRS)
    
    # 3. 准备数据容器
    # 我们需要一个统一的时间索引，先读取一个文件来获取时间范围，或者在合并时处理
    # 为避免内存溢出，我们按类别处理并存储中间结果
    
    # 结果DataFrame列表
    results = []
    
    # --- 处理逻辑 ---
    
    # 提取所有需要的点位数据并进行重采样
    # 为了效率，我们按点位遍历，处理完后存入字典
    # data_cache = { '点位号': Series(Hourly) }
    data_cache = {}
    
    print("开始读取并预处理数据...")
    for idx, row in points_df.iterrows():
        tag = row['点位号']
        if tag in file_map:
            series = read_data_series(file_map[tag])
            if series is not None:
                # 预处理：按小时重采样
                # 对于累积量（燃料气），我们取每小时最后一个值（或最大值）
                # 对于瞬时量（压力、温度、流量、转速），我们取平均值
                signal_desc = row['信号描述']
                
                if '累积' in signal_desc or '累计' in signal_desc:
                    # 累积量，取最大值代表该小时的读数（假设单调递增）
                    hourly_data = series.resample('H').max()
                else:
                    # 瞬时量，取平均
                    hourly_data = series.resample('H').mean()
                
                data_cache[tag] = hourly_data
        else:
            print(f"警告: 未找到点位文件 {tag}")

    if not data_cache:
        print("错误: 未读取到任何有效数据。")
        return

    # 创建统一的时间索引 (取并集)
    all_indexes = pd.Index([])
    for s in data_cache.values():
        all_indexes = all_indexes.union(s.index)
    all_indexes = all_indexes.sort_values()
    
    # 创建最终的DataFrame
    final_df = pd.DataFrame(index=all_indexes)
    final_df.index.name = '时间'
    
    print("正在计算各个指标...")
    
    # 辅助函数：获取某站点某类型的所有点位数据
    def get_station_tags(station, signal_type_keyword):
        tags = []
        for _, row in points_df.iterrows():
            if row['区域'] == station and signal_type_keyword in row['信号描述']:
                tags.append(row['点位号'])
        return tags

    # 获取所有站点.
    stations = points_df['区域'].unique()
    stations = [s for s in stations if pd.notna(s) and s.strip() != '']

    # 1. 日输量
    # 规则：日输量 = (西二线压缩机进口瞬时流量 * 1000 + 西三线压缩机进口瞬时流量 * 100) * 24
    # 特殊规则：2025-02-01 以后，西三线 2# 和 3# 压缩机的数据要 / 10 (即 * 100 / 10 = * 10)
    
    total_flow = pd.Series(0.0, index=final_df.index)
    
    # 获取连木沁压气站的所有工艺气进口瞬时流量点位
    lianmuqin_rows = points_df[
        (points_df['区域'] == '连木沁压气站') & 
        (points_df['信号描述'] == '工艺气进口瞬时流量')
    ]

    for _, row in lianmuqin_rows.iterrows():
        tag = row['点位号']
        line = row['线']       # 西气东输二线 / 西气东输三线
        device = row['设备']     # 1#压缩机 / 西三线2#压缩机 等

        if tag in data_cache:
            series = data_cache[tag].fillna(0)
            
            weighted_series = None
            
            if '西气东输二线' in line:
                # 西二线：流量 * 1000
                weighted_series = series * 1000
            elif '西气东输三线' in line:
                # 西三线：基础 * 100
                # 检查是否是西三线 2# 或 3# 压缩机
                is_special_device = ('2#' in device) or ('3#' in device)
                
                if is_special_device:
                    # 分段处理
                    cutoff_date = pd.Timestamp('2025-02-01')
                    
                    # 2025-02-01 之前: * 100
                    part1 = series[series.index < cutoff_date] * 100
                    
                    # 2025-02-01 之后: / 10 * 100 = * 10
                    part2 = series[series.index >= cutoff_date] * 10
                    
                    # 合并并对齐索引
                    weighted_series = pd.concat([part1, part2]).reindex(series.index).fillna(0)
                else:
                    # 西三线 1# 压缩机: * 100
                    weighted_series = series * 100
            
            if weighted_series is not None:
                total_flow = total_flow.add(weighted_series, fill_value=0)
    
    # 最后总和 * 24
    final_df['日输量'] = total_flow * 24 / 10000

    # 2. 沿线平均温度 (所有 进站温度 的平均值)
    temp_tags = []
    for _, row in points_df.iterrows():
        if '进站温度' in row['信号描述']:
            temp_tags.append(row['点位号'])
    
    if temp_tags:
        temp_df = pd.DataFrame({tag: data_cache[tag] for tag in temp_tags if tag in data_cache})
        final_df['沿线平均温度'] = temp_df.mean(axis=1)
    else:
        final_df['沿线平均温度'] = 0

    # 3. 各个站点的指标
    for station in stations:
        # 3.1 进站压力 (平均)
        in_pressure_tags = get_station_tags(station, '进站压力')
        if in_pressure_tags:
            p_df = pd.DataFrame({tag: data_cache[tag] for tag in in_pressure_tags if tag in data_cache})
            final_df[f'进站压力-{station}'] = p_df.mean(axis=1)
        
        # 3.2 出站压力 (平均)
        out_pressure_tags = get_station_tags(station, '出站压力')
        if out_pressure_tags:
            p_df = pd.DataFrame({tag: data_cache[tag] for tag in out_pressure_tags if tag in data_cache})
            final_df[f'出站压力-{station}'] = p_df.mean(axis=1)

        # 3.3 每小时燃料气消耗 (累积流量差值求和)
        fuel_tags = get_station_tags(station, '燃料气累积流量')
        station_fuel = pd.Series(0.0, index=final_df.index)
        has_fuel_data = False
        for tag in fuel_tags:
            if tag in data_cache:
                s = data_cache[tag]
                # 计算差值 (每小时消耗)
                diff = s.diff()
                # 过滤负值（处理计数器重置或异常）
                diff = diff.apply(lambda x: x if x >= 0 else 0) 
                station_fuel = station_fuel.add(diff.fillna(0), fill_value=0)
                has_fuel_data = True
        if has_fuel_data:
            final_df[f'每小时燃料气消耗-{station}'] = station_fuel

        # 3.4 压缩机启停 (转速 > 1500)
        # 遍历该站点的所有压缩机
        # 需处理点位表中，设备列指明了压缩机名称
        station_rows = points_df[points_df['区域'] == station]
        for _, row in station_rows.iterrows():
            if '压缩机转速' in row['信号描述']:
                tag = row['点位号']
                compressor_name = row['设备']
                # 如果设备名为空，尝试用描述补充，或直接忽略
                if pd.isna(compressor_name) or compressor_name == '':
                    compressor_name = "未知设备"
                
                col_name = f'压缩机启停-{station}-{compressor_name}'
                
                if tag in data_cache:
                    s = data_cache[tag]
                    # 判断状态: > 1500 为 1 (运行), 否则 0 (停机)
                    status = s.apply(lambda x: 1 if x > 1500 else 0)
                    final_df[col_name] = status

    # 4. 首站压力 (连木沁 二线和三线 出站压力 平均值)
    # 直接复用已经计算好的 '出站压力-连木沁压气站'，因为上面的逻辑已经包含了该站所有出站压力的平均
    if '出站压力-连木沁压气站' in final_df.columns:
        final_df['首站压力'] = final_df['出站压力-连木沁压气站']
    else:
        final_df['首站压力'] = 0 # 或 NaN

    # 5. 整理列顺序
    # 期望: 时间、日输量、首站压力、沿线平均温度、[出站压力...], [进站压力...], [燃料气...], [启停...]
    cols = list(final_df.columns)
    priority_cols = ['日输量', '首站压力', '沿线平均温度']
    
    # 移除优先列以便重新排序
    other_cols = [c for c in cols if c not in priority_cols]
    
    # 对其他列进行简单的分类排序，让同类指标在一起
    def sort_key(name):
        if name.startswith('出站压力'): return '1_' + name
        if name.startswith('进站压力'): return '2_' + name
        if name.startswith('每小时燃料气消耗'): return '3_' + name
        if name.startswith('压缩机启停'): return '4_' + name
        return '9_' + name
        
    other_cols.sort(key=sort_key)
    
    final_cols = priority_cols + other_cols
    # 确保只包含存在的列
    final_cols = [c for c in final_cols if c in final_df.columns]
    
    final_df = final_df[final_cols]
    
    # 保存结果
    print(f"正在保存结果到 {OUTPUT_FILE} ...")
    # encoding='utf-8-sig' 可以让Excel正确打开中文CSV
    try:
        final_df.to_csv(OUTPUT_FILE, encoding='utf-8-sig')
        print("处理完成！")
    except PermissionError:
        import time
        timestamp = int(time.time())
        new_output = OUTPUT_FILE.replace('.csv', f'_{timestamp}.csv')
        print(f"警告: 无法写入 {OUTPUT_FILE} (可能文件被占用)，尝试写入到 {new_output}")
        final_df.to_csv(new_output, encoding='utf-8-sig')
        print(f"处理完成！结果已保存至 {new_output}")

if __name__ == "__main__":
    process_data()
