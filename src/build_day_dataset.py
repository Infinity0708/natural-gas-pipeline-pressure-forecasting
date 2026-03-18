import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "raw")

HOURLY_CSV = os.path.join(RAW_DIR, "processed_data.csv")
DAILY_XLSX = os.path.join(RAW_DIR, "西部管道能耗数据模型-天然气1月.xlsx")

OUT_CSV = os.path.join(RAW_DIR, "day_dataset.csv")


def read_csv_auto(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise RuntimeError(f"无法读取CSV：{path}")


def build_cfg_day(hourly: pd.DataFrame) -> pd.DataFrame:
    cfg_cols = [c for c in hourly.columns if str(c).startswith("压缩机启停-")]
    if not cfg_cols:
        return pd.DataFrame(index=pd.Index([], name="day"))

    day = hourly.index.floor("D")
    # 日配置：当天运行小时数>=12记为开（阈值你可改）
    cfg_day = hourly[cfg_cols].groupby(day).sum(min_count=1).fillna(0)
    cfg_day = (cfg_day >= 12).astype(int)
    cfg_day.index.name = "day"
    return cfg_day


def build_daily_features(hourly: pd.DataFrame) -> pd.DataFrame:
    day = hourly.index.floor("D")

    # 基础三列（你小时表里应该有）
    base = {}
    for col in ["日输量", "首站压力", "沿线平均温度"]:
        if col in hourly.columns:
            base[col] = "mean"

    # 各站压力列
    for c in hourly.columns:
        if str(c).startswith(("进站压力-", "出站压力-")):
            base[c] = "mean"

    daily = hourly.groupby(day).agg(base)
    daily.index.name = "day"

    # 燃料气日能耗（小时消耗求和）
    fuel_cols = [c for c in hourly.columns if str(c).startswith("每小时燃料气消耗-")]
    if fuel_cols:
        fuel_day = hourly[fuel_cols].groupby(day).sum(min_count=1)
        daily["E_fuel_day"] = fuel_day.sum(axis=1)

    # 统一 Q_day：先用日输量（后面你可换成“出站量(万方)”口径）
    if "日输量" in daily.columns:
        daily["Q_day"] = daily["日输量"]
    else:
        daily["Q_day"] = np.nan

    # 统一 E_day
    daily["E_day"] = daily["E_fuel_day"] if "E_fuel_day" in daily.columns else np.nan
    daily["e_day"] = daily["E_day"] / daily["Q_day"]

    return daily


def read_daily_xlsx(path: str) -> pd.DataFrame:
    # 你这个文件的sheet名已确认存在
    xls = pd.ExcelFile(path, engine="openpyxl")

    # 先读“能耗相关接入数据”（通常更像结构化数据）
    sheet = "能耗相关接入数据" if "能耗相关接入数据" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    # 这个文件目前看更像“站场维度表”，可能没有日期列
    # 所以这里只是先返回原表，后续如果你有“按天”的sheet我们再接进去
    return df


def main():
    if not os.path.exists(HOURLY_CSV):
        raise FileNotFoundError(f"缺少：{HOURLY_CSV}")

    hourly = read_csv_auto(HOURLY_CSV)
    # 时间列：你小时表一般是“时间”
    time_col = "时间" if "时间" in hourly.columns else hourly.columns[0]
    hourly[time_col] = pd.to_datetime(hourly[time_col], errors="coerce")
    hourly = hourly.dropna(subset=[time_col]).set_index(time_col).sort_index()

    daily_feat = build_daily_features(hourly)
    cfg_day = build_cfg_day(hourly)

    out = daily_feat.join(cfg_day, how="left")

    # 读取日/站场xlsx（先验证可读，不强行join）
    if os.path.exists(DAILY_XLSX):
        _ = read_daily_xlsx(DAILY_XLSX)

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.to_csv(OUT_CSV, encoding="utf-8-sig")
    print(f"[OK] 输出：{OUT_CSV}")
    print("rows =", out.shape[0], "cols =", out.shape[1])
    print("示例列：", list(out.columns)[:15])


if __name__ == "__main__":
    main()
