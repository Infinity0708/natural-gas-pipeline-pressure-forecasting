import os
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error


def read_csv_auto(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"Cannot read CSV: {path}")


def make_supervised_direct(df: pd.DataFrame, feature_cols, target_col: str,
                           lookback: int, horizon: int):
    X, y = [], []
    Xv = df[feature_cols].to_numpy(dtype=np.float32)
    yv = df[target_col].to_numpy(dtype=np.float32)

    n = len(df)
    for t in range(lookback - 1, n - horizon):
        x_win = Xv[t - lookback + 1:t + 1].reshape(-1)
        yy = yv[t + horizon]
        if np.any(np.isnan(x_win)) or np.isnan(yy):
            continue
        X.append(x_win)
        y.append(yy)

    return np.asarray(X, np.float32), np.asarray(y, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="raw/数据处理结果.csv")
    ap.add_argument("--target", default="出站压力-连木沁压气站")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--lookbacks", default="12,24,48")
    ap.add_argument("--alphas", default="0.01,0.1,1,10,100")
    ap.add_argument("--outdir", default="runs/hpo_ridge")
    ap.add_argument("--max_features", type=int, default=0)
    args = ap.parse_args()

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

    feature_cols = []
    for c in df.columns:
        if c in ["time", time_col, args.target]:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)

    if args.max_features and args.max_features > 0:
        feature_cols = feature_cols[:args.max_features]

    # train only split (we HPO on train portion)
    n = len(df)
    n_train = int(n * 0.70)
    df_train = df.iloc[:n_train].copy()

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaler.fit(df_train[feature_cols].values)
    y_scaler.fit(df_train[[args.target]].values)

    df_train_s = df_train.copy()
    df_train_s[feature_cols] = x_scaler.transform(df_train_s[feature_cols].values)
    df_train_s[args.target] = y_scaler.transform(df_train_s[[args.target]].values)

    lookbacks = [int(x) for x in args.lookbacks.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    tscv = TimeSeriesSplit(n_splits=5)

    results = []
    best = (1e18, None)

    for L in lookbacks:
        Xtr, ytr = make_supervised_direct(df_train_s, feature_cols, args.target, L, args.horizon)
        if len(Xtr) < 200:
            continue

        for a in alphas:
            maes = []
            model = Ridge(alpha=a, random_state=0)

            for tr_idx, va_idx in tscv.split(Xtr):
                model.fit(Xtr[tr_idx], ytr[tr_idx])
                pred = model.predict(Xtr[va_idx])
                maes.append(mean_absolute_error(ytr[va_idx], pred))

            mae = float(np.mean(maes))
            results.append({"lookback": L, "alpha": a, "cv_mae_scaled": mae})

            if mae < best[0]:
                best = (mae, (L, a))

    if best[1] is None:
        raise RuntimeError("No valid HPO result (too few samples). Try smaller lookback or check NaNs.")

    df_res = pd.DataFrame(results).sort_values("cv_mae_scaled", ascending=True)
    df_res.to_csv(os.path.join(args.outdir, f"ridge_hpo_h{args.horizon}.csv"), index=False)

    best_L, best_a = best[1]
    meta = {
        "best_cv_mae_scaled": best[0],
        "best_lookback": best_L,
        "best_alpha": best_a,
        "horizon": args.horizon,
        "target": args.target,
        "n_features": len(feature_cols),
    }
    joblib.dump(meta, os.path.join(args.outdir, f"best_ridge_h{args.horizon}.joblib"))

    print("=== Ridge HPO Done ===")
    for k, v in meta.items():
        print(f"{k}: {v}")
    print(f"[OK] Saved ranking CSV to: {os.path.join(args.outdir, f'ridge_hpo_h{args.horizon}.csv')}")


if __name__ == "__main__":
    main()
