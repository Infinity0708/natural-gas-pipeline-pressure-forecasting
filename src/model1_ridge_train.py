import os
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


def read_csv_auto(path: str) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"Cannot read CSV: {path}")


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_supervised_direct(df: pd.DataFrame, feature_cols, target_col: str,
                           lookback: int, horizon: int):
    """
    Direct forecasting:
      input = past lookback hours [t-L+1..t] (flattened)
      label = target at t+horizon
    """
    X, y, t_out = [], [], []
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
        t_out.append(df.iloc[t + horizon]["time"])


    return np.asarray(X, np.float32), np.asarray(y, np.float32), pd.to_datetime(t_out)


def plot_curve(time_index, y_true, y_pred, out_path, title):
    plt.figure(figsize=(12, 4))
    plt.plot(time_index, y_true, label="Actual")
    plt.plot(time_index, y_pred, label="Forecast")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="raw/processed_data.csv")
    ap.add_argument("--target", default="出站压力-连木沁压气站")
    ap.add_argument("--lookback", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--outdir", default="runs/model1_ridge")
    ap.add_argument("--max_features", type=int, default=0,
                    help="0 = use all numeric features except time/target; >0 = keep first N")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = read_csv_auto(args.csv)
    df.columns = [str(c).strip() for c in df.columns]

    time_col = "时间" if "时间" in df.columns else df.columns[0]
    df["time"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    if args.target not in df.columns:
        raise ValueError(f"Target not found: {args.target}")

    # numeric conversion
    for c in df.columns:
        if c in ["time", time_col]:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # features = all numeric cols except target
    feature_cols = []
    for c in df.columns:
        if c in ["time", time_col, args.target]:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)

    if args.max_features and args.max_features > 0:
        feature_cols = feature_cols[:args.max_features]

    # time split 70/15/15
    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train:n_train + n_val].copy()
    df_test = df.iloc[n_train + n_val:].copy()

    # scalers fit on train only
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(df_train[feature_cols].values)
    y_scaler.fit(df_train[[args.target]].values)

    def scale(part):
        p = part.copy()
        p[feature_cols] = x_scaler.transform(p[feature_cols].values)
        p[args.target] = y_scaler.transform(p[[args.target]].values)
        return p

    tr_s = scale(df_train)
    va_s = scale(df_val)
    te_s = scale(df_test)

    # supervised
    Xtr, ytr, _ = make_supervised_direct(tr_s, feature_cols, args.target, args.lookback, args.horizon)
    Xva, yva, _ = make_supervised_direct(va_s, feature_cols, args.target, args.lookback, args.horizon)
    Xte, yte, tte = make_supervised_direct(te_s, feature_cols, args.target, args.lookback, args.horizon)

    if len(Xtr) < 100:
        raise RuntimeError(f"Too few train samples after windowing: {len(Xtr)}")

    # train on train+val
    Xtrva = np.vstack([Xtr, Xva])
    ytrva = np.concatenate([ytr, yva])

    model = Ridge(alpha=args.alpha, random_state=0)
    model.fit(Xtrva, ytrva)

    # predict (scaled)
    pred = model.predict(Xte)

    # inverse scale
    y_true = y_scaler.inverse_transform(yte.reshape(-1, 1)).reshape(-1)
    y_pred = y_scaler.inverse_transform(pred.reshape(-1, 1)).reshape(-1)

    metrics = {
        "model": "Ridge",
        "target": args.target,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "alpha": args.alpha,
        "test_mae": float(mean_absolute_error(y_true, y_pred)),
        "test_rmse": rmse(y_true, y_pred),
        "n_train_samples": int(len(Xtr)),
        "n_test_samples": int(len(Xte)),
        "n_raw_features": int(len(feature_cols)),
        "n_flat_features": int(Xtr.shape[1]),
    }

    # save
    joblib.dump(
        {
            "model": model,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "feature_cols": feature_cols,
            "metrics": metrics,
        },
        os.path.join(args.outdir, f"ridge_h{args.horizon}.joblib"),
    )

    pd.DataFrame([metrics]).to_csv(os.path.join(args.outdir, f"metrics_h{args.horizon}.csv"), index=False)

    title = f"Ridge Forecast | target={args.target} | horizon={args.horizon}h | lookback={args.lookback}h"
    plot_curve(
        tte,
        y_true,
        y_pred,
        os.path.join(args.outdir, f"forecast_curve_h{args.horizon}.png"),
        title
    )

    print("=== Ridge Training Done ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
