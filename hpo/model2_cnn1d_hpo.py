import os
import json
import csv
import subprocess
from pathlib import Path

import optuna


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="runs/model2_cnn1d/hpo")
    ap.add_argument("--train_script", default="src/model2_cnn1d_training.py")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    trials_csv = outdir / "trials.csv"
    best_json = outdir / "best.json"

    if not trials_csv.exists():
        with open(trials_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "trial", "val_rmse",
                "lookback", "lr", "dropout", "channels", "kernel_size", "batch_size",
                "train_ratio", "val_ratio", "drop_compressor_states"
            ])

    def run_train(out_path: Path, params: dict) -> float:
        cmd = [
            os.environ.get("PYTHON", "python"),
            args.train_script,
            "--csv", args.csv,
            "--target", args.target,
            "--lookback", str(params["lookback"]),
            "--horizon", str(args.horizon),
            "--epochs", str(params["epochs"]),
            "--patience", str(params["patience"]),
            "--lr", str(params["lr"]),
            "--dropout", str(params["dropout"]),
            "--channels", str(params["channels"]),
            "--kernel_size", str(params["kernel_size"]),
            "--batch_size", str(params["batch_size"]),
            "--train_ratio", str(params["train_ratio"]),
            "--val_ratio", str(params["val_ratio"]),
            "--seed", str(args.seed),
            "--outdir", str(out_path),
        ]
        if params["drop_compressor_states"]:
            cmd.append("--drop_compressor_states")

        subprocess.check_call(cmd)

        rep = json.load(open(out_path / "model2_cnn1d_report.json", "r", encoding="utf-8"))
        return float(rep["metrics"]["val"]["rmse"])

    def objective(trial: optuna.Trial):
        params = {
            "lookback": trial.suggest_categorical("lookback", [24, 48, 72]),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "channels": trial.suggest_categorical("channels", [32, 64, 128, 256]),
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7, 9]),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "epochs": 100,
            "patience": 20,
            "train_ratio": 0.7,
            "val_ratio": 0.1,
            "drop_compressor_states": trial.suggest_categorical("drop_compressor_states", [False, True]),
        }

        trial_dir = outdir / f"trial_{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        val_rmse = run_train(trial_dir, params)

        with open(trials_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                trial.number, val_rmse,
                params["lookback"], params["lr"], params["dropout"], params["channels"],
                params["kernel_size"], params["batch_size"],
                params["train_ratio"], params["val_ratio"], params["drop_compressor_states"]
            ])

        return val_rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.trials)

    best = {"best_value(val_rmse)": study.best_value, "best_params": study.best_params}
    json.dump(best, open(best_json, "w", encoding="utf-8"), indent=2)

    # final retrain into runs/model2_cnn1d (top folder with fixed artifact names)
    final_out = Path("runs/model2_cnn1d")
    final_out.mkdir(parents=True, exist_ok=True)

    final_params = {
        **study.best_params,
        "epochs": 160,
        "patience": 25,
        "train_ratio": 0.7,
        "val_ratio": 0.1,
        "batch_size": study.best_params.get("batch_size", 128),
        "drop_compressor_states": study.best_params.get("drop_compressor_states", False),
    }

    # ensure we pass the same set of keys used in run_train
    val_rmse = run_train(final_out, final_params)

    print("[OK] HPO finished.")
    print("[OK] best.json:", best_json)
    print("[OK] trials.csv:", trials_csv)
    print("[OK] final artifacts saved to:", final_out)
    print("[OK] final val_rmse:", val_rmse)


if __name__ == "__main__":
    main()
