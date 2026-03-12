import os
import sys
import json
import argparse
from pathlib import Path

import optuna

# add project_root/src to sys.path
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import model5_cnn_bilstm_train as M  # noqa


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="raw/数据处理结果.csv")
    ap.add_argument("--target", default="出站压力-连木沁压气站")
    ap.add_argument("--target_mode", default="delta", choices=["abs", "delta"])
    ap.add_argument("--lookback", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=1)

    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--max_epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)

    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--drop_compressor_states", action="store_true")
    ap.add_argument("--include_target_lag", action="store_true")

    ap.add_argument("--outdir", default="runs/model5_cnn_bilstm/hpo")
    return ap


def main():
    args = build_argparser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # reuse the same data preparation as train script
    class Dummy:
        pass

    base = Dummy()
    base.csv = args.csv
    base.target = args.target
    base.target_mode = args.target_mode
    base.lookback = args.lookback
    base.horizon = args.horizon
    base.train_ratio = args.train_ratio
    base.val_ratio = args.val_ratio
    base.seed = args.seed
    base.drop_compressor_states = args.drop_compressor_states
    base.include_target_lag = args.include_target_lag

    # placeholders required by prepare_all
    base.channels = 128
    base.kernel_size = 7
    base.lstm_hidden = 64
    base.lstm_layers = 1
    base.dropout = 0.1
    base.fc = 64
    base.epochs = args.max_epochs
    base.lr = 3e-4
    base.batch_size = 128
    base.patience = args.patience
    base.weight_decay = 0.0
    base.outdir = args.outdir

    _, built = M.prepare_all(base)
    Xtr, ytr, ttr = built["Xtr"], built["ytr"], built["t_tr"]
    Xva, yva, tva = built["Xva"], built["yva"], built["t_va"]
    n_features = built["n_features"]
    n_samples = built["n_samples"]

    device = M.pick_device()

    print("=== CNN+BiLSTM HPO (Optuna) ===")
    print(f"target: {args.target} | target_mode: {args.target_mode}")
    print(f"lookback: {args.lookback} | horizon: {args.horizon}")
    print(f"features: {n_features}")
    print(f"trials: {args.trials} | max_epochs: {args.max_epochs} | patience: {args.patience}")
    print(f"outdir: {args.outdir}")
    print(f"drop_compressor_states: {args.drop_compressor_states} | include_target_lag: {args.include_target_lag}")
    print(f"train/val/test samples: {n_samples['train']} {n_samples['val']} {n_samples['test']}")
    print(f"device: {device}")

    def objective(trial: optuna.Trial) -> float:
        # search space
        channels = trial.suggest_categorical("channels", [32, 64, 128])
        kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7, 9])
        lstm_hidden = trial.suggest_categorical("lstm_hidden", [32, 64, 128])
        lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.45)
        fc = trial.suggest_categorical("fc", [32, 64, 128])
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        weight_decay = trial.suggest_float("weight_decay",1e-8, 1e-3, log=True)

        dl_tr = M.DataLoader(M.SeqDataset(Xtr, ytr, ttr), batch_size=batch_size, shuffle=False, drop_last=False)
        dl_va = M.DataLoader(M.SeqDataset(Xva, yva, tva), batch_size=batch_size, shuffle=False, drop_last=False)

        model = M.CNNBiLSTM(
            n_features=n_features,
            channels=channels,
            kernel_size=kernel_size,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
            fc=fc,
        ).to(device)

        optim = M.torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val = float("inf")
        bad = 0

        for ep in range(1, args.max_epochs + 1):
            _ = M.train_one_epoch(model, dl_tr, optim, device)
            val_rmse_scaled = M.eval_rmse_scaled(model, dl_va, device)

            trial.report(val_rmse_scaled, step=ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_rmse_scaled < best_val - 1e-9:
                best_val = val_rmse_scaled
                bad = 0
            else:
                bad += 1

            if bad >= args.patience:
                break

        return float(best_val)

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=args.trials)

    best = {
        "best_value(val_rmse_scaled)": float(study.best_value),
        "best_params": study.best_params,
        "fixed_args": {
            "csv": args.csv,
            "target": args.target,
            "target_mode": args.target_mode,
            "lookback": args.lookback,
            "horizon": args.horizon,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "drop_compressor_states": args.drop_compressor_states,
            "include_target_lag": args.include_target_lag,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
        }
    }

    out_path = os.path.join(args.outdir, "model5_cnn_bilstm_hpo_best.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved: {out_path}")
    print("best_value(val_rmse_scaled):", best["best_value(val_rmse_scaled)"])
    print("best_params:", best["best_params"])


if __name__ == "__main__":
    main()