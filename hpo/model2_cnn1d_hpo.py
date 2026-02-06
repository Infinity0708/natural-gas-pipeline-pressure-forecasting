# hpo/model2_cnn1d_hpo.py
from __future__ import annotations
import optuna
from pathlib import Path

from hpo.hpo_utils import make_run_dir, get_trial_dir, save_json, append_csv
from src.model2_cnn1d_train import OptimizedTorchCNN1DForecast

DATA_PATH_DEFAULT = Path(__file__).resolve().parent.parent / "raw" / "数据处理结果.csv"


def objective(trial: optuna.Trial, run_dir: Path, csv_path: Path, target: str, horizon: int) -> float:
    # ----- search space (similar style to your LSTM HPO) -----
    lookback = trial.suggest_categorical("lookback", [12, 24, 48])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    epochs = trial.suggest_int("epochs", 30, 90, step=15)
    patience = trial.suggest_int("patience", 8, 18)

    hparams_model = {
        "channels": trial.suggest_categorical("channels", [32, 64, 128]),
        "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
    }

    # fixed switches (keep same across trials for fair comparison)
    include_target_lags = True
    drop_compressor_states = False

    # ----- per-trial directory -----
    tdir = get_trial_dir(run_dir, trial.number)
    ckpt = tdir / "best_model.pt"

    # ----- train -----
    forecaster = OptimizedTorchCNN1DForecast(lookback=lookback, horizon=horizon)
    df = forecaster.load_and_analyze_data(csv_path, target_col=target)

    train_loader, val_loader, test_loader, X_test, y_test, t_test, n_feat = forecaster.prepare_data(
        df,
        feature_cols=None,
        drop_compressor_states=drop_compressor_states,
        include_target_lags=include_target_lags,
        batch_size=batch_size,
        test_batch_size=256,
    )

    forecaster.build_model(n_feat, hparams=hparams_model)

    forecaster.train(
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        patience=patience,
        ckpt_path=str(ckpt),
        use_amp=False,
    )

    # load best + eval
    forecaster.model.load_state_dict(__import__("torch").load(ckpt, map_location="cpu"))
    metrics, y_true, y_pred = forecaster.evaluate(test_loader, X_test, y_test, n_feat)

    # ----- save trial artifacts -----
    save_json(tdir / "hparams.json", {
        "train": {"lookback": lookback, "horizon": horizon, "batch_size": batch_size, "lr": lr, "epochs": epochs, "patience": patience},
        "model": hparams_model,
        "data": {"csv": str(csv_path.resolve()), "target": target, "include_target_lags": include_target_lags, "drop_compressor_states": drop_compressor_states}
    })
    save_json(tdir / "metrics.json", metrics)

    append_csv(run_dir / "trials.csv", {
        "trial": trial.number,
        "rmse": metrics["RMSE"],
        "mae": metrics["MAE"],
        "r2": metrics["R²"],
        "lookback": lookback,
        "horizon": horizon,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "patience": patience,
        "channels": hparams_model["channels"],
        "kernel_size": hparams_model["kernel_size"],
        "dropout": hparams_model["dropout"],
    })

    # minimize RMSE
    return metrics["RMSE"]


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(DATA_PATH_DEFAULT))
    p.add_argument("--target", default="出站压力-连木沁压气站")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--trials", type=int, default=30)
    args = p.parse_args()

    run_dir = make_run_dir(model_tag="cnn1d", target_tag="outlet_p")
    print("Run dir:", run_dir)

    csv_path = Path(args.csv).resolve()

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, run_dir, csv_path, args.target, args.horizon), n_trials=args.trials)

    save_json(run_dir / "best.json", {
        "best_rmse": study.best_value,
        "best_params": study.best_trial.params
    })

    print("Best RMSE:", study.best_value)
    print("Best params:", study.best_trial.params)


if __name__ == "__main__":
    main()
