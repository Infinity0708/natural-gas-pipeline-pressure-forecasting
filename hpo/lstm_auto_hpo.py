from __future__ import annotations
import optuna
from pathlib import Path

from hpo.hpo_utils import make_run_dir, get_trial_dir, save_json, append_csv

from src.lstm_model_training import OptimizedTorchLSTMForecast

DATA_PATH = Path(__file__).resolve().parent.parent / "raw" / "raw.csv"

def objective(trial: optuna.Trial, run_dir: Path) -> float:
    # ===== 搜索空间（你可以随时调范围）=====
    seq_len = trial.suggest_int("sequence_length", 20, 120, step=10)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    epochs = trial.suggest_int("epochs", 20, 60, step=10)
    patience = trial.suggest_int("patience", 5, 10)

    hparams_model = {
        "hidden1": trial.suggest_categorical("hidden1", [32, 64, 128, 256]),
        "hidden2": trial.suggest_categorical("hidden2", [16, 32, 64, 128]),
        "dropout1": trial.suggest_float("dropout1", 0.0, 0.4),
        "dropout2": trial.suggest_float("dropout2", 0.0, 0.4),
        "dropout3": trial.suggest_float("dropout3", 0.0, 0.4),
        "fc": trial.suggest_categorical("fc", [8, 16, 32, 64]),
    }

    # ===== 每个 trial 单独目录（绝不覆盖）=====
    tdir = get_trial_dir(run_dir, trial.number)

    # ===== 训练流程 =====
    forecaster = OptimizedTorchLSTMForecast(sequence_length=seq_len)
    df = forecaster.load_and_analyze_data(DATA_PATH)

    train_loader, val_loader, test_loader, X_test, y_test, n_feat = forecaster.prepare_data(
        df,
        batch_size=batch_size,
        test_batch_size=256
    )

    forecaster.build_model(input_size=n_feat, hparams=hparams_model)
    ckpt = tdir / "best_model.pt"
    forecaster.train(train_loader, val_loader, epochs=epochs, lr=lr, patience=patience, ckpt_path=str(ckpt))

    metrics, y_true, y_pred = forecaster.evaluate(test_loader, X_test, y_test, n_feat)

    # ===== 保存 trial 产物 =====
    save_json(tdir / "hparams.json", {
        "train": {"sequence_length": seq_len, "batch_size": batch_size, "lr": lr, "epochs": epochs, "patience": patience},
        "model": hparams_model
    })
    save_json(tdir / "metrics.json", metrics)

    # 你如果想：把 trial 的图也存进 trial_dir
    # forecaster.plot_training()  # 注意这会弹窗，建议你改成只保存不 show
    # forecaster.plot_test_comparison(y_true, y_pred)

    append_csv(run_dir / "trials.csv", {
        "trial": trial.number,
        "rmse": metrics["RMSE"],
        "mae": metrics["MAE"],
        "r2": metrics["R²"],
        "sequence_length": seq_len,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "patience": patience,
        "hidden1": hparams_model["hidden1"],
        "hidden2": hparams_model["hidden2"],
        "dropout1": hparams_model["dropout1"],
        "dropout2": hparams_model["dropout2"],
        "dropout3": hparams_model["dropout3"],
        "fc": hparams_model["fc"],
    })

    # 目标：最小 RMSE
    return metrics["RMSE"]

def main():
    run_dir = make_run_dir(model_tag="lstm", target_tag="w2_inlet_p")
    print("Run dir:", run_dir)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, run_dir), n_trials=30)

    save_json(run_dir / "best.json", {
        "best_rmse": study.best_value,
        "best_params": study.best_trial.params
    })

    print("Best RMSE:", study.best_value)
    print("Best params:", study.best_trial.params)

if __name__ == "__main__":
    main()
