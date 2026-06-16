import os
import warnings
from datetime import datetime
from pathlib import Path
import gc

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import csv
import torch
import torch.multiprocessing as mp

# ============================================================
# CONFIG
# ============================================================
number_test_sets = 3
iterations = 100

patience = 50
min_delta = 1e-5
seed = 1048596
metric = "mape"
country = "chile"


# ============================================================
# DIRECT TRIAL
# ============================================================
def trial_worker_direct(params, df, target_col, prediction_size):
    from functions import run_darts_tft_with_for

    deleted_sample = int(params["test_size"])

    tmp = df.iloc[: -deleted_sample * number_test_sets] if deleted_sample > 0 else df

    return run_darts_tft_with_for(
        df=tmp,
        target_col=target_col,
        test_size=int(params["test_size"]),
        window_size=int(params["window_size"]),
        hidden_size=int(params["hidden_size"]),
        lstm_layers=int(params["lstm_layers"]),
        num_attention_heads=int(params["num_attention_heads"]),
        dropout=float(params["dropout"]),
        batch_size=int(params["batch_size"]),
        n_epochs=int(params["epochs"]),
        lr=float(params["lr"]),
        grad_clip=float(params["grad_clip"]),
        patience=patience,
        min_delta=min_delta,
        seed=seed,
        sample_sets=3,
    )


# ============================================================
# ONE GPU = ONE BO SEARCH
# ============================================================
def run_bo(gpu_id, jobs):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from lightning import seed_everything
    from ax.service.ax_client import AxClient
    from ax.service.utils.instantiation import ObjectiveProperties

    torch.cuda.set_device(0)  # important after masking
    seed_everything(seed, workers=True)

    BASE_DIR = Path(__file__).resolve().parent
    results_dir = BASE_DIR / "results_start"
    results_dir.mkdir(exist_ok=True)

    df = pd.read_csv(BASE_DIR / "weekly_chile_data.csv")
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df = df.sort_values("FECHA")

    print(f"\n{'=' * 60}")
    print(f"GPU {gpu_id} running jobs: {jobs}")
    print(f"{'=' * 60}\n")

    for target_col, prediction_size in jobs:

        print(f"\n=== {target_col} | H={prediction_size} | GPU {gpu_id} ===")

        # --------------------------------------------------
        # CSV RESUME
        # --------------------------------------------------

        pattern = f"tft_trials_{country}_{target_col}_*_{prediction_size}.csv"
        existing = sorted(results_dir.glob(pattern))

        if existing:
            csv_path = existing[-1]
            old_df = pd.read_csv(csv_path)
            completed_trials = len(old_df)

            csv_file = csv_path.open("a", newline="")
            print(f"Resuming CSV → {completed_trials} trials")

        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = results_dir / f"tft_trials_{country}_{target_col}_{timestamp}_{prediction_size}.csv"

            old_df = None
            completed_trials = 0

            csv_file = csv_path.open("w", newline="")
            print("Starting NEW experiment")

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "trial_index", "test_size", "window_size", "hidden_size",
                "lstm_layers", "num_attention_heads", "dropout", "batch_size",
                "lr", "epochs", "grad_clip", "mae", "mape", "mse", "rmse",
                "r2", "sd", "runtime_s", "started_at", "epochs_ran",
            ],
        )

        if completed_trials == 0:
            writer.writeheader()

        # --------------------------------------------------
        # AX STATE
        # --------------------------------------------------

        ax_state = results_dir / f"ax_state_{country}_{target_col}_{prediction_size}.json"

        if ax_state.exists():

            print("Loading AX state...")
            ax = AxClient.load_from_json_file(ax_state)

        else:

            print("Creating new AX experiment...")

            ax = AxClient()

            ax.create_experiment(
                name=f"tft_{target_col}_H{prediction_size}",
                parameters=[
                    {"name": "test_size", "type": "choice", "values": [int(prediction_size)], "value_type": "int"},
                    {"name": "window_size", "type": "range", "bounds": [8, 52], "value_type": "int"},
                    {"name": "hidden_size", "type": "choice", "values": [32, 64, 128, 256], "value_type": "int"},
                    {"name": "lstm_layers", "type": "range", "bounds": [1, 4], "value_type": "int"},
                    {"name": "num_attention_heads", "type": "choice", "values": [2, 4], "value_type": "int"},
                    {"name": "dropout", "type": "range", "bounds": [0.1, 0.6]},
                    {"name": "batch_size", "type": "choice", "values": [32, 64, 128]},
                    {"name": "lr", "type": "range", "bounds": [1e-5, 1e-3], "log_scale": True},
                    {"name": "grad_clip", "type": "range", "bounds": [0.5, 3.0]},
                    {"name": "epochs", "type": "fixed", "value": 2000},
                ],
                objectives={metric: ObjectiveProperties(minimize=True)},
            )

            # rebuild if csv exists
            if old_df is not None:

                print("Rebuilding AX from CSV...")

                for _, row in old_df.iterrows():
                    params = {
                        "test_size": int(row.test_size),
                        "window_size": int(row.window_size),
                        "hidden_size": int(row.hidden_size),
                        "lstm_layers": int(row.lstm_layers),
                        "num_attention_heads": int(row.num_attention_heads),
                        "dropout": float(row.dropout),
                        "batch_size": int(row.batch_size),
                        "lr": float(row.lr),
                        "grad_clip": float(row.grad_clip),
                        "epochs": 2000,
                    }

                    _, t_idx = ax.attach_trial(parameters=params)

                    ax.complete_trial(
                        trial_index=t_idx,
                        raw_data={metric: float(row.mape)}
                    )

            ax.save_to_json_file(ax_state)

        remaining = iterations - len(ax.experiment.trials)

        if remaining <= 0:
            print("BO already finished.")
            csv_file.close()
            continue

        print(f"Running {remaining} remaining trials...\n")

        # --------------------------------------------------
        # BO LOOP
        # --------------------------------------------------

        for _ in range(remaining):

            params, trial_index = ax.get_next_trial()

            print(f"GPU {gpu_id} | Trial {trial_index}")

            started_at = datetime.now()

            try:
                mae, mape, mse, rmse, r2, epochs_ran, sd = trial_worker_direct(
                    params, df, target_col, prediction_size
                )

            except Exception as e:
                print("Trial failed:", e)
                ax.log_trial_failure(trial_index)
                ax.save_to_json_file(ax_state)
                continue

            runtime_s = (datetime.now() - started_at).total_seconds()

            ax.complete_trial(trial_index, raw_data={metric: float(mape)})
            ax.save_to_json_file(ax_state)

            writer.writerow({
                "trial_index": trial_index,
                **params,
                "mae": mae,
                "mape": mape,
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "sd": sd,
                "runtime_s": runtime_s,
                "started_at": started_at.isoformat(timespec="seconds"),
                "epochs_ran": epochs_ran,
            })

            csv_file.flush()

            torch.cuda.empty_cache()
            gc.collect()

        csv_file.close()


# ============================================================
# ENTRY
# ============================================================
if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    available_gpus = min(torch.cuda.device_count(), 2)

    if available_gpus == 0:
        raise RuntimeError("No GPUs detected.")

    jobs = [
        ("FE", 12),
        ("FE", 4),
        ("SE", 12),
    ]

    splits = np.array_split(jobs, available_gpus)

    processes = []

    for gpu_id, split in enumerate(splits):

        split_jobs = list(split)

        if not split_jobs:
            continue

        p = mp.Process(
            target=run_bo,
            args=(gpu_id, split_jobs),
        )

        p.start()
        processes.append(p)

    for p in processes:
        p.join()
