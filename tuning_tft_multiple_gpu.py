import os
import warnings
import subprocess
from datetime import datetime
from pathlib import Path
from multiprocessing import Process, Queue, set_start_method
import gc

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import csv

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
# UTILS
# ============================================================
def get_gpu_count() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        stderr=subprocess.STDOUT,
    ).decode(errors="ignore")
    names = [line.strip() for line in out.splitlines() if line.strip()]
    return len(names)


# ============================================================
# TRIAL SUBPROCESS (runs ONE trial only)
# ============================================================
def trial_worker(q, params, df, target_col, prediction_size):
    try:
        import torch
        from functions import run_darts_tft_with_for

        tmp = df.copy()
        deleted_sample = int(params["test_size"])
        if deleted_sample > 0:
            tmp = tmp.iloc[: -deleted_sample * number_test_sets]

        mae, mape, mse, rmse, r2, epochs_ran, sd = run_darts_tft_with_for(
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

        q.put((mae, mape, mse, rmse, r2, epochs_ran, sd))

    except Exception as e:
        q.put(e)

    finally:
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except:
            pass


# ============================================================
# WORKER: ONE GPU
# ============================================================
def run_targets_on_gpu(gpu_id: int, targets):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    torch.cuda.set_device(0)

    from lightning import seed_everything
    from ax.service.ax_client import AxClient
    from ax.service.utils.instantiation import ObjectiveProperties

    print(f"GPU {gpu_id} started | {len(targets)} targets")

    seed_everything(seed, workers=True)

    df = pd.read_csv("weekly_chile_data.csv")
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df = df.sort_values("FECHA")

    results_dir = Path("asax_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    for target_col, prediction_size in targets:

        existing = list(results_dir.glob(f"tft_trials_{country}_{target_col}_*_{prediction_size}.csv"))
        if existing:
            print(f"GPU {gpu_id} | SKIP {target_col}_H{prediction_size} (found {existing[0].name})")
            continue

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        csv_path = results_dir / f"tft_trials_{country}_{target_col}_{timestamp}_{prediction_size}.csv"
        print(f"GPU {gpu_id} | Writing to: {csv_path.name}")
        csv_file = csv_path.open("w", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "trial_index", "test_size", "window_size", "hidden_size",
                "lstm_layers", "num_attention_heads", "dropout", "batch_size",
                "lr", "epochs", "grad_clip", "mae", "mape", "mse", "rmse",
                "r2", "sd", "runtime_s", "started_at", "epochs_ran",
            ],
        )
        csv_writer.writeheader()

        ax = AxClient()
        ax.create_experiment(
            name=f"tft_{target_col}_H{prediction_size}_gpu{gpu_id}",
            parameters=[
                {"name": "test_size", "type": "choice", "values": [int(prediction_size)], "value_type": "int"},
                {"name": "window_size", "type": "range", "bounds": [8, 32], "value_type": "int"},
                {"name": "hidden_size", "type": "choice", "values": [32, 64, 128], "value_type": "int"},
                {"name": "lstm_layers", "type": "range", "bounds": [1, 4], "value_type": "int"},
                {"name": "num_attention_heads", "type": "choice", "values": [2, 4], "value_type": "int"},
                {"name": "dropout", "type": "range", "bounds": [0.1, 0.6]},
                {"name": "batch_size", "type": "choice", "values": [32, 64, 128]},
                {"name": "lr", "type": "range", "bounds": [1e-5, 1e-3], "log_scale": True},
                {"name": "grad_clip", "type": "range", "bounds": [0.5, 3.0], "value_type": "float"},
                {"name": "epochs", "type": "fixed", "value": 2000},
            ],
            objectives={metric: ObjectiveProperties(minimize=True)},
        )

        for i in range(iterations):

            print(f"GPU {gpu_id} | {target_col}_H{prediction_size} | Trial {i + 1}/{iterations}")

            params, trial_index = ax.get_next_trial()
            started_at = datetime.now()

            q = Queue()
            p = Process(
                target=trial_worker,
                args=(q, params, df, target_col, prediction_size),
            )

            p.start()
            p.join()

            result = q.get()

            if isinstance(result, Exception):
                print(f"GPU {gpu_id} | {target_col}_H{prediction_size} | Trial {i + 1} FAILED: {result}")
                ax.log_trial_failure(trial_index)
                continue

            mae, mape, mse, rmse, r2, epochs_ran, sd = result
            runtime_s = (datetime.now() - started_at).total_seconds()

            ax.complete_trial(trial_index, raw_data={metric: float(mape)})

            csv_writer.writerow({
                "trial_index": trial_index,
                **params,
                "mae": mae, "mape": mape, "mse": mse, "rmse": rmse,
                "r2": r2, "sd": sd,
                "runtime_s": runtime_s,
                "started_at": started_at.isoformat(timespec="seconds"),
                "epochs_ran": epochs_ran,
            })
            csv_file.flush()

        csv_file.close()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    set_start_method("spawn", force=True)

    target_cols = ["NAW", "NAE", "NE", "SAW", "SAE", "SE", "FE"]
    prediction_sizes = [4, 12]

    n_gpus = min(get_gpu_count(), 2)
    if n_gpus <= 0:
        raise RuntimeError("No GPUs detected.")

    jobs = [(t, p) for t in target_cols for p in prediction_sizes]
    splits = np.array_split(jobs, n_gpus)

    processes = []
    for gpu_id, split_jobs in enumerate(splits):
        split_jobs = list(split_jobs)
        if not split_jobs:
            continue

        p = Process(target=run_targets_on_gpu, args=(gpu_id, split_jobs))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
