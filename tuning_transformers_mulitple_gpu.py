# ============================================================
#  MULTI-GPU PARALLEL TRANSFORMER BO (PER TARGET)
# ============================================================

import warnings

warnings.filterwarnings("ignore")

import os
import csv
import torch
import numpy as np
import pandas as pd
import multiprocessing as mp

from pathlib import Path
from datetime import datetime
from lightning import seed_everything
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

from functions import (
    run_transformer_with_for,
    clean_gpu,
)

# ============================================================
#  GLOBAL CONFIG
# ============================================================
initial_test_size = 12
number_test_sets = 3
iterations = 100

patience = 200
min_delta = 1e-5
seed = 1048596

target_cols = ["FE", "NAE", "NAW", "NE", "SE", "SAW", "SAE"]

target_cols = [ "SAE","SAE_CHINA"]

# ============================================================
#  LOAD DATA (ONCE)
# ============================================================
df = pd.read_csv("weekly_chile_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"])
df = df.sort_values("FECHA").reset_index(drop=True)


# ============================================================
#  WORKER FUNCTION (ONE TARGET, ONE GPU)
# ============================================================
def run_target(target_col, gpu_id):
    # --- pin process to one GPU ---
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    print(f"[GPU {gpu_id}] Starting target {target_col}")

    metric = "mape"
    country = "chile"


    # ========================================================
    #  CSV SETUP
    # ========================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    csv_path = results_dir / f"new2_transformer_trials_{country}_{target_col}_{timestamp}_{initial_test_size}.csv"
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "trial_index", "test_size", "window_size", "d_model", "n_head", "num_layers",
        "dropout", "batch_size", "lr", "epochs", "optimizer", "weight_decay",
        "mae", "mape", "mse", "rmse", "r2", "sd",
        "runtime_s", "started_at", "epochs_ran",
    ])
    csv_writer.writeheader()

    # ========================================================
    #  AX SEARCH SPACE
    # ========================================================
    ax = AxClient()


    ax.create_experiment(
        name=f"transformer_experiment_{target_col}",
        parameters=[
            {"name": "test_size", "type": "choice", "values": [initial_test_size]},
            {"name": "window_size", "type": "range", "bounds": [2, 5]},
            {"name": "d_model", "type": "choice", "values": [32, 64, 128, 256]},
            {"name": "n_head", "type": "choice", "values": [2, 4, 8]},
            {"name": "num_layers", "type": "range", "bounds": [1, 4]},
            {"name": "dropout", "type": "range", "bounds": [0.0, 0.6]},
            {"name": "batch_size", "type": "choice", "values": [32, 64, 128]},
            {"name": "lr", "type": "range", "bounds": [1e-5, 1e-3], "log_scale": True},
            {"name": "optimizer", "type": "choice", "values": ["adam"]},
            {"name": "weight_decay", "type": "range", "bounds": [1e-6, 1e-3], "log_scale": True},
            {"name": "epochs", "type": "fixed", "value": 2000},
        ],
        objectives={metric: ObjectiveProperties(minimize=True)},
    )

    # ========================================================
    #  BAYESIAN OPT LOOP
    # ========================================================
    for i in range(iterations):
        print(f"[GPU {gpu_id}] {target_col} | Trial {i + 1}/{iterations}")

        params, trial_index = ax.get_next_trial()
        started_at = datetime.now()

        clean_gpu()
        seed_everything(seed)

        tmp = df.copy()
        tmp = tmp.iloc[:-initial_test_size * number_test_sets]

        if target_col == "SAE_CHINA":
            #delete all columns of tmp except SAE and CSI300_volume
            tmp = tmp[["FECHA", "SAE", "csi300_volume"]]




        try:
            target_col = "SAE"
            mae, mape, mse, rmse, r2, epochs_ran, sd = run_transformer_with_for(
                df=tmp,
                target_col=target_col,
                window_size=int(params["window_size"]),
                test_size=int(params["test_size"]),
                batch_size=int(params["batch_size"]),
                d_model=int(params["d_model"]),
                n_head=int(params["n_head"]),
                num_layers=int(params["num_layers"]),
                epoch_number=int(params["epochs"]),
                lr=float(params["lr"]),
                dropout=float(params["dropout"]),
                device=device,
                seed=seed,
                optimizer_type=str(params["optimizer"]),
                weight_decay=float(params["weight_decay"]),
                early_stop=True,
                patience=patience,
                min_delta=min_delta,
                n_runs=3,
            )

            runtime_s = (datetime.now() - started_at).total_seconds()
            ax.complete_trial(trial_index, raw_data={metric: float(mape)})

            csv_writer.writerow({
                "trial_index": trial_index,
                "test_size": params["test_size"],
                "window_size": params["window_size"],
                "d_model": params["d_model"],
                "n_head": params["n_head"],
                "num_layers": params["num_layers"],
                "dropout": params["dropout"],
                "batch_size": params["batch_size"],
                "lr": params["lr"],
                "epochs": params["epochs"],
                "optimizer": params["optimizer"],
                "weight_decay": params["weight_decay"],
                "mae": mae,
                "mape": mape,
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "sd": sd,
                "runtime_s": runtime_s,
                "started_at": started_at.isoformat(timespec="seconds"),
                "epochs_ran": float(epochs_ran),
            })
            csv_file.flush()

        except Exception as e:
            print(f"[GPU {gpu_id}] Error:", e)
            ax.log_trial_failure(trial_index)

    csv_file.close()
    print(f"[GPU {gpu_id}] Finished {target_col}")


# ============================================================
#  MAIN: PARALLEL EXECUTION (2 GPUS)
# ============================================================
if __name__ == "__main__":

    gpus = [0, 1]  # your two GPUs
    processes = []

    for i, target in enumerate(target_cols):
        p = mp.Process(
            target=run_target,
            args=(target, gpus[i % len(gpus)])
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
