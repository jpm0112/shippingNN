import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch, csv
from lightning import seed_everything
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from datetime import datetime
from pathlib import Path

from functions import (
    run_lstm_with_for,
    error_metrics,
    clean_gpu,
)

# ============================================================
#  LOAD DATA
# ============================================================
initial_test_size = 26

df = pd.read_csv("weekly_chile_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"])
df = df.sort_values("FECHA")

target_cols = ["FE", "NAE", "NAW", "NE", "SE", "SAW", "SAE"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for target_col in target_cols:

    metric = "mape"
    country = "chile"

    patience = 100
    min_delta = 1e-5
    seed = 1048596

    # ============================================================
    #  RESULT CSV
    # ============================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / f"lstm_trials_{country}_{target_col}_{timestamp}_{initial_test_size}.csv"
    csv_file = csv_path.open("w", newline="")

    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "trial_index",
        "test_size",
        "window_size",
        "hidden_size",
        "num_layers",
        "batch_size",
        "lr",
        "epochs",
        "mae", "mape", "mse", "rmse", "r2",
        "sd",
        "runtime_s",
        "started_at",
        "epochs_ran",
    ])
    csv_writer.writeheader()

    # ============================================================
    #  AX SEARCH SPACE
    # ============================================================
    ax = AxClient()
    ax.create_experiment(
        name=f"lstm_experiment_{target_col}",
        parameters=[
            {"name": "test_size", "type": "choice", "values": [initial_test_size], "value_type": "int"},
            {"name": "window_size", "type": "range", "bounds": [26, 52], "value_type": "int"},
            {"name": "hidden_size", "type": "choice", "values": [32, 64, 128, 256], "value_type": "int"},
            {"name": "num_layers", "type": "range", "bounds": [1, 3], "value_type": "int"},
            {"name": "batch_size", "type": "choice", "values": [32, 64, 128], "value_type": "int"},
            {"name": "lr", "type": "range", "bounds": [1e-5, 1e-3], "log_scale": True},
            {"name": "epochs", "type": "fixed", "value": 2000},
        ],
        objectives={metric: ObjectiveProperties(minimize=True)},
    )

    # ============================================================
    #  BAYES OPT LOOP
    # ============================================================
    iterations = 25

    for i in range(iterations):
        print(f"\n=== Trial {i + 1}/{iterations} ===")
        params, trial_index = ax.get_next_trial()
        started_at = datetime.now()

        clean_gpu()
        seed_everything(seed)

        try:
            tmp = df.copy().sort_values("FECHA")
            deleted_sample = int(params["test_size"])
            if deleted_sample > 0:
                tmp = tmp.iloc[:-deleted_sample]

            mae, mape, mse, rmse, r2, epochs_ran, sd = run_lstm_with_for(
                df=tmp,
                target_col=target_col,
                window_size=int(params["window_size"]),
                test_size=int(params["test_size"]),
                batch_size=int(params["batch_size"]),
                hidden_size=int(params["hidden_size"]),
                num_layers=int(params["num_layers"]),
                epoch_number=int(params["epochs"]),
                lr=float(params["lr"]),
                device=device,
                seed=seed,
                patience=patience,
                min_delta=min_delta,
            )

            runtime_s = (datetime.now() - started_at).total_seconds()

            ax.complete_trial(trial_index, raw_data={metric: float(mape)})

            print("____________________________________________________________")
            print(
                f"Trial {trial_index} results | "
                f"MAE={mae:.4f} MAPE={mape:.4f} RMSE={rmse:.4f} "
                f"R2={r2:.4f} Epochs={epochs_ran}"
            )
            print("Parameters:", params)

            csv_writer.writerow({
                "trial_index": trial_index,
                "test_size": params["test_size"],
                "window_size": params["window_size"],
                "hidden_size": params["hidden_size"],
                "num_layers": params["num_layers"],
                "batch_size": params["batch_size"],
                "lr": params["lr"],
                "epochs": params["epochs"],
                "mae": mae, "mape": mape, "mse": mse, "rmse": rmse, "r2": r2,
                "sd": sd,
                "runtime_s": runtime_s,
                "started_at": started_at.isoformat(timespec="seconds"),
                "epochs_ran": epochs_ran,
            })
            csv_file.flush()

        except Exception as e:
            print("Error in trial:", e)
            ax.log_trial_failure(trial_index)

    csv_file.close()
    print("\nSaved:", csv_path)
