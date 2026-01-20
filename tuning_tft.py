import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch, csv, gc
from lightning import seed_everything
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from datetime import datetime
from pathlib import Path
from functions import run_darts_tft, error_metrics, run_darts_tft_with_for, clean_gpu

# ============================================================
#  LOAD DATA
# ============================================================

prediction_size = 12
number_test_sets = 3



df = pd.read_csv("weekly_chile_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"])
df = df.sort_values("FECHA")

target_cols = ["FE", "NAE","NAW","NE","SE","SAW","SAE"] #I can rerun NAE as it crashed at 97 iterations
# target_cols = ["NAW", "NE", "SE", "SAW", "SAE"]
for target_col in target_cols:


    metric = "mape"
    minimize = True
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

    csv_path = results_dir / f"tft_trials_{country}_{target_col}_{timestamp}_{prediction_size}.csv"
    csv_file = csv_path.open("w", newline="")

    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "trial_index",
        "test_size",  # output_chunk_length
        "window_size", # input_chunk_length
        "hidden_size",
        "lstm_layers",
        "num_attention_heads",
        "dropout",
        "batch_size",
        "lr",
        "epochs",
        "grad_clip",
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
        name="tft_experiment",
        parameters=[
            {"name": "test_size", "type": "choice", "values": [prediction_size], "value_type": "int"},
            {"name": "window_size", "type": "range", "bounds": [8,52], "value_type": "int"},
            {"name": "hidden_size", "type": "choice", "values": [32, 64, 128, 256], "value_type": "int"},
            {"name": "lstm_layers", "type": "range", "bounds": [1, 4], "value_type": "int"},
            {"name": "num_attention_heads", "type": "choice", "values": [2, 4], "value_type": "int"},
            {"name": "dropout", "type": "range", "bounds": [0.1, 0.6],"value_type": "float"},
            {"name": "batch_size", "type": "choice", "values": [32, 64, 128], "value_type": "int"},
            {"name": "lr", "type": "range", "bounds": [1e-5, 1e-3], "log_scale": True},
            {"name": "grad_clip", "type": "range", "bounds": [0.5, 3],"value_type": "float"},
            {"name": "epochs", "type": "fixed", "value": 2000},
        ],

        objectives={metric: ObjectiveProperties(minimize=True)},
    )

    # ============================================================
    #  BAYES OPT LOOP
    # ============================================================
    iterations = 50

    for i in range(iterations):
        print(f"\n=== Trial {i + 1}/{iterations} ===")
        params, trial_index = ax.get_next_trial()
        started_at = datetime.now()

        clean_gpu()
        # Seed for reproducibility

        try:

            tmp = df.copy().sort_values("FECHA")
            deleted_sample = int(params["test_size"])  # delete the test samples from the end
            if deleted_sample > 0:
                tmp = tmp.iloc[:-deleted_sample*number_test_sets]


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
                sample_sets=3
            )
            print("____________________________________________________________")
            print(f"Trial {trial_index} results: MAE={mae:.4f}, MAPE={mape:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, Epochs Ran={epochs_ran}")
            print("Parameters:", params)

            runtime_s = (datetime.now() - started_at).total_seconds()

            ax.complete_trial(trial_index, raw_data={metric: float(mape)})

            # Write CSV
            csv_writer.writerow({
                "trial_index": trial_index,
                "test_size": params["test_size"],
                "window_size": params["window_size"],
                "hidden_size": params["hidden_size"],
                "lstm_layers": params["lstm_layers"],
                "num_attention_heads": params["num_attention_heads"],
                "dropout": params["dropout"],
                "batch_size": params["batch_size"],
                "lr": params["lr"],
                "epochs": params["epochs"],
                "grad_clip": params["grad_clip"],
                "mae": mae, "mape": mape, "mse": mse, "rmse": rmse, "r2": r2,"sd":sd,
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
