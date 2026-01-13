import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import csv
from datetime import datetime
from pathlib import Path

from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler

from functions import error_metrics, run_sarima, run_sarima_with_for  # same one you use elsewhere


# ============================================================
#  LOAD DATA
# ============================================================
initial_test_size = 24
iterations = 100
number_test_sets = 3

df = pd.read_csv("weekly_chile_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"])
df = df.sort_values("FECHA").reset_index(drop=True)

target_cols = ["FE", "NAE", "NAW", "NE", "SE", "SAW", "SAE"]

# If you have weekly data and expect yearly seasonality, m=52 is typical


for target_col in target_cols:

    metric = "mape"
    country = "chile"
    seed = 1048596

    # ============================================================
    #  RESULT CSV
    # ============================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / f"sarima_trials_{country}_{target_col}_{timestamp}_{initial_test_size}.csv"
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "trial_index",
        "test_size",
        "p","d","q",
        "P","D","Q","m",
        "mae","mape","mse","rmse","r2",
        "sd",
        "runtime_s",
        "started_at",
    ])
    csv_writer.writeheader()

    # ============================================================
    #  AX SEARCH SPACE
    # ============================================================
    ax = AxClient()
    ax.create_experiment(
        name=f"sarima_experiment_{target_col}",
        parameters=[
            {"name": "test_size", "type": "choice", "values": [initial_test_size], "value_type": "int"},

            # non-seasonal
            {"name": "p", "type": "range", "bounds": [0, 4], "value_type": "int"},
            {"name": "d", "type": "choice", "values": [0, 1, 2], "value_type": "int"},
            {"name": "q", "type": "range", "bounds": [0, 4], "value_type": "int"},

            # seasonal
            {"name": "P", "type": "range", "bounds": [0, 2], "value_type": "int"},
            {"name": "D", "type": "choice", "values": [0, 1], "value_type": "int"},
            {"name": "Q", "type": "range", "bounds": [0, 2], "value_type": "int"},
            {"name": "m", "type": "choice", "values": [52], "value_type": "int"},
        ],
        objectives={metric: ObjectiveProperties(minimize=True)},
    )

    # ============================================================
    #  BAYES OPT LOOP
    # ============================================================


    for i in range(iterations):
        print(f"\n=== Trial {i + 1}/{iterations} ===")
        params, trial_index = ax.get_next_trial()
        started_at = datetime.now()

        try:

            tmp = df.copy().sort_values("FECHA")
            deleted_sample = int(params["test_size"])
            if deleted_sample > 0:
                tmp = tmp.iloc[:-deleted_sample * number_test_sets]
            mae, mape, mse, rmse, r2, epochs_ran, sd = run_sarima_with_for(
                df=tmp,
                target_col=target_col,
                test_size=int(params["test_size"]),
                p=int(params["p"]), d=int(params["d"]), q=int(params["q"]),
                P=int(params["P"]), D=int(params["D"]), Q=int(params["Q"]),
                m=int(params["m"]),
                seed=seed,
                n_runs=3
            )

            if (mape is None) or (not np.isfinite(mape)):
                raise ValueError(f"Non-finite objective mape={mape}")

            runtime_s = (datetime.now() - started_at).total_seconds()

            ax.complete_trial(trial_index, raw_data={metric: float(mape)})

            print("____________________________________________________________")
            print(f"Trial {trial_index} results: MAE={mae:.4f}, MAPE={mape:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, SD={sd:.4f}")
            print("Parameters:", params)

            csv_writer.writerow({
                "trial_index": trial_index,
                "test_size": params["test_size"],
                "p": params["p"], "d": params["d"], "q": params["q"],
                "P": params["P"], "D": params["D"], "Q": params["Q"], "m": params["m"],
                "mae": mae, "mape": mape, "mse": mse, "rmse": rmse, "r2": r2,
                "sd": sd,
                "runtime_s": runtime_s,
                "started_at": started_at.isoformat(timespec="seconds"),
            })
            csv_file.flush()

        except Exception as e:
            print("Error in trial:", e)
            ax.log_trial_failure(trial_index)

    csv_file.close()
    print("\nSaved:", csv_path)
