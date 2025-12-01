import torch
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
import pandas as pd
from pathlib import Path
import csv
from functions import error_metrics, run_transformer
from datetime import datetime

# --- CSV setup ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
csv_path = results_dir / f"transformer_trials_{timestamp}.csv"
csv_file = csv_path.open("w", newline="")
csv_writer = csv.DictWriter(csv_file, fieldnames=[
    "trial_index",
    "window_size", "batch_size", "d_model", "num_layers", "epoch_number", "lr", "n_head",
    "mae", "mape", "mse", "rmse", "r2", "runtime_s", "started_at"
])
csv_writer.writeheader()

deleted_sample = 0
test_size    = 24
target_col   = "FE"
# Load data
df = pd.read_csv("chile_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"] + "-5", format="%Y-%W-%w")
df = df.sort_values("FECHA")
df = df.rename(columns=lambda x: x.replace(".", "_"))

# Preprocess
tmp = df.copy()
tmp = tmp.sort_values("FECHA").copy()
tmp["series"] = "kz"
tmp["time_idx"] = tmp.groupby("series").cumcount()
tmp["dow"] = tmp["FECHA"].dt.weekday.astype(int)
tmp["month"] = tmp["FECHA"].dt.month.astype(int)
if deleted_sample > 0:
    tmp = tmp.iloc[:-deleted_sample]

train_cut = tmp["time_idx"].max() - test_size
print("BayesOpt HP Example")

client = AxClient()
print("Creating experiment...")
client.create_experiment(
    name="transformer_experiment",
    parameters=[
        {"name": "window_size", "type": "range", "bounds": [12, 168], "value_type": "int"},
        {"name": "epoch_number", "type": "range", "bounds": [50, 200], "value_type": "int"},
        {"name": "batch_size", "type": "choice", "values": [16, 32, 64, 128]},
        {"name": "lr", "type": "range", "bounds": [1e-5, 3e-3], "log_scale": True},
        {"name": "d_model", "type": "choice", "values": [64, 96, 128, 160, 192, 256, 320, 384, 512]},
        {"name": "n_head", "type": "choice", "values": [1, 2, 4, 8]},
        {"name": "num_layers", "type": "range", "bounds": [1, 6], "value_type": "int"},
    ],
    objectives={"mape": ObjectiveProperties(minimize=True)}
)

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
iterations = 5
for _ in range(iterations):
    params, trial_index = client.get_next_trial()
    started_at = datetime.now()
    y_true, y_pred = run_transformer(
        df = tmp,
        target_col = target_col,
        window_size = int(params["window_size"]),
        test_size = test_size,
        batch_size = int(params["batch_size"]),
        d_model = int(params["d_model"]),
        n_head = int(params["n_head"]),
        num_layers= int(params["num_layers"]),
        epoch_number = int(params["epoch_number"]),
        lr = float(params["lr"]),
        device = device,
        seed = 1048596,
    )
    mae, mape, mse, rmse, r2 = error_metrics(y_true, y_pred)
    runtime_s = (datetime.now() - started_at).total_seconds()

    # Report only the objective metric to Ax
    client.complete_trial(trial_index=trial_index, raw_data={"mape": float(mape)})

    # Persist one row per trial to CSV
    csv_writer.writerow({
        "trial_index": trial_index,
        "window_size": int(params["window_size"]),
        "batch_size": int(params["batch_size"]),
        "d_model": int(params["d_model"]),
        "num_layers": int(params["num_layers"]),
        "epoch_number": int(params["epoch_number"]),
        "lr": float(params["lr"]),
        "n_head": int(params["n_head"]),
        "mae": mae, "mape": mape, "mse": mse, "rmse": rmse, "r2": r2,
        "runtime_s": runtime_s,
        "started_at": started_at.isoformat(timespec="seconds"),
    })
    csv_file.flush()

csv_file.close()
print(f"Wrote per-trial results to {csv_path.resolve()}")
