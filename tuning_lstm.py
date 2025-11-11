# bayesopt_lstm_ax.py
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch, csv

from functions import run_lstm, error_metrics  # uses your signature

# ==== Data prep (yours) ====
deleted_sample = 0
test_size = 24
target_col = "FE"

df = pd.read_csv("chile_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"] + "-5", format="%Y-%W-%w")
df = df.sort_values("FECHA")
df = df.rename(columns=lambda x: x.replace(".", "_"))

tmp = df.copy().sort_values("FECHA")
tmp["series"] = "kz"
tmp["time_idx"] = tmp.groupby("series").cumcount()
tmp["dow"] = tmp["FECHA"].dt.weekday.astype(int)
tmp["month"] = tmp["FECHA"].dt.month.astype(int)
if deleted_sample > 0:
    tmp = tmp.iloc[:-deleted_sample]

# ==== CSV ====
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = Path("results");
results_dir.mkdir(exist_ok=True)
csv_path = results_dir / f"lstm_trials_{timestamp}.csv"
csv_file = csv_path.open("w", newline="")
csv_writer = csv.DictWriter(csv_file, fieldnames=[
    "trial_index",
    "window_size", "batch_size", "hidden_size", "num_layers", "epoch_number", "lr",
    "mae", "mape", "mse", "rmse", "r2", "runtime_s", "started_at"
])
csv_writer.writeheader()

# ==== Ax setup ====
client = AxClient()
client.create_experiment(
    name="lstm_experiment",
    parameters=[
        {"name": "window_size", "type": "range", "bounds": [12, 168], "value_type": "int"},
        {"name": "lr", "type": "range", "bounds": [1e-5, 1e-1], "log_scale": True},
        {"name": "epoch_number", "type": "range", "bounds": [50, 200], "value_type": "int"},
        {"name": "batch_size", "type": "choice", "values": [16, 32, 64, 128]},
        {"name": "hidden_size", "type": "range", "bounds": [32, 256], "value_type": "int"},
        {"name": "num_layers", "type": "range", "bounds": [1, 5], "value_type": "int"},
    ],
    objectives={"r2": ObjectiveProperties(minimize=False)}

)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
iterations = 5

for _ in range(iterations):
    params, trial_index = client.get_next_trial()
    started_at = datetime.now()

    # === Train/eval using your function ===
    y_true, y_pred = run_lstm(
        df=tmp,
        target_col=target_col,
        window_size=int(params["window_size"]),
        test_size=test_size,
        batch_size=int(params["batch_size"]),
        hidden_size=int(params["hidden_size"]),
        num_layers=int(params["num_layers"]),
        epoch_number=int(params["epoch_number"]),
        lr=float(params["lr"]),
        device=device,
        seed=1048596,
    )

    mae, mape, mse, rmse, r2 = error_metrics(y_true, y_pred)
    runtime_s = (datetime.now() - started_at).total_seconds()

    # Report to Ax (objective is "mape")
    client.complete_trial(trial_index=trial_index, raw_data={"mape": float(mape)})

    # Persist row
    csv_writer.writerow({
        "trial_index": trial_index,
        "window_size": int(params["window_size"]),
        "batch_size": int(params["batch_size"]),
        "hidden_size": int(params["hidden_size"]),
        "num_layers": int(params["num_layers"]),
        "epoch_number": int(params["epoch_number"]),
        "lr": float(params["lr"]),
        "mae": mae, "mape": mape, "mse": mse, "rmse": rmse, "r2": r2,
        "runtime_s": runtime_s,
        "started_at": started_at.isoformat(timespec="seconds"),
    })
    csv_file.flush()

csv_file.close()
print(f"Wrote per-trial results to {csv_path.resolve()}")
