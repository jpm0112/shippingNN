# bayesopt_tft_ax.py
import warnings

warnings.filterwarnings("ignore")

from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch, csv

from functions import run_tft, error_metrics  # must match your signature

# ===== Data prep (same as your single-run) =====
deleted_sample = 0
test_size = 24
target_col = "FE"
metric = "mape"  # objective to minimize

df = pd.read_csv("chile_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"] + "-5", format="%Y-%W-%w")
df = df.sort_values("FECHA")
df = df.rename(columns=lambda x: x.replace(".", "_"))

tmp = df.copy().sort_values("FECHA")
tmp["series"] = "kz"
tmp["time_idx"] = tmp.groupby("series").cumcount()
# add calendar features if you like:
# tmp["dow"] = tmp["FECHA"].dt.weekday.astype(int)
# tmp["month"] = tmp["FECHA"].dt.month.astype(int)
if deleted_sample > 0:
    tmp = tmp.iloc[:-deleted_sample]

# training cutoff like your script
train_cut = tmp["time_idx"].max() - test_size

# ===== CSV =====
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = Path("results");
results_dir.mkdir(parents=True, exist_ok=True)
csv_path = results_dir / f"tft_trials_{timestamp}.csv"
csv_file = csv_path.open("w", newline="")
csv_writer = csv.DictWriter(csv_file, fieldnames=[
    "trial_index",
    "window_size", "batch_size", "d_model", "n_head", "num_layers",
    "epoch_number", "lr", "dropout", "grad_clip",
    "mae", "mape", "mse", "rmse", "r2", "runtime_s", "started_at"
])
csv_writer.writeheader()

# ===== Ax setup =====
ax = AxClient()
ax.create_experiment(
    name="tft_experiment",
    parameters=[
        # keep window near your current value; widen if desired
        {"name": "window_size", "type": "range", "bounds": [42,43], "value_type": "int"},
        # model width and heads; ensure divisibility (checked below)
        {"name": "d_model", "type": "choice", "values": [64, 96, 128, 256, 512]},
        {"name": "n_head", "type": "choice", "values": [2, 4, 6, 8]},
        # LSTM layers inside TFT encoder/decoder in your run_tft
        {"name": "num_layers", "type": "range", "bounds": [1, 20], "value_type": "int"},
        {"name": "dropout", "type": "range", "bounds": [0.0, 0.5]},
        {"name": "grad_clip", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "epoch_number", "type": "range", "bounds": [100, 1000], "value_type": "int"},
        {"name": "batch_size", "type": "choice", "values": [16, 64]},
        {"name": "lr", "type": "range", "bounds": [1e-6, 1e-3], "log_scale": True},
    ],
    objectives={metric: ObjectiveProperties(minimize=True)},
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
iterations = 500  # adjust as you like

for _ in range(iterations):
    params, trial_index = ax.get_next_trial()
    started_at = datetime.now()

    # enforce d_model % n_head == 0 for multihead attention
    d_model = int(params["d_model"])
    n_head = int(params["n_head"])
    if d_model % n_head != 0:
        ax.log_trial_failure(trial_index, metadata={"reason": "d_model % n_head != 0"})
        continue

    try:
        # === Train/eval ===
        y_true, y_pred, *_ = run_tft(
            tmp,
            target_col=target_col,
            window_size=int(params["window_size"]),
            test_size=test_size,
            grad_clip=float(params["grad_clip"]),
            d_model=d_model,
            n_head=n_head,
            num_layers=int(params["num_layers"]),
            epoch_number=int(params["epoch_number"]),
            lr=float(params["lr"]),
            batch_size=int(params["batch_size"]),
            seed=1048596,
            train_cut=int(train_cut),
        )

        mae, mape, mse, rmse, r2 = error_metrics(y_true, y_pred)
        runtime_s = (datetime.now() - started_at).total_seconds()

        # report objective
        ax.complete_trial(trial_index=trial_index, raw_data={metric: float(mape)})

        # persist row
        csv_writer.writerow({
            "trial_index": trial_index,
            "window_size": int(params["window_size"]),
            "batch_size": int(params["batch_size"]),
            "d_model": d_model,
            "n_head": n_head,
            "num_layers": int(params["num_layers"]),
            "epoch_number": int(params["epoch_number"]),
            "lr": float(params["lr"]),
            "dropout": float(params["dropout"]),
            "grad_clip": float(params["grad_clip"]),
            "mae": mae, "mape": mape, "mse": mse, "rmse": rmse, "r2": r2,
            "runtime_s": runtime_s,
            "started_at": started_at.isoformat(timespec="seconds"),
        })
        csv_file.flush()

    except Exception as e:
        ax.log_trial_failure(trial_index, metadata={"exception": str(e)})

csv_file.close()
print(f"Wrote per-trial results to {csv_path.resolve()}")
