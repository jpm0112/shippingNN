# bayesopt_tft_ax.py
import warnings

warnings.filterwarnings("ignore")

from lightning import seed_everything
import numpy as np

from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch, csv

from functions import run_tft_tftorch, error_metrics

# ===== Data prep =====
deleted_sample = 0
test_size = 4
target_col = "FE"
metric = "mape"  # objective to minimize
minimize = True
country = "chile"

df = pd.read_csv("weekly_chile_data.csv")
tmp = df.copy().sort_values("FECHA")
tmp["series"] = "chile"  # specific for TFT

if deleted_sample > 0:
    tmp = tmp.iloc[:-deleted_sample]



# ===== CSV =====
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
csv_path = results_dir / f"tft_trials_{country}_{timestamp}.csv"
csv_file = csv_path.open("w", newline="")
csv_writer = csv.DictWriter(
    csv_file,
    fieldnames=[
        "trial_index", "test_size",
        "window_size", "batch_size", "d_model", "n_head", "num_layers",
        "epoch_number", "lr", "grad_clip",
        "mae", "mape", "mse", "rmse", "r2",
        "runtime_s", "started_at",
    ],
)
csv_writer.writeheader()

# ===== Ax setup =====
ax = AxClient()
ax.create_experiment(
    name="tft_experiment",
    parameters=[
        {"name": "window_size", "type": "choice", "values": [24,48], "value_type": "int"},
        {"name": "test_size", "type": "choice", "values": [4, 8, 12]},
        {"name": "d_model", "type": "choice", "values": [256, 512]},
        {"name": "n_head", "type": "choice", "values": [4, 8]},
        {"name": "num_layers", "type": "range", "bounds": [1, 3], "value_type": "int"},
        {"name": "epoch_number", "type": "range", "bounds": [300, 1500], "value_type": "int"},
        {"name": "batch_size", "type": "choice", "values": [8, 16, 32]},
        {"name": "lr", "type": "range", "bounds": [1e-5, 1e-2], "log_scale": True},
        {"name": "grad_clip", "type": "range", "bounds": [0.01, 1.0], "log_scale": True},
    ],
    objectives={metric: ObjectiveProperties(minimize=minimize)},
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
iterations = 500

for i in range(iterations):
    print("=== Trial %d ===" % (i + 1))
    print("________________________________")
    params, trial_index = ax.get_next_trial()
    started_at = datetime.now()

    d_model = int(params["d_model"])
    n_head = int(params["n_head"])
    if d_model % n_head != 0:
        ax.log_trial_failure(trial_index, metadata={"reason": "d_model % n_head != 0"})
        continue

    try:
        seed = 1048596
        seed_everything(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # assumes weekly_chile_data.csv already has time_idx

        # === Train/eval TFT ===
        y_true, y_pred, model, train_loader, val_loader, train_ds = run_tft_tftorch(
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
            seed=seed,
            train_cut=tmp["time_idx"].max() - int(params["test_size"])
        )

        print(f"Trial {trial_index} completed.")
        print("window_size ", params["window_size"])
        print("test_size ", params["test_size"])
        print("batch_size ", params["batch_size"])
        print("d_model ", d_model)
        print("n_head ", n_head)
        print("num_layers ", params["num_layers"])
        print("epoch_number ", params["epoch_number"])
        print("lr ", params["lr"])
        print("grad_clip ", params["grad_clip"])

        mae, mape, mse, rmse, r2 = error_metrics(y_true, y_pred)
        runtime_s = (datetime.now() - started_at).total_seconds()

        ax.complete_trial(trial_index=trial_index, raw_data={metric: float(mape)})

        csv_writer.writerow({
            "trial_index": trial_index,
            "test_size": int(params["test_size"]),
            "window_size": int(params["window_size"]),
            "batch_size": int(params["batch_size"]),
            "d_model": d_model,
            "n_head": n_head,
            "num_layers": int(params["num_layers"]),
            "epoch_number": int(params["epoch_number"]),
            "lr": float(params["lr"]),
            "grad_clip": float(params["grad_clip"]),
            "mae": mae,
            "mape": mape,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "runtime_s": runtime_s,
            "started_at": started_at.isoformat(timespec="seconds"),
        })
        csv_file.flush()

    except Exception as e:
        ax.log_trial_failure(trial_index, metadata={"exception": str(e)})
        print("Trial failed:", e)

csv_file.close()
print(f"Wrote per-trial results to {csv_path.resolve()}")
