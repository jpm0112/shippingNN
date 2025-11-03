from ax import Client, RangeParameterConfig, ChoiceParameterConfig
import pandas as pd
from pathlib import Path
import csv
from functions import error_metrics, run_tft
from datetime import datetime

# --- CSV setup ---
csv_path = Path("tft_trials.csv")
write_header = not csv_path.exists()
csv_file = csv_path.open("a", newline="")
csv_writer = csv.DictWriter(csv_file, fieldnames=[
    "trial_index",
    "LR","epochs","gradclip","attention_heads","hidden_size",
    "mae","mape","mse","rmse","r2","runtime_s","started_at"
])
if write_header:
    csv_writer.writeheader()

deleted_sample = 0
test_size    = 4
target_col   = "FE"
# Load data
df = pd.read_csv("test_final_kz.csv")
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
client = Client()
print("Creating experiment...")
client.configure_experiment(
    name="tft_experiment",
    parameters=[
        RangeParameterConfig(
            name="LR",
            bounds=(0.00001, 0.1),
            parameter_type="float",
        ),
        RangeParameterConfig(
            name="epochs",
            bounds=(3, 5),
            parameter_type="int",
        ),
        RangeParameterConfig(
            name="gradclip",
            bounds=(0.01, 1.0),
            parameter_type="float",
        ),
        ChoiceParameterConfig(
            name="attention_heads",
            values=[1,2,4,8],
            parameter_type="int",
        ),
        RangeParameterConfig(
            name="hidden_size",
            bounds=(8, 128),
            parameter_type="int",
        ),
    ],
)
client.configure_optimization(objective="-1 * mape")

for _ in range(100):
    # Use higher value of `max_trials` to run trials in parallel.
    for trial_index, parameters in client.get_next_trials(max_trials=1).items():
        started_at = datetime.now()
        y_true, y_pred, _, _, _ = run_tft(
            data = tmp,
            target_col = target_col,
            window_size = 8,
            test_size = test_size,
            grad_clip = parameters["gradclip"],
            d_model = parameters["hidden_size"],
            n_head = parameters["attention_heads"],
            num_layers = 2,
            epoch_number = parameters["epochs"],
            lr = parameters["LR"],
            batch_size = 64,
            seed = 1048596,
            train_cut = train_cut,
        )
        mae, mape, mse, rmse, r2 = error_metrics(y_true, y_pred)
        runtime_s = (datetime.now() - started_at).total_seconds()
        client.complete_trial(
            trial_index=trial_index,
            raw_data={
                # metrics the client can reference in your objective expression
                "mae": mae,
                "mape": mape,
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "runtime_s": runtime_s,
                # include params too (optional, but handy for dashboards/export)
                "LR": parameters["LR"],
                "epochs": parameters["epochs"],
                "gradclip": parameters["gradclip"],
                "attention_heads": parameters["attention_heads"],
                "hidden_size": parameters["hidden_size"],
            },
        )
        # 2) Persist one row per trial to CSV
        csv_writer.writerow({
            "trial_index": trial_index,
            "LR": parameters["LR"],
            "epochs": parameters["epochs"],
            "gradclip": parameters["gradclip"],
            "attention_heads": parameters["attention_heads"],
            "hidden_size": parameters["hidden_size"],
            "mae": mae, "mape": mape, "mse": mse, "rmse": rmse, "r2": r2,
            "runtime_s": runtime_s,
            "started_at": started_at.isoformat(timespec="seconds"),
        })
        csv_file.flush()  # ensure it’s written to disk as the run proceeds
csv_file.close()
print(f"Wrote per-trial results to {csv_path.resolve()}")
