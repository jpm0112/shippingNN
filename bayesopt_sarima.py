from ax import Client, RangeParameterConfig, ChoiceParameterConfig
import pandas as pd
from pathlib import Path
import csv
from functions import error_metrics, run_sarima
from datetime import datetime

# --- CSV setup ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = Path("results") / f"sarima_trials_{timestamp}.csv"
write_header = not csv_path.exists()
csv_file = csv_path.open("a", newline="")
csv_writer = csv.DictWriter(csv_file, fieldnames=[
    "trial_index",
    "p", "d", "q", "P", "D", "Q", "m",
    "mae", "mape", "mse", "rmse", "r2", "runtime_s", "started_at"
])
if write_header:
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
client = Client()
print("Creating experiment...")
client.configure_experiment(
    name="sarima_experiment",
    parameters=[
        RangeParameterConfig(name="p", bounds=(0, 48), parameter_type="int"),
        RangeParameterConfig(name="d", bounds=(0, 2), parameter_type="int"),
        RangeParameterConfig(name="q", bounds=(0, 5), parameter_type="int"),
        RangeParameterConfig(name="P", bounds=(0, 2), parameter_type="int"),
        RangeParameterConfig(name="D", bounds=(0, 2), parameter_type="int"),
        RangeParameterConfig(name="Q", bounds=(0, 2), parameter_type="int"),
        ChoiceParameterConfig(name="m", values=[7, 12, 24], parameter_type="int", is_ordered=True),
    ],
)
client.configure_optimization(objective="-mape")
iterations = 20
for _ in range(iterations):
    # Use higher value of `max_trials` to run trials in parallel.
    for trial_index, parameters in client.get_next_trials(max_trials=1).items():
        started_at = datetime.now()
        y_true, y_pred = run_sarima(
            df = tmp,
            target_col = target_col,
            test_size = test_size,
            p = parameters["p"],
            d = parameters["d"],
            q = parameters["q"],
            P =parameters["P"],
            D = parameters["D"],
            Q = parameters["Q"],
            m = parameters["m"],
            seed = 1048596,
        )
        mae, mape, mse, rmse, r2 = error_metrics(y_true, y_pred)
        runtime_s = (datetime.now() - started_at).total_seconds()
        client.complete_trial(trial_index=trial_index, raw_data = {
        "mae": float(mae),
        "mape": float(mape),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "runtime_s": float(runtime_s),
        })
        # 2) Persist one row per trial to CSV
        csv_writer.writerow({
            "trial_index": trial_index,
            "p": parameters["p"],
            "d": parameters["d"],
            "q": parameters["q"],
            "P": parameters["P"],
            "D": parameters["D"],
            "Q": parameters["Q"],
            "m": parameters["m"],
            "mae": mae, "mape": mape, "mse": mse, "rmse": rmse, "r2": r2,
            "runtime_s": runtime_s,
            "started_at": started_at.isoformat(timespec="seconds"),
        })
        csv_file.flush()
csv_file.close()


print(f"Wrote per-trial results to {csv_path.resolve()}")
