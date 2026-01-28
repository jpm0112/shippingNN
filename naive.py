import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from functions import error_metrics

# ============================================================
# Settings
# ============================================================
country = "chile"
target_cols = ["FE", "NAE", "NAW", "NE", "SE", "SAW", "SAE"]

test_size = 4  # forecast horizon H
sample_sets = 3  # number of rolling test sets (TFT-style)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

CSV_COLS = [
    "trial_index",
    "test_size",
    "window_size",
    "hidden_size",
    "lstm_layers",
    "num_attention_heads",
    "dropout",
    "batch_size",
    "lr",
    "epochs",
    "grad_clip",
    "mae",
    "mape",
    "mse",
    "rmse",
    "r2",
    "sd",
    "runtime_s",
    "started_at",
    "epochs_ran",
]


# ============================================================
# Naive predictor (last observed value)
# ============================================================
def predict_naive_last(df, target_col, test_size):
    df = df.sort_values("FECHA").reset_index(drop=True)

    y_true = df[target_col].iloc[-test_size:].values
    last_val = df[target_col].iloc[-test_size - 1]

    y_pred = np.full(test_size, last_val)
    return y_true, y_pred


# ============================================================
# Evaluation wrapper (matches your TFT-style deletion scheme)
# ============================================================
def eval_with_for(df, method_fn, target_col, test_size, sample_sets=3, seed=1048596):
    mae_values, mape_values, mse_values, rmse_values, r2_values = [], [], [], [], []

    df_sorted = df.sort_values("FECHA").reset_index(drop=True)

    for i in range(sample_sets):
        tmp = df_sorted.copy()

        OFFSET = 0  # skip the last 3 test sets
        deleted_sample = test_size * (OFFSET + i)
        if deleted_sample > 0:
            tmp = tmp.iloc[:-deleted_sample]

        y_true, y_pred = method_fn(tmp, target_col, test_size)

        mae, mape, mse, rmse, r2 = error_metrics(y_true, y_pred)

        mae_values.append(mae)
        mape_values.append(mape)
        mse_values.append(mse)
        rmse_values.append(rmse)
        r2_values.append(r2)

    return (
        float(np.mean(mae_values)),
        float(np.mean(mape_values)),
        float(np.mean(mse_values)),
        float(np.mean(rmse_values)),
        float(np.mean(r2_values)),
        np.nan,  # epochs (not applicable)
        float(np.std(mape_values, ddof=0)),
    )


# ============================================================
# Save one-row CSV
# ============================================================
def write_one_row_csv(method_name, target_col, metrics, started_at, runtime_s):
    mae, mape, mse, rmse, r2, mean_epochs, sd = metrics

    row = {c: np.nan for c in CSV_COLS}
    row.update({
        "trial_index": 0,
        "test_size": test_size,
        "window_size": np.nan,
        "mae": mae,
        "mape": mape,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "sd": sd,
        "runtime_s": runtime_s,
        "started_at": started_at.isoformat(timespec="seconds"),
        "epochs_ran": mean_epochs,
    })

    csv_path = RESULTS_DIR / f"test_naive_last_{country}_{target_col}_{timestamp}_{test_size}.csv"
    pd.DataFrame([row], columns=CSV_COLS).to_csv(csv_path, index=False)
    print("Saved:", csv_path)


# ============================================================
# Run
# ============================================================
df = pd.read_csv("weekly_chile_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"])

for target_col in target_cols:
    print(f"\n=== NAIVE BASELINE | {target_col} ===")

    start = datetime.now()
    metrics = eval_with_for(
        df=df,
        method_fn=predict_naive_last,
        target_col=target_col,
        test_size=test_size,
        sample_sets=sample_sets
    )
    runtime = (datetime.now() - start).total_seconds()

    print("Metrics:", metrics)
    write_one_row_csv("naive_last", target_col, metrics, start, runtime)

print("\nDone.")

print(metrics)