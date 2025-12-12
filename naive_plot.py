import pandas as pd
from functions import error_metrics, run_darts_tft
import matplotlib
matplotlib.use("Agg")   # <-- no GUI, safe in PyCharm
import matplotlib.pyplot as plt
from lightning.pytorch import  seed_everything
import torch
from datetime import datetime
import numpy as np
from darts.explainability.tft_explainer import TFTExplainer
import shap



# ==== 1. Load your data ====
df = pd.read_csv("weekly_chile_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"])
df = df.sort_values("FECHA").reset_index(drop=True)


target_col = "FE"

test_size = 24 # number of weeks to forecast
window_size = 52
hidden_size = 128
lstm_layers = 2
num_attention_heads = 2
dropout = 0.195661864
batch_size = 32
lr = 0.000920696018319848
n_epochs = 2000
grad_clip = 1


patience = 100
min_delta = 1e-5

seed = 1048596

# ==== 2. Run Darts TFT ====
# without delete sample
tmp = df.copy().sort_values("FECHA")
# === Naive baselines (use history before test window) ===
hist = tmp[target_col].values
hist_train = hist[:-test_size]          # all data before test window
naive_mean = hist_train.mean()         # mean of history
naive_last = hist_train[-1]            # last observed value


start = datetime.now()
y_true, y_pred, out = run_darts_tft(tmp,target_col,test_size,window_size,hidden_size,lstm_layers,num_attention_heads,dropout,
        batch_size,n_epochs,lr,grad_clip,patience=patience,min_delta=min_delta,seed=seed)
mae, mape, mse, rmse, r2 = error_metrics(y_true, y_pred)
runtime = (datetime.now() - start).total_seconds()
print("prediction errors")
print(f"MAE={mae:.3f}, MAPE={mape:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}, Time={runtime:.1f}s")

print(y_true)
print(y_pred)

# --- History window used by TFT ---
full_series = tmp[target_col].values
history_window = full_series[-(window_size + test_size):-test_size]  # length = window_size

hist_x = np.arange(len(history_window))
test_x = np.arange(len(history_window), len(history_window) + len(y_true))

# === Build naive prediction vectors for the test horizon ===
y_naive_mean = np.repeat(naive_mean, len(y_true))
y_naive_last = np.repeat(naive_last, len(y_true))
hist_train = full_series[:-test_size]
# === Naive global mean ===
naive_mean_global = hist_train.mean()
# === Naive last ===
naive_last = hist_train[-1]
# === Naive window mean (mean of last `window_size` points before forecast) ===
naive_mean_window = hist_train[-window_size:].mean()
y_naive_global_mean  = np.repeat(naive_mean_global, len(y_true))
y_naive_last         = np.repeat(naive_last, len(y_true))
y_naive_window_mean  = np.repeat(naive_mean_window, len(y_true))




# === Error metrics for naive baselines ===
mae_m, mape_m, mse_m, rmse_m, r2_m = error_metrics(y_true, y_naive_mean)
mae_l, mape_l, mse_l, rmse_l, r2_l = error_metrics(y_true, y_naive_last)

print("[Naive mean]   MAE={:.3f}, MAPE={:.3f}, RMSE={:.3f}, R2={:.3f}"
      .format(mae_m, mape_m, rmse_m, r2_m))
print("[Naive last]   MAE={:.3f}, MAPE={:.3f}, RMSE={:.3f}, R2={:.3f}"
      .format(mae_l, mape_l, rmse_l, r2_l))

print("\nNaive baselines:")
print("Global mean:",  error_metrics(y_true, y_naive_global_mean))
print("Last value:",    error_metrics(y_true, y_naive_last))
print("Window mean:",   error_metrics(y_true, y_naive_window_mean))



plt.figure(figsize=(12, 6))
plt.plot(y_true, marker="o", label="Real")
plt.plot(y_pred, marker="o", label="Prediction")
plt.plot(y_naive_last, linestyle="--", label="Naive last")
plt.plot(y_naive_mean, linestyle="--", label="Naive mean")
plt.title("Prediction (original scale)")
plt.xlabel("Weeks")
plt.ylabel(target_col)
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(f"plots/tft_prediction_with_naive_{target_col}.png", dpi=200)



plt.figure(figsize=(12, 6))

# --- History window ---
plt.plot(hist_x, history_window, "ko-", label=f"History (last {window_size} weeks)")

# --- naive mean window
# plt.plot(test_x, y_naive_window_mean, "--", color="green", label="Naive window mean")
# --- Ground truth (test) ---
plt.plot(test_x, y_true, "o-", color="black", label="Ground truth (test)")

plt.plot(test_x, y_pred_real_lasso, "o-", color="red", label="Lasso")

# --- TFT prediction ---
plt.plot(test_x, y_pred, "o--", color="magenta", label="TFT prediction")

# --- Naive baselines ---
plt.plot(test_x, y_naive_last, "--", color="gray", label="Naive last")
plt.plot(test_x, y_naive_mean, "--", color="orange", label="Naive mean")

plt.title(f"TFT forecast for {target_col}: history window + {test_size}-step horizon")
plt.xlabel("Index")
plt.ylabel(target_col)
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(f"plots/tft_prediction_with_history_{target_col}.png", dpi=200)



results = []

def add_row(name, y_hat):
    mae, mape, mse, rmse, r2 = error_metrics(y_true, y_hat)
    results.append({
        "model": name,
        "MAE": mae,
        "MAPE": mape,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    })

# TFT
add_row("TFT", y_pred)

# Naive baselines
add_row("Naive last",          y_naive_last)
add_row("Naive global mean",   y_naive_global_mean)
add_row("Naive window mean",   y_naive_window_mean)

metrics_df = pd.DataFrame(results)
print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
