# ==========================================
# TRANSFORMER vs NAIVE (FULL SCRIPT)
# ==========================================

import warnings

warnings.filterwarnings("ignore")

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from functions import (
    run_transformer_with_for_xai,
    error_metrics
)

# ==========================================
# SETUP
# ==========================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# LOAD DATA
# ==========================================
df = pd.read_csv("weekly_chile_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"])
df = df.sort_values("FECHA").reset_index(drop=True)

# ==========================================
# CONFIG (EDIT HERE)
# ==========================================
# BEST SAE 4
test_size = 4
target_col = "SAE"
window_size = 32
batch_size = 32
d_model = 128
n_head = 2
num_layers = 4  # lstm layers
epoch_number = 2000
lr = 0.000482691
dropout = 0.517106545
weight_decay = 7.75E-06

# BEST SAE 12
test_size = 12
target_col = "SAE"
window_size = 45
batch_size = 64
d_model = 256
n_head = 8
num_layers = 3  # lstm layers
lr = 0.000496422
dropout = 0.506098107
weight_decay = 1.00E-06

seed = 1048596
patience = 200
min_delta = 1e-5
n_runs = 3

# ==========================================
# RUN TRANSFORMER
# ==========================================
(
    avg_mae,
    avg_mape,
    avg_mse,
    avg_rmse,
    avg_r2,
    avg_epochs_ran,
    sd,
    out,
    y_true,
    y_pred,
) = run_transformer_with_for_xai(
    df,
    target_col,
    window_size,
    test_size,
    batch_size,
    d_model,
    n_head,
    num_layers,
    epoch_number,
    lr,
    dropout,
    device,
    seed,
    optimizer_type="adam",
    weight_decay=weight_decay,
    early_stop=True,
    patience=patience,
    min_delta=min_delta,
    n_runs=n_runs,
)

# ==========================================
# TRANSFORMER METRICS
# ==========================================
mae, mape, mse, rmse, r2 = error_metrics(y_true, y_pred)

# ==========================================
# INPUT WINDOW
# ==========================================
y_hist = df[target_col].values[-(window_size + test_size):-test_size]

# ==========================================
# NAIVE LINEAR FORECAST
# ==========================================

naive_pred = np.repeat(y_hist[-1], test_size)

# ==========================================
# NAIVE METRICS
# ==========================================
naive_mae, naive_mape, naive_mse, naive_rmse, naive_r2 = error_metrics(
    y_true, naive_pred
)

# ==========================================
# PRINT METRICS
# ==========================================
print("\n=== Transformer ===")
print(f"MAE:  {mae:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

print("\n=== Naive (linear) ===")
print(f"MAE:  {naive_mae:.4f}")
print(f"MAPE: {naive_mape:.2f}%")
print(f"RMSE: {naive_rmse:.4f}")
print(f"R²:   {naive_r2:.4f}")

# ==========================================
# X-AXIS
# ==========================================
x_hist = np.arange(len(y_hist))
x_forecast = np.arange(len(y_hist), len(y_hist) + test_size)

# ==========================================
# METRICS TEXT FOR PLOT
# ==========================================
metrics_text = (
    "Transformer\n"
    f"MAPE = {mape:.2f}%\n"
    "Naive (linear)\n"
    f"MAPE = {naive_mape:.2f}%\n"

)

# ==========================================
# PLOT
# ==========================================
plt.figure(figsize=(12, 6))

plt.plot(x_hist, y_hist, marker="o", label="Input window (history)")
plt.plot(x_forecast, y_pred, marker="o", label="Transformer forecast")
plt.plot(x_forecast, y_true, marker="o", label="Real (future)")
plt.plot(
    x_forecast,
    naive_pred,
    marker="o",
    linestyle="--",
    label="Naive linear forecast",
)

plt.axvline(x_hist[-1], color="gray", linestyle=":", label="Forecast start")

plt.text(
    0.02,
    0.98,
    metrics_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
)

plt.title(f"{target_col} – Input window and forecasts (H={test_size})")
plt.xlabel("Time steps")
plt.ylabel(target_col)
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()

plt.savefig(
    f"plots/transformer_vs_naive_{target_col}_{test_size}_{timestamp}.png",
    dpi=200,
)
plt.show()
