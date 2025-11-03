# --- SINGLE RUN VERSION ---
import warnings
warnings.filterwarnings("ignore")
import pytorch_lightning as pl

import pandas as pd
import torch
import numpy as np
from datetime import datetime
from functions import error_metrics, run_tft
import os

torch.set_float32_matmul_precision("high")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


# --- TFT function ---



# Load data
df = pd.read_csv("test_final_kz.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"] + "-5", format="%Y-%W-%w")
df = df.sort_values("FECHA")
df = df.rename(columns=lambda x: x.replace(".", "_"))

# --- single model parameters ---
target_col   = "FE"
window_size  = 8
test_size    = 4
d_model      = 34
n_head       = 4
num_layers   = 1
epoch_number = 10
lr           = 1e-05
batch_size   = 16
dropout      = 0.5
grad_clip    = 0.2
seed         = 1048596
deleted_sample = 0

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
start = datetime.now()
y_true, y_pred, tft, val_loader, training = run_tft(
    tmp, target_col, window_size, test_size, grad_clip,
    d_model, n_head, num_layers, epoch_number, lr, batch_size, seed, train_cut
)
mae, mape, mse, rmse, r2 = error_metrics(y_true, y_pred)
runtime = (datetime.now() - start).total_seconds()

print(f"MAE={mae:.3f}, MAPE={mape:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}, Time={runtime:.1f}s")

# Quick plots
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

# indices/values
hist_idx = tmp.time_idx.iloc[train_cut - window_size + 1: train_cut + 1].to_numpy()
hist_vals = tmp[target_col].iloc[train_cut - window_size + 1: train_cut + 1].to_numpy()
future_idx = tmp.time_idx.iloc[train_cut + 1: train_cut + 1 + len(y_true)].to_numpy()
future_vals = y_true
pred_vals = y_pred

# 1) full ground-truth line (continuous)
full_x = np.concatenate([hist_idx, future_idx])
full_y = np.concatenate([hist_vals, future_vals])

plt.figure()
plt.plot(full_x, full_y, label="Actual", linewidth=2)

# # 2) forecast only on future x
plt.plot(future_idx, pred_vals, label="Predicted")
#

plt.title("History + Forecast")
plt.xlabel("Weeks");
plt.ylabel(target_col)
plt.xticks(full_x)
plt.legend();
plt.show()


# #second option
# plt.figure()
# # history (black)
# plt.plot(hist_idx, hist_vals, color="black", label="History", linewidth=2)
# # actual future (blue)
# plt.plot(future_idx, future_vals, color="blue", label="Actual Future", linewidth=2)
# # predicted (orange)
# plt.plot(future_idx, pred_vals, color="orange", label="Predicted", linewidth=2)
#
# plt.title("History + Forecast")
# plt.xlabel("Weeks")
# plt.ylabel(target_col)
# plt.xticks(full_x)
# plt.legend()
# plt.show()




# plot attentions:


# after training
raw_pred, x, *_ = tft.predict(val_loader, return_x=True, mode="raw")
interpret = tft.interpret_output(raw_pred, reduction="none")
A = np.asarray(interpret["attention"])           # attention

print("attention shape:", A.shape)

# Average to "lag importance" (length = enc_len)
if A.ndim == 4:          # [B, dec, enc, heads]
    attn_mean_lag = A.mean(axis=(0,1,3))
elif A.ndim == 3:        # [B, dec, enc]
    attn_mean_lag = A.mean(axis=(0,1))
elif A.ndim == 2:        # [dec, enc]
    attn_mean_lag = A.mean(axis=0)
else:
    raise ValueError(f"Unexpected attention ndim={A.ndim}")




# heatmap for a single example
if A.ndim == 4:
    heat = A[0].mean(axis=-1)          # avg heads -> [dec, enc]
elif A.ndim == 3:
    heat = A[0]                        # [dec, enc]
else:
    heat = A                           # [dec, enc]

plt.figure(); plt.imshow(heat, aspect="auto", origin="lower")
plt.colorbar(label="attention"); plt.xlabel("encoder steps"); plt.ylabel("decoder step")
plt.title("TFT attention"); plt.show()

# if single-step forecast, also show line over lags
if heat.shape[0] == 1:
    plt.figure(); plt.plot(attn_mean_lag)
    plt.xlabel("encoder step (old→new)"); plt.ylabel("avg attention")
    plt.title("Average attention over encoder lags"); plt.show()
