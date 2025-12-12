# --- SINGLE RUN VERSION ---
import warnings

warnings.filterwarnings("ignore")
import pytorch_lightning as pl

import pandas as pd
import torch
import numpy as np
from datetime import datetime
from functions import error_metrics, run_transformer, get_attention_maps
import os

torch.set_float32_matmul_precision("high")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Load data
df = pd.read_csv("weekly_chile_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"] )
df = df.sort_values("FECHA")
df = df.rename(columns=lambda x: x.replace(".", "_"))

# --- single model parameters ---
target_col = "FE"
window_size = 48
test_size = 24
d_model = 128 * 2
n_head = 4 * 2
num_layers = 2  # lstm layers
epoch_number = 20
lr = 1e-4
batch_size = 32
dropout = 0.2
grad_clip = 0.2
seed = 1048596
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

# get the number of indices for training ( np.int64(213) for example)
train_cut = tmp["time_idx"].max() - test_size

start = datetime.now()
y_true, y_pred, list = run_transformer(df, target_col, window_size, test_size, batch_size, d_model,
                                                            n_head, num_layers, epoch_number, lr,
                                                            device, seed, optimizer_type='adam', weight_decay=1e-4)
mae, mape, mse, rmse, r2 = error_metrics(y_true, y_pred)
runtime = (datetime.now() - start).total_seconds()
print("prediction errors")
print(f"MAE={mae:.3f}, MAPE={mape:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}, Time={runtime:.1f}s")

print(y_true)
print(y_pred)

model = list[0]
sample = torch.tensor(list[1], dtype=torch.float32).to(device)

attn_maps = get_attention_maps(model, sample)
# ---- VISUALIZE ----
import matplotlib.pyplot as plt

layer_idx = 0
head_idx = 0

attn = attn_maps[layer_idx]  # shape: [batch, n_heads, T, T]
attn = attn[0, head_idx].numpy()  # pick first batch, first head

plt.imshow(attn, aspect="auto")
plt.title(f"Attention - Layer {layer_idx}, Head {head_idx}")
plt.colorbar()
plt.show()
