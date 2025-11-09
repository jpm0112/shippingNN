import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from functions import error_metrics, run_transformer

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = torch.device("cpu")

print("Using device:", device)

import random

SEED = 1048596
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Leer y preparar datos
df = pd.read_csv("chile_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"] + "-5", format="%Y-%W-%w")
df = df.sort_values('FECHA')

# Parámetros
target_col = 'FE'

# good parameters for container prediction
window_size = 48
test_size = 24
batch_size = 16
d_model = 256
n_head = 2
num_layers = 2
epoch_number = 500
lr = 0.0001

real, preds = run_transformer(df, target_col, window_size, test_size, batch_size, d_model, n_head, num_layers, epoch_number, lr, device)

print('')
print("Error Metrics:")
error_metrics(real, preds)

plt.figure(figsize=(12, 6))
plt.plot(real, marker="o", label="Real")
plt.plot(preds, marker="o", label="Prediction")
plt.title("Prediction (original scale)")
plt.xlabel("Weeks")
plt.ylabel(target_col)
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("plots/z_transformers_prediction.png", dpi=200)



