import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from functions import *

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = torch.device("cpu")

print("Using device:", device)

import random

seed = 1048596
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Leer y preparar datos
df = pd.read_csv("weekly_chile_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"])
df = df.sort_values('FECHA')

# Parámetros
target_col = 'FE'


# good parameters for container prediction
window_size = 41
test_size = 26
batch_size = 64
hidden_size = 256
num_layers = 3
epoch_number = 2000
lr = 0.000333205812804923
patience = 100
min_delta = 1e-5

real, preds, out = run_lstm(df, target_col, window_size, test_size, batch_size,
             hidden_size, num_layers, epoch_number, lr,
             device, seed, patience, min_delta)
print('')
print("Error Metrics:")
error_metrics(real, preds)

# Visualización
plt.figure(figsize=(12, 6))
plt.plot(real, marker="o", label="Real")
plt.plot(preds, marker="o", label="Prediction")
plt.title("Prediction (original scale)")
plt.xlabel("Weeks")
plt.ylabel(target_col)
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(f"plots/lstm_prediction_{target_col}_{test_size}.png", dpi=200)
