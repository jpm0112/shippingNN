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

seed = 1048596
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Leer y preparar datos
df = pd.read_csv("chile_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"] + "-5", format="%Y-%W-%w")

df = pd.read_csv("weekly_uruguay_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"])
df = df.drop(columns=["WEEK"])

df = df.sort_values('FECHA')

# Parámetros
target_col = 'FE'

# good parameters for container prediction
window_size = 48
test_size = 24
batch_size = 16
d_model = 512
n_head = 8
num_layers = 1
epoch_number = 600

lr = 0.000320022890109655

real, preds = run_transformer(df, target_col, window_size, test_size, batch_size, d_model, n_head, num_layers, epoch_number, lr, device, seed)

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



