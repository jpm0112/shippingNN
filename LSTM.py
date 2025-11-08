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

SEED = 1048596
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Leer y preparar datos
df = pd.read_csv("test_daily.csv")
df['FECHA'] = pd.to_datetime(df['FECHA'])
df = df.sort_values('FECHA')

# Parámetros
target_col = 'TOTAL_TEUS'
# target_col = 'MEAN_FLETE_POR_BULTO'
# target_col = 'hong kong - san  antonio'
feature_cols = [col for col in df.columns if col not in ['FECHA', target_col]]

# good parameters for container prediction
window_size = 120
test_size = 30
batch_size = 16
hidden_size = 64
num_layers = 2
epoch_number = 2000
lr = 0.0001

window_size = 48
test_size = 24
batch_size = 16
hidden_size = 32
num_layers = 2
epoch_number = 1000
lr = 0.01

# Split train/test
train_df = df[:-test_size]
test_df = df[-(test_size + window_size):]

# Escalamiento manual
train_features = train_df[feature_cols].values
train_target = train_df[target_col].values
min_vals = train_features.min(axis=0)
max_vals = train_features.max(axis=0)
target_min = train_target.min()
target_max = train_target.max()

scaled_train = (train_features - min_vals) / (max_vals - min_vals + 1e-8)
scaled_target = (train_target - target_min) / (target_max - target_min + 1e-8)

# Crear ventanas para entrenamiento
X_train, y_train = [], []
for i in range(len(train_df) - window_size):
    X_train.append(scaled_train[i:i + window_size])
    y_train.append(scaled_target[i + window_size])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Preparar test
test_features = test_df[feature_cols].values
test_target = test_df[target_col].values
scaled_test = (test_features - min_vals) / (max_vals - min_vals + 1e-8)
scaled_test_target = (test_target - target_min) / (target_max - target_min + 1e-8)

X_test, y_test = [], []
for i in range(test_size):
    X_test.append(scaled_test[i:i + window_size])
    y_test.append(scaled_test_target[i + window_size])

X_test = np.array(X_test)
y_test = np.array(y_test)

# Tensores
X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

# DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size, shuffle=False)


# LSTM model
class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size=hidden_size, num_layers=num_layers):
        super(LSTMForecast, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


model = LSTMForecast(input_size=X_train.shape[2]).to(device)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-4)

# Entrenamiento
for epoch in range(epoch_number):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# Evaluación
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
    preds_scaled = model(X_test_tensor).squeeze().cpu().numpy()

preds = preds_scaled   * (target_max - target_min + 1e-8) + target_min
real = np.array(y_test)   * (target_max - target_min + 1e-8) + target_min

print('')
print("Error Metrics:")
error_metrics(real, preds)

# Visualización
plt.figure(figsize=(12, 6))
plt.plot(real, label='Real', marker='o')
plt.plot(preds, label='Prediction', marker='o')
plt.title("Last month prediction")
plt.xlabel("Days")
plt.ylabel("Prediction target")
plt.legend()
plt.xticks(ticks=range(0, len(real), max(1, len(real) // 30)))
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Grid lines
plt.show()
