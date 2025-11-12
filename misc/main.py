import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Leer y preparar datos
df = pd.read_csv("test_daily.csv")
df['FECHA'] = pd.to_datetime(df['FECHA'])
df = df.sort_values('FECHA')

target_col = 'TOTAL_CONTENEDORES'
feature_cols = [col for col in df.columns if col not in ['FECHA', target_col]]

features = df[feature_cols].values
target = df[target_col].values
target_min = target.min()
target_max = target.max()
target_scaled = (target - target_min) / (target_max - target_min + 1e-8)

# Min-Max Scaling manual
min_vals = features.min(axis=0)
max_vals = features.max(axis=0)
scaled_features = (features - min_vals) / (max_vals - min_vals + 1e-8)  # avoid division by zero



target_min = target.min()
target_max = target.max()
target_scaled = (target - target_min) / (target_max - target_min + 1e-8)

# define parameters
window_size = 90
batch_size = 32
hidden_size = 48
num_layers = 2
epoch_number = 500
lr = 0.01

# Crear ventanas deslizantes
X, y = [], []
for i in range(len(df) - window_size):
    X.append(scaled_features[i:i + window_size])
    y.append(target_scaled[i + window_size])

X = np.array(X)
y = np.array(y)

# Tensores
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

# DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size, shuffle=False)


# Modelo
class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size = hidden_size, num_layers = num_layers):
        super(LSTMForecast, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


model = LSTMForecast(input_size=X.shape[2]).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr)

# Entrenamiento
for epoch in range(epoch_number):
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# Predicción
model.eval()
with torch.no_grad():
    predictions = model(X_tensor).squeeze().cpu().numpy()

real = y_tensor.squeeze().cpu().numpy()

# Visualización
plt.figure(figsize=(12, 6))
plt.plot(real, label='Real')
plt.plot(predictions, label='Predicción')
plt.legend()
plt.title("Predicción de TOTAL CONTENEDORES")
plt.xlabel("Días")
plt.ylabel("Contenedores")
plt.show()


