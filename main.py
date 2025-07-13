import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Leer y preparar datos
df = pd.read_csv("test_daily.csv")  # Ajusta la ruta
df['FECHA'] = pd.to_datetime(df['FECHA'])
df = df.sort_values('FECHA')

target_col = 'TOTAL_CONTENEDORES'
feature_cols = [col for col in df.columns if col not in ['FECHA', target_col]]

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[feature_cols])
target = df[target_col].values

# Ventanas deslizantes
window_size = 30
X, y = [], []
for i in range(len(df) - window_size):
    X.append(scaled_features[i:i+window_size])
    y.append(target[i+window_size])

X = np.array(X)
y = np.array(y)

# Tensores
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

# DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Modelo
class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size=640, num_layers=10):
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento
for epoch in range(10):  # Ajusta las épocas
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Predicción
model.eval()
with torch.no_grad():
    predictions = model(X_tensor).squeeze().cpu().numpy()

# Visualización
real = y_tensor.squeeze().cpu().numpy()

plt.figure(figsize=(12, 6))
plt.plot(real, label='Real')
plt.plot(predictions, label='Predicción')
plt.legend()
plt.title("Predicción de TOTAL CONTENEDORES")
plt.xlabel("Días")
plt.ylabel("Contenedores")
plt.show()
