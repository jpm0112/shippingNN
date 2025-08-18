import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from functions import *  # Make sure this contains `error_metrics`

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Set seed
SEED = 1048596
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print("Loading data")
# Load data
df = pd.read_csv("test_daily.csv")
df['FECHA'] = pd.to_datetime(df['FECHA'])
df = df.sort_values('FECHA')

target_col = 'TOTAL_TEUS'
feature_cols = [col for col in df.columns if col not in ['FECHA', target_col]]

# Model parameters
window_size = 30
test_size = 30
batch_size = 16
epoch_number = 1000
lr = 0.01

# Split
train_df = df[:-test_size]
test_df = df[-(test_size + window_size):]

# Normalize
train_features = train_df[feature_cols].values
train_target = train_df[target_col].values
min_vals = train_features.min(axis=0)
max_vals = train_features.max(axis=0)
target_min = train_target.min()
target_max = train_target.max()

scaled_train = (train_features - min_vals) / (max_vals - min_vals + 1e-8)
scaled_target = (train_target - target_min) / (target_max - target_min + 1e-8)

X_train, y_train = [], []
for i in range(len(train_df) - window_size):
    X_train.append(scaled_train[i:i + window_size])
    y_train.append(scaled_target[i + window_size])
X_train, y_train = np.array(X_train), np.array(y_train)

test_features = test_df[feature_cols].values
test_target = test_df[target_col].values
scaled_test = (test_features - min_vals) / (max_vals - min_vals + 1e-8)
scaled_test_target = (test_target - target_min) / (target_max - target_min + 1e-8)

X_test, y_test = [], []
for i in range(test_size):
    X_test.append(scaled_test[i:i + window_size])
    y_test.append(scaled_test_target[i + window_size])
X_test, y_test = np.array(X_test), np.array(y_test)

# Flatten inputs for DNN
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Tensors
X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

# DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size, shuffle=False)

# DNN Model
class DNNForecast(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64, 32]):
        super(DNNForecast, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_sizes[1], 1)
        )

    def forward(self, x):
        return self.layers(x)

model = DNNForecast(input_size=X_train.shape[1]).to(device)

# Training
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-4)

print("Running model")
for epoch in range(epoch_number):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    preds_scaled = model(X_test_tensor).squeeze().cpu().numpy()

preds = preds_scaled
real = np.array(y_test)

print("\nError Metrics:")
error_metrics(real, preds)

# Plot predictions
plt.figure(figsize=(12, 6))
plt.plot(real, label='Real', marker='o')
plt.plot(preds, label='Prediction', marker='o')
plt.title("Last month prediction")
plt.xlabel("Days")
plt.ylabel("Prediction target")
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()

# Integrated Gradients
ig = IntegratedGradients(model)
input_tensor = X_test_tensor[0:1].clone().detach().requires_grad_(True)
attr, delta = ig.attribute(input_tensor, target=0, return_convergence_delta=True)

# Attribution heatmap
attr = attr.squeeze().detach().cpu().numpy()
attr = attr.reshape(window_size, len(feature_cols))

plt.figure(figsize=(12, 8))
plt.imshow(attr.T, aspect='auto', cmap='bwr')
plt.colorbar(label='Attribution Score')
plt.title("Integrated Gradients - Sample 0")
plt.xlabel("Time Step")
plt.ylabel("Feature")
plt.yticks(ticks=np.arange(len(feature_cols)), labels=feature_cols, fontsize=8)
plt.tight_layout()
plt.show()

# Top N feature heatmap
feature_importance = np.abs(attr).sum(axis=0)
top_n = 20
top_indices = np.argsort(feature_importance)[-top_n:]
attr_top = attr[:, top_indices]

plt.figure(figsize=(10, 6))
plt.imshow(attr_top.T, aspect='auto', cmap='bwr')
plt.colorbar(label='Attribution Score')
plt.title(f"Integrated Gradients - Top {top_n} Features")
plt.xlabel("Time Step")
plt.ylabel("Feature")
selected_labels = [feature_cols[i] for i in top_indices]
plt.yticks(ticks=np.arange(top_n), labels=selected_labels, fontsize=8)
plt.tight_layout()
plt.show()
