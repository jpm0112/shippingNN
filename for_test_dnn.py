import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from functions import *
from datetime import datetime
import random
import os

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Leer y preparar datos
df = pd.read_csv("test_daily.csv")
df['FECHA'] = pd.to_datetime(df['FECHA'])
df = df.sort_values('FECHA')

# Hiperparámetros
window_sizes = [30, 60, 120, 360]
test_sizes = [30]
hidden_sizes_list = [[32, 32], [32, 64], [64, 64], [64, 124]]
nums_layers = [2]
epoch_numbers = [500, 1000]
target_cols = ['MEAN_FLETE_POR_BULTO']
lrs = [0.01, 0.001]
seeds = [1048596]
deleted_samples = [0]
batch_size = 16

# Resultados
results_df = pd.DataFrame(columns=[
    "seed", "deleted_samples", "window_size", "test_size", "hidden_size", "num_layers", "epoch_number", "Target", "LR",
    "MAE", "MSE", "RMSE", "r2"
])

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for seed in seeds:
    for window_size in window_sizes:
        for test_size in test_sizes:
            for hidden_sizes in hidden_sizes_list:
                for num_layers in nums_layers:
                    for epoch_number in epoch_numbers:
                        for target_col in target_cols:
                            for lr in lrs:
                                for deleted_sample in deleted_samples:

                                    print(f"Running with seed={seed}, window_size={window_size}, test_size={test_size}, "
                                          f"hidden_sizes={hidden_sizes}, num_layers={num_layers}, epoch_number={epoch_number}, "
                                          f"target_col={target_col}, lr={lr}, deleted_sample={deleted_sample}")

                                    # Set seeds
                                    random.seed(seed)
                                    np.random.seed(seed)
                                    torch.manual_seed(seed)
                                    torch.cuda.manual_seed_all(seed)
                                    torch.backends.cudnn.deterministic = True
                                    torch.backends.cudnn.benchmark = False

                                    feature_cols = [col for col in df.columns if col not in ['FECHA', target_col]]

                                    # Split train/test
                                    temp_df = df.copy()
                                    train_df = temp_df[:-test_size]
                                    test_df = temp_df[-(test_size + window_size):]

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

                                    # Flatten for DNN
                                    X_train = X_train.reshape(X_train.shape[0], -1)
                                    X_test = X_test.reshape(X_test.shape[0], -1)

                                    # Tensores
                                    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
                                    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

                                    # DataLoader
                                    dataset = TensorDataset(X_tensor, y_tensor)
                                    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

                                    # DNN Model
                                    class DNNForecast(nn.Module):
                                        def __init__(self, input_size, hidden_sizes):
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

                                    model = DNNForecast(input_size=X_train.shape[1], hidden_sizes=hidden_sizes).to(device)

                                    criterion = nn.MSELoss()
                                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

                                    # Entrenamiento
                                    model.train()
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

                                    preds = preds_scaled
                                    real = np.array(y_test)

                                    print("\nError Metrics:")
                                    mae, mse, rmse, r2 = error_metrics(real, preds)
                                    print("")

                                    results_df.loc[len(results_df)] = [
                                        seed, deleted_sample, window_size, test_size, hidden_sizes, num_layers, epoch_number, target_col, lr,
                                        mae, mse, rmse, r2
                                    ]
                                    results_df.to_csv(os.path.join("results", f"model_results_dnn_{timestamp}.csv"),index=False)
