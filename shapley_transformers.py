# --- SINGLE RUN VERSION ---
import warnings

warnings.filterwarnings("ignore")
import pytorch_lightning as pl

import pandas as pd
import torch
import numpy as np
from datetime import datetime
from functions import error_metrics, run_transformer
import os


torch.set_float32_matmul_precision("high")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df = pd.read_csv("weekly_chile_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"])
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
y_true, y_pred, model_list = run_transformer(df, target_col, window_size, test_size, batch_size, d_model,
                                             n_head, num_layers, epoch_number, lr,
                                             device, seed, optimizer_type='adam', weight_decay=1e-4)
mae, mape, mse, rmse, r2 = error_metrics(y_true, y_pred)
runtime = (datetime.now() - start).total_seconds()
print("prediction errors")
print(f"MAE={mae:.3f}, MAPE={mape:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}, Time={runtime:.1f}s")

print(y_true)
print(y_pred)

# __________________________________

# SHAPLEY

model = model_list[0]
X_test = model_list[1]
X_train = model_list[2]

import shap

# 1) Get trained model
model = model_list[0].to(device)
model.eval()

# 2) Get the data matrices used by the model
#    -> you already create sliding windows somewhere (inside run_transformer).
#       Suppose you have:
#       X_train: (n_train, window_size, n_features)
#       X_test:  (n_test,  window_size, n_features)

# Use a small background sample for KernelExplainer
background_size = min(100, X_train.shape[0])
X_background = X_train[:background_size]

# Subset of test points to explain
X_explain = X_test[:50]  # or all test points, but this is slower


# 3) Wrap model into a prediction function for SHAP
def model_predict(x_np):
    """
    x_np: numpy array with shape (batch, window_size, n_features)
    returns: numpy array with predictions (batch,)
    """
    x_tensor = torch.from_numpy(x_np).float().to(device)
    with torch.no_grad():
        preds = model(x_tensor).view(-1)
    return preds.cpu().numpy()


# 4) Build SHAP explainer (model-agnostic, works with any PyTorch model)
explainer = shap.KernelExplainer(model_predict, X_background)

# 5) Compute SHAP values for the selected instances
shap_values = explainer.shap_values(X_explain, nsamples=200)

# 6) Some basic plots (run in notebook)
shap.initjs()
# Single prediction (flattened features; good for debugging)
shap.force_plot(explainer.expected_value, shap_values[0].reshape(-1))

# Summary plot by feature (you may want to first reshape/aggregate over time)
shap.summary_plot(shap_values.reshape(X_explain.shape[0], -1),
                  X_explain.reshape(X_explain.shape[0], -1))

