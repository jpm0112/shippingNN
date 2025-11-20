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



# torch.set_float32_matmul_precision("high")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df = pd.read_csv("weekly_chile_data.csv")
tmp = df.copy().sort_values("FECHA")

# --- single model parameters ---
target_col = "FE"
window_size = 24
test_size = 24
d_model = 512
n_head = 4
num_layers = 9  # lstm layers
epoch_number = 397
lr = 0.00166941602502568
batch_size = 16

seed = 1048596
deleted_sample = 0



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

model = model_list[0]
X_test = model_list[1]
X_train = model_list[2]




#__________________________________

#LIME

from lime.lime_tabular import LimeTabularExplainer

# X_test: [N, T, F] → [N, T*F]
N, T, F = X_test.shape
X_lime = X_test.reshape(N, T * F)

# Optional: simple feature names
feature_names = [f"t-{T - 1 - t}_f{j}" for t in range(T) for j in range(F)]


def predict_fn(flat_X):
    # flat_X: [n_samples, T*F]
    arr = flat_X.reshape(-1, T, F)
    with torch.no_grad():
        x_tensor = torch.tensor(arr, dtype=torch.float32).to(device)
        preds = model(x_tensor).cpu().numpy().ravel()  # shape [n_samples]
    return preds


explainer = LimeTabularExplainer(
    X_lime,
    feature_names=feature_names,
    mode="regression"
)

idx = -1  # or any 0..N-1
x0 = X_lime[idx]

exp = explainer.explain_instance(
    x0,
    predict_fn,
    num_features=15  # top 15 "flattened" features
)

print(exp.as_list())

n_features = X_lime.shape[1]
agg_pos = np.zeros(n_features)
agg_neg = np.zeros(n_features)
agg_abs = np.zeros(n_features)

for idx in range(N):
    exp = explainer.explain_instance(
        X_lime[idx],
        predict_fn,
        num_features=n_features  # use all features
    )
    class_id = list(exp.local_exp.keys())[0]  # regression → single key
    for feat_idx, weight in exp.local_exp[class_id]:
        agg_pos[feat_idx] += max(weight, 0)
        agg_neg[feat_idx] += min(weight, 0)
        agg_abs[feat_idx] += abs(weight)

lime_summary = pd.DataFrame({
    "feature": feature_names,
    "importance_abs": agg_abs,
    "importance_pos": agg_pos,
    "importance_neg": agg_neg,
})

lime_summary = lime_summary.sort_values("importance_abs", ascending=False)
print(lime_summary.head(30))





# TIMESHAP

#____________________________________________________

from timeshap.explainer import DeepExplainer

# Background = typical training windows
bg_size = min(100, X_train.shape[0])
background_data = X_train[:bg_size]  # shape: [bg_size, T, F]

explainer = DeepExplainer(model, background_data)
window_importance = explainer.explain_temporal(X_test)
feature_importance = explainer.explain_features(X_test)




# SHAP interactions

import shap

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)


def model_predict(flat_X):
    # flat_X: [n_samples, T*F]
    arr = flat_X.reshape(-1, T, F)
    with torch.no_grad():
        x_tensor = torch.tensor(arr, dtype=torch.float32).to(device)
        preds = model(x_tensor).cpu().numpy().ravel()
    return preds


bg_size_shap = min(50, X_train_flat.shape[0])
X_background = X_train_flat[:bg_size_shap]

# Sample of test points to explain
X_sample = X_test_flat[:100]  # or fewer if needed

explainer = shap.KernelExplainer(model_predict, X_background)
shap_values = explainer.shap_values(X_sample)

# Interaction decomposition
interaction = shap.TreeExplainer(model, feature_perturbation="interventional") \
    .shap_interaction_values(X_sample)





# Integrated Gradients

#___________________________________________________


from captum.attr import IntegratedGradients

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

ig = IntegratedGradients(model)
attr = ig.attribute(X_test_tensor, target=0, n_steps=64)
attr = attr.detach().cpu().numpy()

global_feature_importance = np.mean(np.abs(attr), axis=(0, 1))  # over window & samples
window_importance = np.mean(np.abs(attr), axis=(0, 2))  # over features & samples
