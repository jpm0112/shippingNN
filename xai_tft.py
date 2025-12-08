import pandas as pd
from functions import error_metrics, run_darts_tft
import matplotlib.pyplot as plt
from lightning.pytorch import  seed_everything
import torch



# ==== 1. Load your data ====
df = pd.read_csv("weekly_chile_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"])
df = df.sort_values("FECHA").reset_index(drop=True)





target_col = "FE"

test_size = 12 # number of weeks to forecast
window_size = 52
hidden_size = 256
lstm_layers = 1
num_attention_heads = 4
dropout = 0.412859964370727
batch_size = 32
lr = 0.000595802
n_epochs = 2000
grad_clip = 1.0


patience = 100
min_delta = 1e-5

seed = 1048596

# ==== 2. Run Darts TFT ====

deleted_sample = test_size  # delete the test samples from the end
tmp = df.copy().sort_values("FECHA")
if deleted_sample > 0:
        tmp = tmp.iloc[:-deleted_sample]


y_true, y_pred, out = run_darts_tft(tmp,target_col,test_size,window_size,hidden_size,lstm_layers,num_attention_heads,dropout,
        batch_size,n_epochs,lr,grad_clip,patience=patience,min_delta=min_delta,seed=seed)
mae, mape, mse, rmse, r2 = error_metrics(y_true, y_pred)
runtime = (datetime.now() - start).total_seconds()
print("prediction errors")
print(f"MAE={mae:.3f}, MAPE={mape:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}, Time={runtime:.1f}s")

print(y_true)
print(y_pred)

out_list = [model, train, val, scaler_y, scaler_cov, epochs_ran, feature_cols]

model = out[0]
X_train = out[1]
X_test = out[2]
feature_names = model_list[6]

plt.figure(figsize=(12, 6))
plt.plot(y_true, marker="o", label="Real")
plt.plot(y_pred, marker="o", label="Prediction")
plt.title("Prediction (original scale)")
plt.xlabel("Weeks")
plt.ylabel(target_col)
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("plots/z_transformers_prediction.png", dpi=200)


def predict_fn(flat_X):
    # flat_X: [n_samples, T*F]
    arr = flat_X.reshape(-1, T, F)
    with torch.no_grad():
        x_tensor = torch.tensor(arr, dtype=torch.float32).to(device)
        preds = model(x_tensor).cpu().numpy().ravel()  # shape [n_samples]
    return preds

#__________________________________

#LIME

from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Prepare X for LIME ---
N, T, F = X_test.shape
X_lime = X_test.reshape(N, T * F)

# --- 2. Original feature names (length F, in the SAME order as X_test last dim) ---
# Replace this list with the actual order you used when creating X_train/X_test
original_feature_names = df.columns.to_list()

# Build readable names: <col>_t<k>
feature_names = [
    f"{original_feature_names[j]}_t{T - 1 - t}"  # or just t{t} if you prefer
    for t in range(T)
    for j in range(F)
]

# --- 3. LIME explainer ---
explainer = LimeTabularExplainer(
    X_lime,
    feature_names=feature_names,
    mode="regression"
)

# Quick check on a single instance (optional)
# idx = -1
# x0 = X_lime[idx]
# exp = explainer.explain_instance(
#     x0,
#     predict_fn,
#     num_features=15
# )
# print(exp.as_list())
# print(len(exp.as_list()))

# --- 4. Aggregate importance over all test points ---
n_features = X_lime.shape[1]
agg_pos = np.zeros(n_features)
agg_neg = np.zeros(n_features)
agg_abs = np.zeros(n_features)

for i in range(N):
    exp_i = explainer.explain_instance(
        X_lime[i],
        predict_fn,
        num_features=n_features
    )
    class_id = list(exp_i.local_exp.keys())[0]  # regression → single key
    for feat_idx, weight in exp_i.local_exp[class_id]:
        agg_pos[feat_idx] += max(weight, 0.0)
        agg_neg[feat_idx] += min(weight, 0.0)
        agg_abs[feat_idx] += abs(weight)

lime_summary = pd.DataFrame({
    "feature": feature_names,
    "importance_abs": agg_abs,
    "importance_pos": agg_pos,
    "importance_neg": agg_neg,
}).sort_values("importance_abs", ascending=False)

print(lime_summary.head(30))

# --- 5. Plot with readable names ---
top = lime_summary.head(20)
signed = top["importance_pos"] + top["importance_neg"]  # negatives included

plt.figure(figsize=(8, 6))
colors = ["green" if v >= 0 else "red" for v in signed]

plt.barh(top["feature"], signed, color=colors)
plt.gca().invert_yaxis()
plt.xlabel("Signed contribution")
plt.title("LIME Signed Contributions")
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# signed contribution per flat feature
lime_summary["signed"] = lime_summary["importance_pos"] + lime_summary["importance_neg"]

# remove the time suffix: "MEAN_FLETE_per_TEU_ALEMANIA_t0" → "MEAN_FLETE_per_TEU_ALEMANIA"
lime_summary["base_feature"] = lime_summary["feature"].str.rsplit("_t", n=1).str[0]

# aggregate over all time steps
agg_feat = (
    lime_summary
    .groupby("base_feature", as_index=False)
    .agg(
        signed=("signed", "sum"),
        importance_abs=("importance_abs", "sum"),
        importance_pos=("importance_pos", "sum"),
        importance_neg=("importance_neg", "sum"),
    )
    .sort_values("importance_abs", ascending=False)
)

# plot top 20 base features
top = agg_feat.head(20)
colors = ["green" if v >= 0 else "red" for v in top["signed"]]

plt.figure(figsize=(8, 6))
plt.barh(top["base_feature"], top["signed"], color=colors)
plt.gca().invert_yaxis()
plt.xlabel("Signed contribution (aggregated over time)")
plt.title("LIME Signed Contributions by Original Feature")
plt.tight_layout()
plt.show()

import numpy as np

# 1) Signed contribution
lime_summary["signed"] = (
        lime_summary["importance_pos"] + lime_summary["importance_neg"]
)

# 2) Remove time suffix:  *_t0, *_t1, ...
lime_summary["base_feature"] = (
    lime_summary["feature"].str.rsplit("_t", n=1).str[0]
)


# 3) Define groups: MEAN_FLETE*, SUM_TEU*, *_price, *_weekly_pct_change, *_volume
def feature_group(name: str) -> str:
    if name.startswith("MEAN_FLETE"):
        return "MEAN_FLETE*"
    if name.startswith("SUM_TEU"):
        return "SUM_TEU*"
    if name.endswith("_price"):
        return "price"
    if name.endswith("_weekly_pct_change"):
        return "pct_change"
    if name.endswith("_volume"):
        return "volume"
    return "other"


lime_summary["group"] = lime_summary["base_feature"].apply(feature_group)

# 4) Aggregate by group
agg_group = (
    lime_summary
    .groupby("group", as_index=False)
    .agg(
        signed=("signed", "sum"),
        importance_abs=("importance_abs", "sum"),
    )
    .sort_values("importance_abs", ascending=False)
)

# 5) Plot grouped signed contributions
topg = agg_group  # or .head(k) if you want only top k groups
colors = ["green" if v >= 0 else "red" for v in topg["signed"]]

plt.figure(figsize=(8, 5))
plt.barh(topg["group"], topg["signed"], color=colors)
plt.gca().invert_yaxis()
plt.xlabel("Signed contribution (sum per group)")
plt.title("LIME Signed Contributions by Group")
plt.tight_layout()
plt.show()

# 1) Signed contribution per flat feature (already did, but just in case)
lime_summary["signed"] = lime_summary["importance_pos"] + lime_summary["importance_neg"]

# 2) Extract time step from "..._t<k>"
lime_summary["time"] = lime_summary["feature"].str.extract(r"_t(\d+)$").astype(int)

time_agg = (
    lime_summary
    .groupby("time", as_index=False)
    .agg(
        signed=("signed", "sum"),
        importance_abs=("importance_abs", "sum"),
    )
)

plt.figure(figsize=(8, 4))
plt.bar(time_agg["time"], time_agg["signed"])
plt.xlabel("Time step")
plt.ylabel("Signed contribution (sum over features)")
plt.title("LIME – Contribution by Time Step")
plt.tight_layout()
plt.show()






#___________________________________________________________________________









# ---------- TIMESHAP (fixed) ----------
from timeshap.explainer import (
    local_pruning,
    local_event,
    local_feat,
    local_cell_level,
)


# TimeSHAP expects: f(x: [N, T, F]) -> [N, 1]
def ts_model(x_np):
    x_np = np.asarray(x_np, dtype=np.float32)
    with torch.no_grad():
        x_tensor = torch.tensor(x_np, dtype=torch.float32).to(device)
        preds = model(x_tensor).cpu().numpy().reshape(-1, 1)  # [N, 1]
    return preds


# choose background (subset of training windows)
bg_size = min(100, X_train.shape[0])
background_data = X_train[:bg_size]  # [B, T, F]

# simple baseline = mean window over background
baseline = background_data.mean(axis=0)  # [T, F]

# instance to explain (pick any)
instance = X_test[0:1]  # [1, T, F]

# TimeSHAP parameters

event_dict = {"rs": 42, "nsamples": 32000}
feature_dict = {"rs": 42, "nsamples": 32000}
pruning_dict = {"tol": 0.025}

entity_uuid = 0
entity_col = "entity"
# 1) Pruning
coal_plot_data, coal_prun_idx = local_pruning(
    ts_model,
    instance,
    pruning_dict,
    baseline
)

# 2) Event importance
event_res = local_event(
    ts_model,
    instance,
    event_dict,
    entity_uuid,
    entity_col,
    baseline,
    coal_prun_idx,
)

# 3) Feature importance
feat_res = local_feat(
    ts_model,
    instance,
    feature_dict,
    entity_uuid,
    entity_col,
    baseline,
    coal_prun_idx,
)

print(coal_prun_idx)
print(event_res.head())
print(feat_res.head())

import copy
model_cpu = copy.deepcopy(model).to("cpu").eval()  # freeze + move to CPU

# ---------- SHAP interactions (fixed) ----------
import shap
import numpy as np

# Fix deprecated np.int for SHAP compatibility
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool

# flatten sequences
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)


def model_predict(flat_X):
    # flat_X: [n_samples, T*F]
    flat_X = np.asarray(flat_X, dtype=np.float32)
    n_samples = flat_X.shape[0]
    arr = flat_X.reshape(n_samples, window_size, -1)  # [N, T, F]
    with torch.no_grad():
        x_tensor = torch.tensor(arr, dtype=torch.float32)  # stays on CPU
        preds = model_cpu(x_tensor).numpy().ravel()
    return preds


bg_size_shap = min(50, X_train_flat.shape[0])
X_background = X_train_flat[:bg_size_shap]

# subset of test points to explain
X_sample = X_test_flat[:10]

explainer = shap.KernelExplainer(model_predict, X_background)
# explainer = shap.SamplingExplainer(model_predict, X_background)
shap_values = explainer.shap_values(X_sample)  # [N, T*F]

# --- approximate interaction structure for one feature ---
# pick a feature index, e.g. the most important from LIME or 0 for testing
feat_idx = 0
interaction_indices = shap.utils.approximate_interactions(
    feat_idx, shap_values, X_sample
)

print("Top interacting flat feature indices with", feat_idx, ":", interaction_indices[:10])

# Optional: dependence plot for strongest interaction
shap.dependence_plot(
    feat_idx,
    shap_values,
    X_sample,
    interaction_index=interaction_indices[0],
)











#___________________________________________________

# Integrated Gradients

#___________________________________________________


from captum.attr import IntegratedGradients

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

ig = IntegratedGradients(model)
attr = ig.attribute(X_test_tensor, target=0, n_steps=64)
attr = attr.detach().cpu().numpy()

global_feature_importance = np.mean(np.abs(attr), axis=(0, 1))  # over window & samples
window_importance = np.mean(np.abs(attr), axis=(0, 2))  # over features & samples




plt.figure(figsize=(8, 4))
plt.bar(range(len(global_feature_importance)), global_feature_importance)
plt.xlabel("Feature index")
plt.ylabel("Importance")
plt.title("Global Feature Importance (IG)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(window_importance)
plt.xlabel("Time step")
plt.ylabel("Importance")
plt.title("Window Importance (IG)")
plt.tight_layout()
plt.show()

 # same order as model input last dim
plt.bar(feature_names, global_feature_importance)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

len(feature_names)
print(len(feature_names), global_feature_importance.shape[0])

# Integrated Gradients
ig = IntegratedGradients(model)
input_tensor = X_test_tensor[0:1].clone().detach().requires_grad_(True)
attr, delta = ig.attribute(input_tensor, target=0, return_convergence_delta=True)

# Plot attribution heatmap
attr = attr.squeeze().detach().cpu().numpy()

# Attribution heatmap with feature names
plt.figure(figsize=(12, 8))
plt.imshow(attr.T, aspect='auto', cmap='bwr')
plt.colorbar(label='Attribution Score')
plt.title("Integrated Gradients - Sample 0")
plt.xlabel("Time Step")
plt.ylabel("Feature")

# Map y-axis to actual feature names
plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names, fontsize=8)
plt.tight_layout()
plt.show()

# with the most imporntant features:

# Sum attributions across time for each feature
feature_importance = np.abs(attr).sum(axis=0)

# Get top N most relevant features
top_n = 20
top_indices = np.argsort(feature_importance)[-top_n:]

# Filter the attribution matrix
attr_top = attr[:, top_indices]

# Plot only top features
plt.figure(figsize=(10, 6))
plt.imshow(attr_top.T, aspect='auto', cmap='bwr')
plt.colorbar(label='Attribution Score')
plt.title(f"Integrated Gradients - Top {top_n} Features")
plt.xticks(ticks=np.arange(attr.shape[0]))
plt.xlabel("Time Step")
plt.ylabel("Feature")

# Use actual feature names
selected_labels = [feature_names[i] for i in top_indices]
plt.yticks(ticks=np.arange(top_n), labels=selected_labels, fontsize=8)
plt.xticks(ticks=np.arange(attr.shape[0]))
plt.tight_layout()
plt.show()
