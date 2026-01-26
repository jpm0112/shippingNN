# --- SINGLE RUN VERSION ---
import warnings

warnings.filterwarnings("ignore")
import pytorch_lightning as pl

import pandas as pd
import torch
import numpy as np
from datetime import datetime
from functions import error_metrics, run_transformer_xai, get_attention_maps, run_transformer_with_for_xai
from functions import prettify, rename_specific
import os

import matplotlib.pyplot as plt



# torch.set_float32_matmul_precision("high")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df = pd.read_csv("weekly_chile_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"])
tmp = df.copy().sort_values("FECHA")

# BEST FE 4
test_size = 4
target_col = "FE"
window_size = 42
batch_size = 128
d_model = 64
n_head = 2
num_layers = 4  # lstm layers
epoch_number = 2000
lr = 0.000340605
dropout = 0.069862972
weight_decay = 0.000187754


#BEST NAE 4

test_size = 4
target_col = "NAE"
window_size = 51
batch_size = 32
d_model = 64
n_head = 4
num_layers = 4  # lstm layers
epoch_number = 2000
lr = 0.000397956
dropout = 0.223329688
weight_decay = 0.000103901

#BEST NAW 4
test_size = 4
target_col = "NAW"
window_size = 41
batch_size = 64
d_model = 128
n_head = 4
num_layers = 4  # lstm layers
epoch_number = 2000
lr = 6.47E-05
dropout = 0.121422183
weight_decay = 6.10E-06



# BEST NE 4
test_size = 4
target_col = "NE"
window_size = 30
batch_size = 128
d_model = 32
n_head = 8
num_layers = 4  # lstm layers
epoch_number = 2000
lr = 9.28E-05
dropout = 0.483863325
weight_decay = 0.000197146

# BEST SE 4
test_size = 4
target_col = "SE"
window_size = 36
batch_size = 32
d_model = 256
n_head = 4
num_layers = 4  # lstm layers
epoch_number = 2000
lr = 0.000555638
dropout = 0.312582405
weight_decay = 0.000179268

# BEST SAW 4
test_size = 4
target_col = "SAW"
window_size = 32
batch_size = 128
d_model = 32
n_head = 2
num_layers = 2  # lstm layers
epoch_number = 2000
lr = 0.000147759
dropout = 0.424560519
weight_decay = 8.66E-05

# BEST SAE 4
test_size = 4
target_col = "SAE"
window_size = 32
batch_size = 32
d_model = 128
n_head = 2
num_layers = 4 # lstm layers
epoch_number = 2000
lr = 0.000482691
dropout = 0.517106545
weight_decay = 7.75E-06



# BEST FE 12
test_size = 12
target_col = "FE"
window_size = 28
batch_size = 32
d_model = 128
n_head = 2
num_layers = 4  # lstm layers
epoch_number = 2000
lr = 0.001
dropout = 0.332641601
weight_decay = 0.001


# BEST NAE 12
test_size = 12
target_col = "NAE"
window_size = 26
batch_size = 32
d_model = 64
n_head = 2
num_layers = 4  # lstm layers
lr = 0.000187546
dropout = 0.273919743
weight_decay = 7.55E-06


# BEST NAW 12
test_size = 12
target_col = "NAW"
window_size = 51
batch_size = 128
d_model = 256
n_head = 8
num_layers = 4  # lstm layers
lr = 0.00029803
dropout = 0.165152428
weight_decay = 5.62E-06


# BEST NE 12
test_size = 12
target_col = "NE"
window_size = 47
batch_size = 32
d_model = 64
n_head = 2
num_layers = 4  # lstm layers
lr = 0.0004815
dropout = 0.318221159
weight_decay = 4.00E-05


#BEST SE 12
test_size = 12
target_col = "SE"
window_size = 50
batch_size = 32
d_model = 64
n_head = 8
num_layers = 4  # lstm layers
lr = 0.000427606
dropout = 0.444859662
weight_decay = 1.50E-05

# BEST SAW 12
test_size = 12
target_col = "SAW"
window_size = 49
batch_size = 128
d_model = 256
n_head = 4
num_layers = 4  # lstm layers
lr = 0.000551659
dropout = 0.426466938
weight_decay = 6.90E-05

# BEST SAE 12
test_size = 12
target_col = "SAE"
window_size = 45
batch_size = 64
d_model = 256
n_head = 8
num_layers = 3  # lstm layers
lr = 0.000496422
dropout = 0.506098107
weight_decay = 1.00E-06

# ==============================

# General parameters
seed = 1048596
patience = 200
min_delta = 1e-5
epoch_number = 2000

tmp = df.copy().sort_values("FECHA")



avg_mae, avg_mape, avg_mse, avg_rmse, avg_r2, avg_epochs_ran, sd, out, y_true, y_pred = run_transformer_with_for_xai(
    tmp, target_col, window_size, test_size, batch_size, d_model, n_head,
     num_layers, epoch_number, lr, dropout,
     device, seed, optimizer_type='adam', weight_decay=weight_decay, early_stop=True,
     patience=patience,
     min_delta=min_delta,
     n_runs=3,
)

mae, mape, mse, rmse, r2 = error_metrics(y_true, y_pred)

plt.figure(figsize=(12, 6))
plt.plot(y_true, marker="o", label="Real")
plt.plot(y_pred, marker="o", label="Prediction")
plt.title("Prediction (original scale)")
plt.xlabel("Weeks")
plt.ylabel(target_col)
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(f"plots/transformers_prediction_{target_col}_{test_size}.png", dpi=200)

H = test_size  # 4
HORIZON_TO_EXPLAIN = H - 1  # last step (3). Change to 0..3 if you want.



# LIME _____________________________________________-

import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from darts import TimeSeries

model = out[0]
train_ts = out[1]
val_ts = out[2]
scaler_y = out[3]
scaler_cov = out[4]
feature_cols = out[6]


# Ensure datetime (CRITICAL)
df["FECHA"] = pd.to_datetime(df["FECHA"])

# Rebuild full target series and covariates
full_series = TimeSeries.from_dataframe(df, "FECHA", target_col)
full_cov = TimeSeries.from_dataframe(df, "FECHA", feature_cols)

series_scaled = scaler_y.transform(full_series)
cov_scaled = scaler_cov.transform(full_cov)

F = 1 + len(feature_cols)

# last forecast point
last_idx = len(df) - 1
start_hist = last_idx - test_size + 1 - window_size
assert start_hist >= 0

start_time = df["FECHA"].iloc[start_hist]
end_time = df["FECHA"].iloc[start_hist + window_size - 1]

hist_target = series_scaled.slice(start_time, end_time)
hist_cov = cov_scaled.slice(start_time, end_time)

y_hist_np = hist_target.values(copy=True)
cov_hist_np = hist_cov.values(copy=True)

hist_full_np = np.concatenate([y_hist_np, cov_hist_np], axis=1)
x0 = hist_full_np.flatten()

# Feature names
feat_names = []
cols_all = [target_col] + feature_cols
for t in range(window_size):
    lag = window_size - 1 - t
    for c in cols_all:
        feat_names.append(f"{c}_t-{lag}")

# LIME background
X_lime = np.tile(x0, (200, 1))
X_lime += np.random.normal(scale=0.01, size=X_lime.shape)

explainer = LimeTabularExplainer(
    X_lime,
    feature_names=feat_names,
    mode="regression",
    verbose=False,
    discretize_continuous=False,
)



def predict_fn(flat_X):
    flat_X = np.asarray(flat_X)
    preds = []

    for i in range(flat_X.shape[0]):
        arr = flat_X[i].reshape(window_size, F)

        # Transformer ONLY uses covariates
        cov_win = arr[:, 1:]  # (T, n_features)

        X = torch.tensor(cov_win, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            y_hat_scaled = model(X).cpu().numpy().flatten()

        # inverse-scale last step
        y_last = (
                y_hat_scaled[-1] * (scaler_y.vmax - scaler_y.vmin)
                + scaler_y.vmin
        )
        preds.append(y_last)

    return np.array(preds)


# ------------------------------------------------------------
# LIME stability analysis for ONE prediction (x0)
# ------------------------------------------------------------
#
# This LIME averages multiple runs to improve stability, while vanilla LIME uses a single random explanation.

# This Lime also disables discretaization to improve stability (because of the gaussian perturbations).

#flatten time series window



N = 20  # number of LIME runs to average
weights = []

for s in range(N):
    np.random.seed(1000 + s)
    exp = explainer.explain_instance(
        x0,
        predict_fn,
        num_features=20
    )
    weights.append(dict(exp.as_list()))

df_local_lime = pd.DataFrame(weights).fillna(0)

mean_importance = df_local_lime.mean().sort_values(ascending=False)
mean_importance.index = rename_specific(
    [prettify(n) for n in mean_importance.index]
)
std_importance = df_local_lime.std().sort_values(ascending=False)

print("Top mean LIME contributions:")
print(mean_importance.head(15))

print("\nTop unstable features (std):")
print(std_importance.head(15))

top_pos = mean_importance.sort_values(ascending=False).head(8)
top_neg = mean_importance.sort_values().head(8)

vals = mean_importance.copy()

# sort by absolute contribution
vals = vals.reindex(vals.abs().sort_values(ascending=False).index)

# keep top-k
vals = vals.head(15)
vals.index = rename_specific([prettify(n) for n in vals.index])
colors = ["green" if v > 0 else "red" for v in vals]

plt.figure(figsize=(12, 6))
plt.barh(vals.index, vals.values, color=colors)
plt.axvline(0, color="black", linewidth=1)
plt.gca().invert_yaxis()
plt.title("Mean LIME contribution (sorted by |value|)")
plt.tight_layout()
plt.savefig(
    f"plots/local_lime_mean_abs_{target_col}_{test_size}.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

# ============================================================
# BUILD X_test (Transformer-style) + GLOBAL LIME
# ============================================================

from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 0. Build X_test manually (Transformer DOES NOT store it)
# ------------------------------------------------------------
# X_test shape: (N, window_size, F)
# F = 1 (target) + n_covariates

cols_all = [target_col] + feature_cols
F = len(cols_all)
T = window_size
N = test_size  # number of forecast origins you want to explain

# scale full series (same scalers used in training)
series_scaled = scaler_y.transform(full_series)
cov_scaled = scaler_cov.transform(full_cov)

X_test_list = []

# we explain the LAST test_size rolling windows
last_idx = len(df) - 1

for i in range(N):
    end = last_idx - i
    start = end - T + 1

    start_time = df["FECHA"].iloc[start]
    end_time = df["FECHA"].iloc[end]

    y_win = series_scaled.slice(start_time, end_time).values(copy=True)  # (T,1)
    cov_win = cov_scaled.slice(start_time, end_time).values(copy=True)  # (T,n_cov)

    X_test_list.append(np.concatenate([y_win, cov_win], axis=1))

X_test = np.stack(X_test_list[::-1])  # (N,T,F) chronological order

print("X_test shape:", X_test.shape)

# ------------------------------------------------------------
# 1. Flatten for LIME
# ------------------------------------------------------------
X_lime = X_test.reshape(N, T * F)

# ------------------------------------------------------------
# 2. Feature names (MATCH flattening order!)
# ------------------------------------------------------------
feature_names = [
    f"{cols_all[j]}_t{T - 1 - t}"
    for t in range(T)
    for j in range(F)
]

# ------------------------------------------------------------
# 3. LIME explainer
# ------------------------------------------------------------
explainer = LimeTabularExplainer(
    X_lime,
    feature_names=feature_names,
    mode="regression",
    verbose=False,
    discretize_continuous=False
)

# ------------------------------------------------------------
# 4. Aggregate LIME over all test windows
# ------------------------------------------------------------
n_features = X_lime.shape[1]

agg_pos = np.zeros(n_features)
agg_neg = np.zeros(n_features)
agg_abs = np.zeros(n_features)

N_REPEATS = 10  # LIME runs per window

for i in range(N):
    print(f"\nExplaining window {i + 1}/{N}...")
    weights_i = []

    for s in range(N_REPEATS):
        print(" LIME run", s + 1)
        np.random.seed(1000 + s)

        exp_i = explainer.explain_instance(
            X_lime[i],
            predict_fn,
            num_features=n_features
        )

        weights_i.append(dict(exp_i.as_list()))

    # average LIME weights for this window
    df_i = pd.DataFrame(weights_i).fillna(0)
    mean_weights = df_i.mean()

    # accumulate averaged weights
    for feat_name, weight in mean_weights.items():
        feat_idx = feature_names.index(feat_name)

        if weight >= 0:
            agg_pos[feat_idx] += weight
        else:
            agg_neg[feat_idx] += weight

        agg_abs[feat_idx] += abs(weight)

# ----------------------------------------
# NORMALIZE across windows
# ----------------------------------------
agg_pos /= N
agg_neg /= N
agg_abs /= N

lime_summary = pd.DataFrame({
    "feature": feature_names,
    "importance_abs": agg_abs,
    "importance_pos": agg_pos,
    "importance_neg": agg_neg,
})

lime_summary["signed"] = (
        lime_summary["importance_pos"] + lime_summary["importance_neg"]
)

lime_summary = lime_summary.sort_values(
    "importance_abs", ascending=False
)

print("\nTop 30 flat features:")
print(lime_summary.head(30))

lime_summary["feature_pretty"] = rename_specific(
    [prettify(f) for f in lime_summary["feature"]]
)

# ------------------------------------------------------------
# 5. Plot: feature × time
# ------------------------------------------------------------
top = lime_summary.head(20)
colors = ["green" if v >= 0 else "red" for v in top["signed"]]

plt.figure(figsize=(9, 6))
plt.barh(top["feature_pretty"], top["signed"], color=colors)
plt.gca().invert_yaxis()
plt.xlabel("Signed contribution")
plt.title("LIME – Feature × Time Contributions")
plt.tight_layout()

plt.savefig(
    f"plots/lime_feature_time_{target_col}_H{test_size}.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

# ------------------------------------------------------------
# 6. Aggregate over TIME → base feature
# ------------------------------------------------------------
lime_summary["base_feature"] = (
    lime_summary["feature"].str.rsplit("_t", n=1).str[0]
)

agg_feat = (
    lime_summary
    .groupby("base_feature", as_index=False)
    .agg(
        signed=("signed", "sum"),
        importance_abs=("importance_abs", "sum"),
    )
    .sort_values("importance_abs", ascending=False)
)

agg_feat["base_feature_pretty"] = rename_specific(
    [prettify(f) for f in agg_feat["base_feature"]]
)
topf = agg_feat.head(20)
colors = ["green" if v >= 0 else "red" for v in topf["signed"]]

plt.figure(figsize=(9, 6))
plt.barh(topf["base_feature_pretty"], topf["signed"], color=colors)
plt.gca().invert_yaxis()
plt.xlabel("Signed contribution (aggregated over time)")
plt.title("LIME – Contributions by Feature")
plt.tight_layout()

plt.savefig(
    f"plots/lime_feature_agg_{target_col}_H{test_size}.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

#
# # ------------------------------------------------------------
# # 7. Aggregate by GROUP
# # ------------------------------------------------------------
# def feature_group(name: str) -> str:
#     if name.startswith("MEAN_FLETE"):
#         return "MEAN_FLETE*"
#     if name.startswith("SUM_TEU"):
#         return "SUM_TEU*"
#     if name.endswith("_price"):
#         return "price"
#     if name.endswith("_weekly_pct_change"):
#         return "pct_change"
#     if name.endswith("_volume"):
#         return "volume"
#     return "other"
#
#
# agg_feat["group"] = agg_feat["base_feature"].apply(feature_group)
#
# agg_group = (
#     agg_feat
#     .groupby("group", as_index=False)
#     .agg(
#         signed=("signed", "sum"),
#         importance_abs=("importance_abs", "sum"),
#     )
#     .sort_values("importance_abs", ascending=False)
# )
#
# colors = ["green" if v >= 0 else "red" for v in agg_group["signed"]]
#
# plt.figure(figsize=(8, 5))
# plt.barh(agg_group["group"], agg_group["signed"], color=colors)
# plt.gca().invert_yaxis()
# plt.xlabel("Signed contribution")
# plt.title("LIME – Contributions by Feature Group")
# plt.tight_layout()
# plt.show()

# ------------------------------------------------------------
# 8. Aggregate by TIME STEP
# ------------------------------------------------------------
lime_summary["time"] = (
    lime_summary["feature"]
    .str.extract(r"_t(\d+)$")
    .astype(int)
)

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
plt.xlabel("Lag (t)")
plt.ylabel("Signed contribution")
plt.title("LIME – Contribution by Lag")
plt.tight_layout()

plt.savefig(
    f"plots/lime_lag_contribution_{target_col}_H{test_size}.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()












#SHAPLEY

# ______________
groups = {}
for j, name in enumerate(feat_names):
    base = name.rsplit("_t", 1)[0]
    groups.setdefault(base, []).append(j)

group_names = list(groups.keys())

import numpy as np


def permutation_shapley(
        predict_fn,
        x0,
        background,
        groups,
        n_perm=200
):
    group_names = list(groups.keys())
    M = len(group_names)
    phi = np.zeros(M)

    base_pred = predict_fn(background.reshape(1, -1))[0]

    for _ in range(n_perm):
        perm = np.random.permutation(M)
        x_curr = background.copy()
        prev = base_pred

        for idx in perm:
            g = group_names[idx]
            x_curr[groups[g]] = x0[groups[g]]
            val = predict_fn(x_curr.reshape(1, -1))[0]
            phi[idx] += val - prev
            prev = val

    return phi / n_perm, group_names


baseline = X_lime.mean(axis=0)
phi, group_names = permutation_shapley(
    predict_fn,
    x0,
    baseline,
    groups,
    n_perm=300
)

import shap

shap_exp = shap.Explanation(
    values=phi,
    base_values=predict_fn(baseline.reshape(1, -1))[0],
    data=None,
    feature_names=rename_specific([prettify(n) for n in group_names])
)

shap.plots.waterfall(shap_exp, max_display=12, show=False)
plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
plt.savefig(
    f"plots/shap_waterfall_{target_col}_{test_size}.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()





import numpy as np
import shap
import matplotlib.pyplot as plt

# -------------------------
# 0) Dimensions and baseline
# -------------------------
T = window_size
F = 1 + len(feature_cols)  # target + covs
H = test_size  # forecast horizon
D = T * F

x0 = np.asarray(x0).reshape(-1)
assert x0.shape[0] == D, f"x0 must be length {D}, got {x0.shape[0]}"

# baseline window (mean of background)
baseline = X_lime.mean(axis=0).copy()
baseline = np.asarray(baseline).reshape(-1)
assert baseline.shape[0] == D

# -----------------------------------------
# 1) Build GROUPS (by base feature, across lags)
# -----------------------------------------
# feat_names are like "<col>_t-<lag>" or "<col>_t<lag>"
groups = {}
for j, name in enumerate(feat_names):
    base = name.rsplit("_t", 1)[0]
    groups.setdefault(base, []).append(j)

group_names = list(groups.keys())
M = len(group_names)
print("Num groups:", M)


# -----------------------------------------
# 2) Transformer multi-horizon predictor
# -----------------------------------------
def predict_all_horizons(flat_X):
    """
    flat_X: (n_samples, D)
    returns: (n_samples, H) in ORIGINAL scale
    """
    flat_X = np.asarray(flat_X)
    n = flat_X.shape[0]
    outs = np.zeros((n, H), dtype=float)

    for i in range(n):
        arr = flat_X[i].reshape(T, F)

        # Transformer input: covariates only (match your training)
        cov_win = arr[:, 1:]  # (T, n_cov)
        X = torch.tensor(cov_win, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            y_hat_scaled = model(X).cpu().numpy().reshape(-1)  # (H,)

        # inverse-scale all horizons
        y_hat = y_hat_scaled * (scaler_y.vmax - scaler_y.vmin) + scaler_y.vmin
        outs[i, :] = y_hat

    return outs


def predict_horizon(h, x_flat):
    """
    h in [0..H-1], x_flat: (D,)
    returns scalar forecast at horizon h (original scale)
    """
    return predict_all_horizons(x_flat.reshape(1, -1))[0, h]


# -----------------------------------------
# 3) Vanilla permutation Shapley for one horizon
# -----------------------------------------
def permutation_shapley_single_h(predict_h, x0, baseline, groups, group_names, n_perm=200):
    phi = np.zeros(len(group_names), dtype=float)

    f_base = predict_h(baseline)

    for _ in range(n_perm):
        perm = np.random.permutation(len(group_names))
        x_curr = baseline.copy()
        prev = f_base

        for k in perm:
            g = group_names[k]
            x_curr[groups[g]] = x0[groups[g]]
            val = predict_h(x_curr)
            phi[k] += val - prev
            prev = val

    return phi / n_perm


# -----------------------------------------
# 4) Compute Shapley per horizon, then aggregate (Option 1)
# -----------------------------------------
n_perm = 300  # increase if you want more stability
phi_horizons = np.zeros((H, M), dtype=float)

for h in range(H):
    phi_horizons[h, :] = permutation_shapley_single_h(
        predict_h=lambda x, hh=h: predict_horizon(hh, x),
        x0=x0,
        baseline=baseline,
        groups=groups,
        group_names=group_names,
        n_perm=n_perm
    )
    print(f"Done horizon {h + 1}/{H}")

# Aggregate across forecast window (mean contribution)
phi_agg = phi_horizons.mean(axis=0)  # (M,)

# Base value and final value as mean forecast across horizons
base_value = predict_all_horizons(baseline.reshape(1, -1))[0].mean()
final_value = predict_all_horizons(x0.reshape(1, -1))[0].mean()

# -----------------------------------------
# 5) Waterfall plot for the WHOLE prediction window (mean forecast)
# -----------------------------------------
shap_exp = shap.Explanation(
    values=phi_agg,
    base_values=base_value,
    feature_names=rename_specific([prettify(n) for n in group_names])
)

shap.plots.waterfall(shap_exp, max_display=12, show=False)
plt.tight_layout()
plt.savefig(
    f"plots/shap_waterfall_horizon_mean_{target_col}_{test_size}.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

print("✅ Saved:", f"plots/shap_waterfall_horizon_mean_{target_col}_{test_size}.png")
print("Base(mean over horizon) =", base_value)
print("Final(mean over horizon) =", final_value)
print("Check additivity:", base_value + phi_agg.sum(), "≈", final_value)

# Parameters beeswarm

N_GLOBAL = 150 # increase if you want smoother beeswarm
n_perm = 50  # permutations per explanation

HORIZON = 11  # 0,1,2,3 → choose which week to explain

# ============================================================
# BUILD MANY WINDOWS FOR GLOBAL EXPLANATION
# ============================================================

T = window_size
n_cov = len(feature_cols)
F = 1 + n_cov
H = test_size

y_all = series_scaled.values(copy=True)  # (N,1)
cov_all = cov_scaled.values(copy=True)  # (N,n_cov)
N_total = len(y_all)

# valid rolling window endpoints (avoid peeking into future)
end_min = T - 1
end_max = N_total - H - 1
valid_ends = np.arange(end_min, end_max + 1)

rng = np.random.default_rng(1048596)
chosen_ends = rng.choice(valid_ends, size=N_GLOBAL, replace=False)

X_global = np.zeros((N_GLOBAL, T * F))

for i, end in enumerate(chosen_ends):
    start = end - T + 1
    y_win = y_all[start:end + 1]
    cov_win = cov_all[start:end + 1]
    X_global[i] = np.concatenate([y_win, cov_win], axis=1).flatten()

print("X_global:", X_global.shape)

baseline_global = X_global.mean(axis=0)


def predict_horizon_fn(h):
    def f(flat_X):
        flat_X = np.asarray(flat_X)
        out = np.zeros(flat_X.shape[0])

        for i in range(flat_X.shape[0]):
            arr = flat_X[i].reshape(T, F)
            cov_win = arr[:, 1:]
            X = torch.tensor(cov_win, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                y_scaled = model(X).cpu().numpy().flatten()

            out[i] = y_scaled[h] * (scaler_y.vmax - scaler_y.vmin) + scaler_y.vmin

        return out

    return f


def permutation_shapley_grouped(predict_fn, x0, baseline, groups, n_perm=n_perm):
    group_names = list(groups.keys())
    M = len(group_names)
    phi = np.zeros(M)

    base_pred = predict_fn(baseline.reshape(1, -1))[0]

    for _ in range(n_perm):
        perm = np.random.permutation(M)
        x_curr = baseline.copy()
        prev = base_pred

        for idx in perm:
            g = group_names[idx]
            x_curr[groups[g]] = x0[groups[g]]
            val = predict_fn(x_curr.reshape(1, -1))[0]
            phi[idx] += val - prev
            prev = val

    return phi / n_perm


predict_fn = predict_horizon_fn(HORIZON)

shap_vals = np.zeros((N_GLOBAL, len(group_names)))

for i in range(N_GLOBAL):
    shap_vals[i] = permutation_shapley_grouped(
        predict_fn,
        X_global[i],
        baseline_global,
        groups,
        n_perm=n_perm
    )

print("SHAP matrix:", shap_vals.shape)

# ------------------------------------------------------------
# FIXED: Feature values for beeswarm coloring
# (mean over time window, per base feature)
# ------------------------------------------------------------

feature_values = np.zeros_like(shap_vals)

for i in range(N_GLOBAL):
    arr = X_global[i].reshape(T, F)  # (T, F)

    for j, g in enumerate(group_names):
        idxs = groups[g]  # flat indices

        vals = []
        for flat_k in idxs:
            t = flat_k // F  # time index
            f = flat_k % F  # feature index (0=target, 1+=cov)

            if f == 0:
                continue  # skip target itself

            vals.append(arr[t, f])

        feature_values[i, j] = np.mean(vals)

import shap
import matplotlib.pyplot as plt

exp = shap.Explanation(
    values=shap_vals,
    # data=feature_values,
    feature_names=rename_specific([prettify(n) for n in group_names])
)
shap.summary_plot(
    exp,
    max_display=20,
    show=False,
    color_bar=False  # <<< THIS FIXES IT
)

fig = plt.gcf()
fig.set_size_inches(10, 6)
plt.tight_layout()
plt.savefig(
    f"plots/shap_beeswarm_h{HORIZON + 1}_{target_col}_{test_size}.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()





# ============================================================
# ATTENTION MAPS FOR TRANSFORMER
# ============================================================

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
LAYER_TO_PLOT = -1  # last Transformer layer
HEAD_TO_PLOT = 0  # specific head
SAVE_DIR = "plots"

os.makedirs(SAVE_DIR, exist_ok=True)

T = window_size
F = 1 + len(feature_cols)

# ------------------------------------------------------------
# 1) ATTENTION FOR SINGLE PREDICTION (x0)
# ------------------------------------------------------------

# rebuild covariate window (Transformer input)
arr = x0.reshape(T, F)
cov_win = arr[:, 1:]  # Transformer only sees covariates

X_attn = torch.tensor(
    cov_win,
    dtype=torch.float32
).unsqueeze(0).to(device)

# get attention maps
attn_maps = get_attention_maps(model, X_attn)

# expected shape: attn_maps[layer] -> (heads, T, T)
print("Attention map shapes:")
for l, A in enumerate(attn_maps):
    print(f"Layer {l}: {A.shape}")

# ------------------------------------------------------------
# 2) PLOT SINGLE HEAD ATTENTION
# ------------------------------------------------------------

A_head = attn_maps[LAYER_TO_PLOT][0, HEAD_TO_PLOT].cpu().numpy()

# plt.figure(figsize=(8, 6))
# sns.heatmap(
#     A_head,
#     cmap="viridis",
#     xticklabels=False,
#     yticklabels=False
# )
# plt.title(
#     f"Attention Map – Layer {LAYER_TO_PLOT}, Head {HEAD_TO_PLOT}"
# )
# plt.xlabel("Key time step (past)")
# plt.ylabel("Query time step")
# plt.tight_layout()
# plt.savefig(
#     f"{SAVE_DIR}/attention_single_layer{LAYER_TO_PLOT}_head{HEAD_TO_PLOT}_{target_col}.png",
#     dpi=200
# )
# plt.close()

# ------------------------------------------------------------
# 3) PLOT MEAN ATTENTION (HEADS AVERAGED)
# ------------------------------------------------------------

A_mean = attn_maps[LAYER_TO_PLOT][0].mean(axis=0).cpu().numpy()

# plt.figure(figsize=(8, 6))
# sns.heatmap(
#     A_mean,
#     cmap="viridis",
#     xticklabels=False,
#     yticklabels=False
# )
# plt.title(
#     f"Mean Attention – Layer {LAYER_TO_PLOT}"
# )
# plt.xlabel("Key time step (past)")
# plt.ylabel("Query time step")
# plt.tight_layout()
# plt.savefig(
#     f"{SAVE_DIR}/attention_mean_layer{LAYER_TO_PLOT}_{target_col}.png",
#     dpi=200
# )
# plt.close()

print(" Saved single-window attention plots")

# ============================================================
# 4) GLOBAL ATTENTION (AVERAGED OVER MANY WINDOWS)
# ============================================================
N_GLOBAL = 80
X_global = np.zeros((N_GLOBAL, T * F))
ATTN_GLOBAL = []

for i in range(N_GLOBAL):
    arr = X_global[i].reshape(T, F)
    cov_win = arr[:, 1:]

    X_attn = torch.tensor(
        cov_win,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        attn = get_attention_maps(model, X_attn)

    ATTN_GLOBAL.append(
        attn[LAYER_TO_PLOT][0].mean(axis=0).cpu().numpy()
    )

ATTN_GLOBAL_MEAN = np.mean(ATTN_GLOBAL, axis=0)  # (T,T)

plt.figure(figsize=(8, 6))
sns.heatmap(
    ATTN_GLOBAL_MEAN,
    cmap="viridis",
    xticklabels=False,
    yticklabels=False
)
plt.title(
    f"Global Mean Attention – Layer {LAYER_TO_PLOT}"
)
plt.xlabel("Key time step (past)")
plt.ylabel("Query time step")
plt.tight_layout()
plt.savefig(
    f"{SAVE_DIR}/attention_global_mean_layer{LAYER_TO_PLOT}_{target_col}.png",
    dpi=200
)
plt.close()


# ============================================================
# GLOBAL TIME × TIME ATTENTION (LAST LAYER, HEADS AVERAGED)
# ============================================================

LAYER = -1
T = window_size
F = 1 + len(feature_cols)

ATTN = []

for i in range(N_GLOBAL):
    arr = X_global[i].reshape(T, F)
    cov_win = arr[:, 1:]

    X = torch.tensor(cov_win, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        attn_maps = get_attention_maps(model, X)
        # attn_maps[layer]: (heads, T, T)
        A = attn_maps[LAYER][0].mean(dim=0)
    # avg over heads → (T,T)

    ATTN.append(A.cpu().numpy())

ATTN_GLOBAL = np.mean(ATTN, axis=0)  # avg over windows → (T,T)

plt.figure(figsize=(8, 6))
sns.heatmap(ATTN_GLOBAL, cmap="viridis")
plt.xlabel("Key time step (past)")
plt.ylabel("Query time step")
plt.title("Global Mean Attention (Time × Time)")
plt.tight_layout()
plt.show()




# # ============================================================
# HEAD SPECIALIZATION OVER TIME
n_layers = len(attn_maps)

for layer in range(n_layers):
    A = attn_maps[layer][0].cpu().numpy()  # (heads, T, T)
    head_importance = A.mean(axis=2)  # (heads, T)

    plt.figure(figsize=(8, 4))
    sns.heatmap(
        head_importance,
        cmap="viridis",
        yticklabels=[f"Head {i}" for i in range(head_importance.shape[0])],
        xticklabels=False
    )
    plt.xlabel("Past time step")
    plt.ylabel("Attention head")
    plt.title(f"Head specialization – Layer {layer}")
    plt.tight_layout()
    plt.savefig(
        f"plots/attention_head_specialization_layer{layer}_{target_col}_h{test_size}.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()




LAYER = 1
n_heads = attn_maps[LAYER].shape[1]
for h in range(n_heads):
    A = attn_maps[LAYER][0, h].cpu().numpy()  # (T_query, T_key)
    var_query = np.mean(np.var(A, axis=0))  # varies by query
    var_key = np.mean(np.var(A, axis=1))  # varies by key
    print(
        f"Head {h}: "
        f"query-var = {var_query:.4e}, "
        f"key-var = {var_key:.4e}"
    )

head_stats = []

n_layers = len(attn_maps)

for L in range(n_layers):
    n_heads = attn_maps[L].shape[1]

    for h in range(n_heads):
        A = attn_maps[L][0, h].cpu().numpy()  # (T_query, T_key)

        query_var = np.mean(np.var(A, axis=0))
        key_var = np.mean(np.var(A, axis=1))
        total_var = query_var + key_var

        head_stats.append({
            "layer": L,
            "head": h,
            "query_var": query_var,
            "key_var": key_var,
            "total_var": total_var
        })

df_heads = (
    pd.DataFrame(head_stats)
    .sort_values("total_var", ascending=False)
    .reset_index(drop=True)
)

print(df_heads.head(10))

# ONE HEAD ATTENTION OVER TIME OF ONE LAYER
LAYER = 1  # which Transformer layer
HEAD = 6  # which attention head

# take ONE window (e.g., the first global window)
arr = X_global[0].reshape(T, F)
cov_win = arr[:, 1:]

X = torch.tensor(cov_win, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    attn_maps = get_attention_maps(model, X)
    # attn_maps[layer]: (heads, T, T)
    A_head = attn_maps[LAYER][0, HEAD].cpu().numpy()  # (T, T)

plt.figure(figsize=(8, 6))
sns.heatmap(
    A_head,
    cmap="viridis",
    linewidths=0,  # <<< removes white grid lines
    linecolor=None,
    cbar=True
)

# thin ticks
step = 5  # show one tick every 5 steps
plt.xticks(
    ticks=np.arange(0, T, step),
    labels=np.arange(0, T, step),
    rotation=0
)
plt.yticks(
    ticks=np.arange(0, T, step),
    labels=np.arange(0, T, step),
    rotation=0
)

plt.xlabel("Key time step (past)")
plt.ylabel("Query time step")
plt.title(f"Attention – Layer {LAYER}, Head {HEAD}")
plt.tight_layout()
plt.show()














