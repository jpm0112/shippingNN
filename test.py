# ============================================================
# DEEP SHAP FOR TRANSFORMER MODEL
# ============================================================
#
# DeepSHAP uses the DeepLIFT algorithm to compute SHAP values efficiently
# for deep neural networks. It provides theoretically grounded feature
# attributions based on Shapley values.

import shap
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import prettify, rename_specific

# ------------------------------------------------------------
# 0) Setup: Extract model and data from your training output
# ------------------------------------------------------------
xai_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = out[0]
train_ts = out[1]
val_ts = out[2]
scaler_y = out[3]
scaler_cov = out[4]
feature_cols = out[6]

torch.set_grad_enabled(True)
# Ensure model is in eval mode
model.to(xai_device)
model.eval()

# Rebuild scaled series (same as your LIME code)
df["FECHA"] = pd.to_datetime(df["FECHA"])
full_series = TimeSeries.from_dataframe(df, "FECHA", target_col)
full_cov = TimeSeries.from_dataframe(df, "FECHA", feature_cols)
series_scaled = scaler_y.transform(full_series)
cov_scaled = scaler_cov.transform(full_cov)

# ------------------------------------------------------------
# 1) Build valid windows (same logic as your LIME code)
# ------------------------------------------------------------
cols_all = [target_col] + feature_cols
F = len(cols_all)
T = window_size
H = test_size

y_all = series_scaled.values(copy=True)  # (N_total, 1)
cov_all = cov_scaled.values(copy=True)  # (N_total, n_cov)
N_total = len(y_all)

end_min = T - 1
end_max = N_total - H - 1
valid_ends = np.arange(end_min, end_max + 1)

# ------------------------------------------------------------
# 2) Create background dataset for SHAP
# ------------------------------------------------------------
N_BACKGROUND = min(104, len(valid_ends))  # DeepSHAP works well with ~100 background samples (2 years)
rng = np.random.default_rng(seed)
bg_ends = rng.choice(valid_ends, size=N_BACKGROUND, replace=False)

# For DeepSHAP, we need the INPUT format the model expects
# Your transformer uses only covariates (not target), shape: (batch, T, n_cov)
n_cov = len(feature_cols)

X_background = np.zeros((N_BACKGROUND, T, n_cov), dtype=np.float32)
for i, end in enumerate(bg_ends):
    start = end - T + 1
    X_background[i] = cov_all[start:end + 1]

X_background_tensor = torch.tensor(X_background, dtype=torch.float32).to(xai_device)

# ------------------------------------------------------------
# 3) Create instances to explain (last N origins)
# ------------------------------------------------------------

N_EXPLAIN = 4
non_overlap_ends = valid_ends[::test_size]  # not really non-overlapping, but spaced by test_size

ends_explain = valid_ends[-N_EXPLAIN:]  # for overlapping windows
ends_explain = non_overlap_ends[-N_ORIGINS:]  # for semi-overlapping windows

X_explain = np.zeros((N_EXPLAIN, T, n_cov), dtype=np.float32)
for i, end in enumerate(ends_explain):
    start = end - T + 1
    X_explain[i] = cov_all[start:end + 1]

X_explain_tensor = torch.tensor(X_explain, dtype=torch.float32).to(xai_device)


# ------------------------------------------------------------
# 4) Wrapper model for SHAP (to select specific horizon output)
# ------------------------------------------------------------
class ModelWrapperForHorizon(torch.nn.Module):
    def __init__(self, model, horizon_idx, vmin, vmax):
        super().__init__()
        self.model = model
        self.h = horizon_idx
        self.vmin = float(vmin)
        self.vmax = float(vmax)

    def forward(self, x):
        out = self.model(x)
        if out.dim() == 3:
            out = out.squeeze(-1)
        y_scaled = out[:, self.h:self.h + 1]
        y_orig = y_scaled * (self.vmax - self.vmin) + self.vmin
        return y_orig


HORIZON_TO_EXPLAIN = 0  # explain the first horizon (index 0)
wrapped_model = ModelWrapperForHorizon(
    model,
    HORIZON_TO_EXPLAIN,
    scaler_y.vmin,
    scaler_y.vmax
)
wrapped_model.eval()

# ------------------------------------------------------------
# 5) Initialize DeepSHAP explainer
# ------------------------------------------------------------
print(f"Initializing DeepSHAP with {N_BACKGROUND} background samples...")
print(f"Explaining {N_EXPLAIN} instances for horizon {HORIZON_TO_EXPLAIN + 1}/{H}")

# DeepExplainer for deep neural networks
try:
    explainer_shap = shap.DeepExplainer(wrapped_model, X_background_tensor)
    print("DeepExplainer initialized successfully!")
except Exception as e:
    print(f"DeepExplainer failed: {e}")
    print("Falling back to GradientExplainer...")
    explainer_shap = shap.GradientExplainer(wrapped_model, X_background_tensor)

# ------------------------------------------------------------
# 6) Compute SHAP values
# ------------------------------------------------------------
print("\nComputing SHAP values...")
shap_values = explainer_shap.shap_values(X_explain_tensor, check_additivity=False)

# Handle output format
if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_values = np.array(shap_values)
print(f"SHAP values shape (raw): {shap_values.shape}")

# FIX: Squeeze any extra dimensions (e.g., output dimension)
shap_values = np.squeeze(shap_values)
print(f"SHAP values shape (squeezed): {shap_values.shape}")

# Ensure shape is (N_EXPLAIN, T, n_cov)
if shap_values.ndim == 2:
    # If flattened, reshape
    shap_values = shap_values.reshape(N_EXPLAIN, T, n_cov)

# ------------------------------------------------------------
# 7) Aggregate SHAP values across instances
# ------------------------------------------------------------
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)  # (T, n_cov)
mean_signed_shap = np.mean(shap_values, axis=0)  # (T, n_cov)

shap_feature_names = []
shap_abs_values = []
shap_signed_values = []

for t in range(T):
    lag = T - 1 - t
    for j, col in enumerate(feature_cols):
        shap_feature_names.append(f"{col}_t-{lag}")
        # FIX: Ensure scalar values
        shap_abs_values.append(float(mean_abs_shap[t, j]))
        shap_signed_values.append(float(mean_signed_shap[t, j]))

shap_summary = pd.DataFrame({
    "feature": shap_feature_names,
    "importance_abs": shap_abs_values,
    "signed": shap_signed_values,
})
shap_summary = shap_summary.sort_values("importance_abs", ascending=False)

print("\nTop 30 features by DeepSHAP:")
print(shap_summary.head(30))

# ------------------------------------------------------------
# 8) Plot: Top features by signed importance
# ------------------------------------------------------------
top_shap = shap_summary.head(20).copy()
top_shap["feature_pretty"] = rename_specific(
    [prettify(f) for f in top_shap["feature"]]
)

# FIX: Ensure values are plain floats for plotting
signed_values = top_shap["signed"].astype(float).tolist()
colors = ["green" if v >= 0 else "red" for v in signed_values]

# plt.figure(figsize=(10, 7))
# plt.barh(top_shap["feature_pretty"].tolist(), signed_values, color=colors)
# plt.gca().invert_yaxis()
# plt.xlabel("Mean SHAP value (contribution to prediction)")
# plt.title(f"DeepSHAP – Feature Importance")
# plt.axvline(0, color="black", linewidth=0.5)
# plt.tight_layout()
# plt.savefig(
#     f"plots/deepshap_feature_importance_{target_col}_H{test_size}.png",
#     dpi=200,
#     bbox_inches="tight"
# )
# plt.show()

# ------------------------------------------------------------
# 9) SHAP Summary Plot (beeswarm style)
# ------------------------------------------------------------
# Reshape for SHAP's built-in plotting
shap_flat = shap_values.reshape(N_EXPLAIN, -1)  # (N_EXPLAIN, T * n_cov)
X_flat = X_explain.reshape(N_EXPLAIN, -1)  # (N_EXPLAIN, T * n_cov)

# Create feature names for flattened data
flat_feature_names = []
for t in range(T):
    lag = T - 1 - t
    for col in feature_cols:
        flat_feature_names.append(f"{col}_t-{lag}")

# Pretty names
flat_feature_names_pretty = rename_specific(
    [prettify(f) for f in flat_feature_names]
)

# Create SHAP Explanation object for built-in plots
explanation = shap.Explanation(
    values=shap_flat,
    base_values=np.zeros(N_EXPLAIN),  # DeepSHAP normalizes around 0
    data=X_flat,
    feature_names=flat_feature_names_pretty
)

# plt.figure(figsize=(12, 8))
# shap.plots.beeswarm(
#     explanation,
#     max_display=20,
#     show=False,
#     color_bar=False
# )
# plt.title(f"DeepSHAP Summary Plot (horizon={HORIZON_TO_EXPLAIN + 1}/{H})")
# plt.savefig(
#     f"plots/deepshap_beeswarm_{target_col}_H{test_size}.png",
#     dpi=200,
#     bbox_inches="tight"
# )
# plt.show()

# ------------------------------------------------------------
# 10) Temporal importance heatmap (Feature × Time)
# ------------------------------------------------------------
# Sum absolute SHAP across all instances
temporal_importance = np.mean(np.abs(shap_values), axis=0).T  # (n_cov, T)

# plt.figure(figsize=(12, 10))
# plt.imshow(temporal_importance, aspect='auto', cmap='YlOrRd')
# plt.colorbar(label='Mean |SHAP value|')
# plt.xlabel('Time step (t-window+1 to t)')
# plt.ylabel('Feature')
# plt.yticks(range(n_cov), feature_cols)
# plt.xticks(range(0, T, max(1, T // 10)), range(0, T, max(1, T // 10)))
# plt.title(f"DeepSHAP Temporal Heatmap (horizon={HORIZON_TO_EXPLAIN + 1}/{H})")
# plt.tight_layout()
# plt.savefig(
#     f"plots/deepshap_temporal_heatmap_{target_col}_H{test_size}.png",
#     dpi=200,
#     bbox_inches="tight"
# )
# plt.show()

# ------------------------------------------------------------
# 11) Per-feature total importance (summed over time)
# ------------------------------------------------------------
feature_total_importance = np.sum(np.abs(mean_abs_shap), axis=0)  # (n_cov,)
feature_importance_df = pd.DataFrame({
    "feature": feature_cols,
    "total_importance": feature_total_importance
}).sort_values("total_importance", ascending=True)

# plt.figure(figsize=(12, 10))
# plt.barh(feature_importance_df["feature"], feature_importance_df["total_importance"], color="steelblue")
# plt.xlabel("Total |SHAP| (summed over time)")
# plt.title(f"DeepSHAP – Feature Total Importance (horizon={HORIZON_TO_EXPLAIN + 1}/{H})")
# plt.tight_layout()
# plt.savefig(
#     f"plots/deepshap_feature_total_{target_col}_H{test_size}.png",
#     dpi=200,
#     bbox_inches="tight"
# )
# plt.show()

print("\n" + "=" * 60)
print("DeepSHAP analysis complete!")
print("=" * 60)

# ------------------------------------------------------------
# Waterfall plot for ONE instance (lag-level)
# ------------------------------------------------------------
instance_idx = 0

# Aggregate SHAP by lag (sum over features)
shap_inst_lag = shap_values[instance_idx].sum(axis=1)  # shape (T,)

# Lag names (t-0 = most recent)
lag_names = [f"t-{T - 1 - t}" for t in range(T)]

x_inst = X_explain[instance_idx].reshape(-1)

base_value = explainer_shap.expected_value

# unwrap all possible containers
if isinstance(base_value, (list, np.ndarray)):
    base_value = np.array(base_value).reshape(-1)[0]

base_value = float(base_value)
waterfall_exp = shap.Explanation(
    values=shap_inst_lag,
    base_values=base_value,
    feature_names=lag_names
)

# plt.figure(figsize=(8, 6))
# shap.plots.waterfall(waterfall_exp, max_display=12, show=False)
# plt.title(
#     f"DeepSHAP Waterfall by Lag (Instance {instance_idx}, "
#     f"Horizon {HORIZON_TO_EXPLAIN + 1}/{H})"
# )
# plt.tight_layout()
# plt.savefig(
#     f"plots/deepshap_waterfall_by_lag_{target_col}_H{test_size}_inst{instance_idx}.png",
#     dpi=200,
#     bbox_inches="tight"
# )
# plt.show()
# ============================================================
# ------------------------------------------------------------
# Waterfall plot – aggregate by FEATURE (sum over lags)
# ------------------------------------------------------------
instance_idx = 0

# shap_values shape: (N_EXPLAIN, T, n_cov)
# Sum over time (lags)
shap_inst_feat = shap_values[instance_idx].sum(axis=0)  # (n_cov,)

base_value = explainer_shap.expected_value
if isinstance(base_value, (list, np.ndarray)):
    base_value = np.array(base_value).reshape(-1)[0]
base_value = float(base_value)

feature_names = rename_specific([prettify(c) for c in feature_cols])

waterfall_exp = shap.Explanation(
    values=shap_inst_feat,
    base_values=base_value,
    feature_names=feature_names
)

plt.figure(figsize=(8, 6))
shap.plots.waterfall(waterfall_exp, max_display=12, show=False)
plt.title(
    f"DeepSHAP Waterfall by Feature (Instance {instance_idx}, "
    f"Horizon {HORIZON_TO_EXPLAIN + 1}/{H})"
)
plt.tight_layout()
plt.savefig(
    f"plots/deepshap_waterfall_by_feature_{target_col}_H{test_size}_inst{instance_idx}.png",
    dpi=200,
    bbox_inches="tight"
)
plt.show()

# BEESWARM PLOT FOR ONE INSTANCE


N_GLOBAL = 200
T = window_size
n_cov = len(feature_cols)
H = test_size

end_min = T - 1
end_max = N_total - H - 1
valid_ends = np.arange(end_min, end_max + 1)

rng = np.random.default_rng(1048596)
chosen_ends = rng.choice(valid_ends, size=N_GLOBAL, replace=False)

X_global = np.zeros((N_GLOBAL, T, n_cov), dtype=np.float32)

for i, end in enumerate(chosen_ends):
    start = end - T + 1
    X_global[i] = cov_all[start:end + 1]

X_global_tensor = torch.tensor(X_global, dtype=torch.float32).to(xai_device)

shap_vals = explainer_shap.shap_values(X_global_tensor, check_additivity=False)

if isinstance(shap_vals, list):
    shap_vals = shap_vals[0]

shap_vals = np.squeeze(np.array(shap_vals))
# shape: (N_GLOBAL, T, n_cov)

# sum over time (lags)
shap_feat = shap_vals.sum(axis=1)  # (N_GLOBAL, n_cov)

feature_values = X_global.mean(axis=1)  # (N_GLOBAL, n_cov)

exp = shap.Explanation(
    values=shap_feat,
    data=feature_values,
    feature_names=rename_specific([prettify(c) for c in feature_cols])
)

shap.summary_plot(
    exp,
    max_display=20,
    show=False,
    color_bar=False
)

# manual legend (same as before)
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Low feature value',
           markerfacecolor='blue', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='High feature value',
           markerfacecolor='red', markersize=8),
]

plt.legend(
    handles=legend_elements,
    loc='lower right',
    frameon=False,
    fontsize=10
)

fig = plt.gcf()
fig.set_size_inches(10, 6)
plt.tight_layout()

plt.savefig(
    f"plots/deepshap_beeswarm_h{HORIZON_TO_EXPLAIN + 1}_{target_col}_{test_size}.png",
    dpi=200,
    bbox_inches="tight"
)
plt.show()
plt.close()

import numpy as np
import shap
import matplotlib.pyplot as plt

# example: X_h is the input matrix used to explain horizon h
X_vals = [
    X_h0,  # horizon 1
    X_h1,  # horizon 2
    X_h2,  # ...
]

# stack all horizons
shap_all = np.vstack(shap_vals)  # (H*n_samples, n_features)
X_all = np.vstack(X_vals)

exp = shap.Explanation(
    values=shap_all,
    data=X_all,
    feature_names=feature_names
)

shap.plots.beeswarm(
    exp,
    max_display=20,
    show=False
)

plt.title("SHAP beeswarm (all horizons pooled)")
plt.tight_layout()
plt.show()
