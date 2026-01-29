import warnings

warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime

from lime.lime_tabular import LimeTabularExplainer
from darts import TimeSeries

from functions import (
    error_metrics,
    run_transformer_with_for_xai,
    prettify,
    rename_specific,
)

# ============================================================
# 0) CONFIG (pick ONE model setting cleanly)
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Choose one target/horizon setup here:
CONFIGS = {
    # -------- H=4 --------
    ("FE", 4): dict(window_size=42, batch_size=128, d_model=64, n_head=2, num_layers=4,
                    lr=0.000340605, dropout=0.069862972, weight_decay=0.000187754),
    ("NAE", 4): dict(window_size=51, batch_size=32, d_model=64, n_head=4, num_layers=4,
                     lr=0.000397956, dropout=0.223329688, weight_decay=0.000103901),
    ("NAW", 4): dict(window_size=41, batch_size=64, d_model=128, n_head=4, num_layers=4,
                     lr=6.47e-05, dropout=0.121422183, weight_decay=6.10e-06),
    ("NE", 4): dict(window_size=30, batch_size=128, d_model=32, n_head=8, num_layers=4,
                    lr=9.28e-05, dropout=0.483863325, weight_decay=0.000197146),
    ("SE", 4): dict(window_size=36, batch_size=32, d_model=256, n_head=4, num_layers=4,
                    lr=0.000555638, dropout=0.312582405, weight_decay=0.000179268),
    ("SAW", 4): dict(window_size=32, batch_size=128, d_model=32, n_head=2, num_layers=2,
                     lr=0.000147759, dropout=0.424560519, weight_decay=8.66e-05),
    ("SAE", 4): dict(window_size=32, batch_size=32, d_model=128, n_head=2, num_layers=4,
                     lr=0.000482691, dropout=0.517106545, weight_decay=7.75e-06),

    # -------- H=12 --------
    ("FE", 12): dict(window_size=28, batch_size=32, d_model=128, n_head=2, num_layers=4,
                     lr=0.001, dropout=0.332641601, weight_decay=0.001),
    ("NAE", 12): dict(window_size=26, batch_size=32, d_model=64, n_head=2, num_layers=4,
                      lr=0.000187546, dropout=0.273919743, weight_decay=7.55e-06),
    ("NAW", 12): dict(window_size=51, batch_size=128, d_model=256, n_head=8, num_layers=4,
                      lr=0.00029803, dropout=0.165152428, weight_decay=5.62e-06),
    ("NE", 12): dict(window_size=47, batch_size=32, d_model=64, n_head=2, num_layers=4,
                     lr=0.0004815, dropout=0.318221159, weight_decay=4.00e-05),
    ("SE", 12): dict(window_size=50, batch_size=32, d_model=64, n_head=8, num_layers=4,
                     lr=0.000427606, dropout=0.444859662, weight_decay=1.50e-05),
    ("SAW", 12): dict(window_size=49, batch_size=128, d_model=256, n_head=4, num_layers=4,
                      lr=0.000551659, dropout=0.426466938, weight_decay=6.90e-05),
    ("SAE", 12): dict(window_size=45, batch_size=64, d_model=256, n_head=8, num_layers=3,
                      lr=0.000496422, dropout=0.506098107, weight_decay=1.00e-06),
}

# ---- choose here ----
target_col = "NAE"
test_size = 4
cfg = CONFIGS[(target_col, test_size)]

window_size = cfg["window_size"]
batch_size = cfg["batch_size"]
d_model = cfg["d_model"]
n_head = cfg["n_head"]
num_layers = cfg["num_layers"]
lr = cfg["lr"]
dropout = cfg["dropout"]
weight_decay = cfg["weight_decay"]

# General training params
seed = 1048596
patience = 200
min_delta = 1e-5
epoch_number = 2000

# LIME params
H = test_size
HORIZON_TO_EXPLAIN = H - 1  # choose 0..H-1
N_BACKGROUND = 500  # windows for LIME background
NUM_SAMPLES = 5000  # LIME perturbations per explanation
N_REPEATS_LOCAL = 20  # repeats for a single window (stability)
N_REPEATS_GLOBAL = 10  # repeats per origin in global LIME
N_ORIGINS = min(test_size, 50)  # how many origins to aggregate (increase for smoother globals)

os.makedirs("plots", exist_ok=True)

# ============================================================
# 1) LOAD DATA
# ============================================================
df = pd.read_csv("weekly_chile_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"])
df = df.sort_values("FECHA").reset_index(drop=True)

# ============================================================
# 2) TRAIN / EVAL MODEL
# ============================================================
avg_mae, avg_mape, avg_mse, avg_rmse, avg_r2, avg_epochs_ran, sd, out, y_true, y_pred = run_transformer_with_for_xai(
    df,
    target_col,
    window_size,
    test_size,
    batch_size,
    d_model,
    n_head,
    num_layers,
    epoch_number,
    lr,
    dropout,
    device,
    seed,
    optimizer_type="adam",
    weight_decay=weight_decay,
    early_stop=True,
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
plt.close()

# Unpack
model = out[0]
train_ts = out[1]
val_ts = out[2]
scaler_y = out[3]
scaler_cov = out[4]
feature_cols = out[6]

# Inference mode for XAI
model.to(device)
model.eval()
torch.set_grad_enabled(False)

# Guard: target leakage in covariates
if target_col in feature_cols:
    print(f"⚠️ WARNING: target_col '{target_col}' is present inside feature_cols (leakage). Remove it.")

# ============================================================
# 3) BUILD FULL SCALED SERIES/COVS
# ============================================================
full_series = TimeSeries.from_dataframe(df, "FECHA", target_col)
full_cov = TimeSeries.from_dataframe(df, "FECHA", feature_cols)

series_scaled = scaler_y.transform(full_series)
cov_scaled = scaler_cov.transform(full_cov)

cols_all = [target_col] + feature_cols
F = len(cols_all)
T = window_size
H = test_size

# Work directly in numpy (no slicing surprises)
y_all = series_scaled.values(copy=True)  # (N_total, 1)
cov_all = cov_scaled.values(copy=True)  # (N_total, n_cov)
N_total = len(y_all)

# Valid origins (no leakage)
end_min = T - 1
end_max = N_total - H - 1
assert end_max >= end_min, f"Not enough data: need at least window_size={T} + horizon={H} points."

valid_ends = np.arange(end_min, end_max + 1)


def build_window_flat(end_idx: int) -> np.ndarray:
    """Flatten [target + covariates] over a rolling window ending at end_idx (inclusive)."""
    start = end_idx - T + 1
    W = np.concatenate([y_all[start:end_idx + 1], cov_all[start:end_idx + 1]], axis=1)  # (T, F)
    return W.reshape(-1)


# ============================================================
# 4) BUILD A REAL BACKGROUND (X_bg) FROM MANY ROLLING WINDOWS
# ============================================================
N_BACKGROUND = min(N_BACKGROUND, len(valid_ends))
rng = np.random.default_rng(seed)
bg_ends = rng.choice(valid_ends, size=N_BACKGROUND, replace=False)

X_bg = np.vstack([build_window_flat(int(e)) for e in bg_ends])  # (N_BACKGROUND, T*F)

# Feature names must match flattening order: time-major then feature
feature_names = [
    f"{cols_all[j]}_t{T - 1 - t}"  # t0=oldest, t(T-1)=most recent
    for t in range(T)
    for j in range(F)
]
feat_to_idx = {n: i for i, n in enumerate(feature_names)}
n_features = len(feature_names)


# ============================================================
# 5) PREDICT FN FOR LIME (expects flat (T*F), uses covariates only)
# ============================================================
def _inverse_scale_y(y_scaled: np.ndarray) -> np.ndarray:
    y_scaled = np.asarray(y_scaled).reshape(-1)

    if hasattr(scaler_y, "vmin") and hasattr(scaler_y, "vmax"):
        vmin = float(np.asarray(scaler_y.vmin).reshape(-1)[0])
        vmax = float(np.asarray(scaler_y.vmax).reshape(-1)[0])
        return y_scaled * (vmax - vmin) + vmin

    if hasattr(scaler_y, "inverse_transform"):
        ts = TimeSeries.from_values(y_scaled.reshape(-1, 1))
        inv = scaler_y.inverse_transform(ts).values(copy=False).reshape(-1)
        return inv

    raise RuntimeError("Unknown scaler_y type: cannot inverse-scale predictions.")


def predict_fn(flat_X: np.ndarray) -> np.ndarray:
    """Return predictions (original scale) for the selected horizon index."""
    flat_X = np.asarray(flat_X)
    preds = np.zeros(flat_X.shape[0], dtype=float)

    for i in range(flat_X.shape[0]):
        arr = flat_X[i].reshape(T, F)
        cov_win = arr[:, 1:]  # drop target column (matches your training)

        X = torch.tensor(cov_win, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            y_hat_scaled = model(X).detach().cpu().numpy().reshape(-1)  # (H,)

        y_hat = _inverse_scale_y(y_hat_scaled)
        preds[i] = float(y_hat[HORIZON_TO_EXPLAIN])

    return preds


# ============================================================
# 6) LIME EXPLAINER (single explainer for both local+global)
# ============================================================
explainer = LimeTabularExplainer(
    X_bg,
    feature_names=feature_names,
    mode="regression",
    verbose=False,
    discretize_continuous=False,
    sample_around_instance=True,  # keeps perturbations local around the instance
    random_state=seed,
)

# ============================================================
# 7) LOCAL LIME (FIXED): average CONTRIBUTIONS, not raw coefficients
# ============================================================
# choose a specific instance to explain (last valid origin)
end0 = int(valid_ends[-1])
x0 = build_window_flat(end0)

# scaled instance values used for contributions
x0_scaled = explainer.scaler.transform(x0.reshape(1, -1))[0]

contrib_runs = []
scores_local = []

for s in range(N_REPEATS_LOCAL):
    np.random.seed(1000 + s)

    exp = explainer.explain_instance(
        x0,
        predict_fn,
        num_features=n_features,
        num_samples=NUM_SAMPLES,
    )

    scores_local.append(getattr(exp, "score", np.nan))

    coef_vec = np.zeros(n_features, dtype=float)
    for feat_name, w in exp.as_list():
        coef_vec[feat_to_idx[feat_name]] = w

    contrib_runs.append(coef_vec * x0_scaled)  # <<< contribution

contrib_runs = np.asarray(contrib_runs)  # (N_REPEATS_LOCAL, n_features)
mean_contrib = contrib_runs.mean(axis=0)
std_contrib = contrib_runs.std(axis=0)

local_df = pd.DataFrame({
    "feature": feature_names,
    "mean_contrib": mean_contrib,
    "std_contrib": std_contrib,
})
local_df["abs_mean"] = local_df["mean_contrib"].abs()
local_df = local_df.sort_values("abs_mean", ascending=False)

print("\nLocal LIME surrogate fidelity (mean R^2):", float(np.nanmean(scores_local)))
print("\nTop mean local contributions:")
print(local_df.head(15)[["feature", "mean_contrib", "std_contrib"]])

top_local = local_df.head(15).copy()
top_local["feature_pretty"] = rename_specific([prettify(f) for f in top_local["feature"]])
colors = ["green" if v > 0 else "red" for v in top_local["mean_contrib"]]

plt.figure(figsize=(12, 6))
plt.barh(top_local["feature_pretty"], top_local["mean_contrib"], color=colors)
plt.axvline(0, color="black", linewidth=1)
plt.gca().invert_yaxis()
plt.title("Local LIME – Mean contribution (coef × x_scaled)")
plt.tight_layout()
plt.savefig(f"plots/local_lime_mean_contrib_{target_col}_{test_size}_h{HORIZON_TO_EXPLAIN + 1}.png", dpi=200,
            bbox_inches="tight")
plt.close()

# ============================================================
# 8) GLOBAL / AGGREGATED LIME (already correct, kept + cleaned)
# ============================================================
# windows to explain (last N_ORIGINS valid origins)
N_ORIGINS = min(N_ORIGINS, len(valid_ends))
ends_explain = valid_ends[-N_ORIGINS:]
X_explain = np.vstack([build_window_flat(int(e)) for e in ends_explain])  # (N_ORIGINS, T*F)

agg_signed = np.zeros(n_features, dtype=float)
agg_abs = np.zeros(n_features, dtype=float)
scores_global = []

for i in range(N_ORIGINS):
    print(f"\nExplaining origin {i + 1}/{N_ORIGINS} (end_idx={int(ends_explain[i])})...")

    x_inst = X_explain[i]
    x_scaled = explainer.scaler.transform(x_inst.reshape(1, -1))[0]

    contrib_runs = []
    for s in range(N_REPEATS_GLOBAL):
        np.random.seed(2000 + i * 100 + s)

        exp_i = explainer.explain_instance(
            x_inst,
            predict_fn,
            num_features=n_features,
            num_samples=NUM_SAMPLES,
        )

        scores_global.append(getattr(exp_i, "score", np.nan))

        coef_vec = np.zeros(n_features, dtype=float)
        for feat_name, w in exp_i.as_list():
            coef_vec[feat_to_idx[feat_name]] = w

        contrib_runs.append(coef_vec * x_scaled)

    contrib_mean = np.mean(contrib_runs, axis=0)

    agg_signed += contrib_mean
    agg_abs += np.abs(contrib_mean)

agg_signed /= N_ORIGINS
agg_abs /= N_ORIGINS

print("\nGlobal LIME surrogate fidelity (mean R^2):", float(np.nanmean(scores_global)))

lime_summary = pd.DataFrame({
    "feature": feature_names,
    "importance_abs": agg_abs,
    "signed": agg_signed,
})
lime_summary["importance_pos"] = np.clip(lime_summary["signed"], 0, None)
lime_summary["importance_neg"] = np.clip(lime_summary["signed"], None, 0)
lime_summary = lime_summary.sort_values("importance_abs", ascending=False)

print("\nTop 30 flat features (global):")
print(lime_summary.head(30)[["feature", "signed", "importance_abs"]])

lime_summary["feature_pretty"] = rename_specific([prettify(f) for f in lime_summary["feature"]])

# ---- Plot: Feature × Time (top 20)
top = lime_summary.head(20).copy()
colors = ["green" if v >= 0 else "red" for v in top["signed"]]

plt.figure(figsize=(9, 6))
plt.barh(top["feature_pretty"], top["signed"], color=colors)
plt.gca().invert_yaxis()
plt.xlabel("Signed contribution (mean over origins)")
plt.title(f"LIME – Feature × Time (horizon={HORIZON_TO_EXPLAIN + 1}/{H})")
plt.tight_layout()
plt.savefig(f"plots/lime_feature_time_{target_col}_H{test_size}_h{HORIZON_TO_EXPLAIN + 1}.png", dpi=200,
            bbox_inches="tight")
plt.close()

# ---- Aggregate over TIME → base feature
lime_summary["base_feature"] = lime_summary["feature"].str.rsplit("_t", n=1).str[0]

agg_feat = (
    lime_summary
    .groupby("base_feature", as_index=False)
    .agg(signed=("signed", "sum"), importance_abs=("importance_abs", "sum"))
    .sort_values("importance_abs", ascending=False)
)

agg_feat["base_feature_pretty"] = rename_specific([prettify(f) for f in agg_feat["base_feature"]])

topf = agg_feat.head(20).copy()
colors = ["green" if v >= 0 else "red" for v in topf["signed"]]

plt.figure(figsize=(9, 6))
plt.barh(topf["base_feature_pretty"], topf["signed"], color=colors)
plt.gca().invert_yaxis()
plt.xlabel("Signed contribution (aggregated over time)")
plt.title(f"LIME – Contributions by Feature (horizon={HORIZON_TO_EXPLAIN + 1}/{H})")
plt.tight_layout()
plt.savefig(f"plots/lime_feature_agg_{target_col}_H{test_size}_h{HORIZON_TO_EXPLAIN + 1}.png", dpi=200,
            bbox_inches="tight")
plt.close()

print("\nSaved plots to ./plots/")
