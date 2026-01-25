import pandas as pd
from functions import error_metrics, run_darts_tft_with_for_xai
import matplotlib
matplotlib.use("Agg")   # <-- no GUI, safe in PyCharm
import matplotlib.pyplot as plt
from lightning.pytorch import  seed_everything
import torch
from datetime import datetime
import numpy as np
from darts.explainability.tft_explainer import TFTExplainer
import shap



# ==== 1. Load your data ====
df = pd.read_csv("weekly_chile_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"])
df = df.sort_values("FECHA").reset_index(drop=True)


target_col = "SAE"

test_size = 4 # number of weeks to forecast
window_size = 23
hidden_size = 64
lstm_layers = 2
num_attention_heads = 4
dropout = 0.166722297
batch_size = 64
lr = 0.000258799
n_epochs = 2000
grad_clip = 2.125473514
patience = 100
min_delta = 1e-5
seed = 1048596

# ==== 2. Run Darts TFT ====

deleted_sample = test_size*3  # delete the test samples from the end
tmp = df.copy().sort_values("FECHA")
if deleted_sample > 0:
        tmp = tmp.iloc[:-deleted_sample]

start = datetime.now()
# y_true, y_pred, out = run_darts_tft(tmp,target_col,test_size,window_size,hidden_size,lstm_layers,num_attention_heads,dropout,
#         batch_size,n_epochs,lr,grad_clip,patience=patience,min_delta=min_delta,seed=seed)
# mae, mape, mse, rmse, r2 = error_metrics(y_true, y_pred)
runtime = (datetime.now() - start).total_seconds()


mae, mape, mse, rmse, r2, epochs_ran, sd, out, y_true, y_pred = run_darts_tft_with_for_xai(
    df=tmp,
    target_col=target_col,
    test_size=test_size,
    window_size=window_size,
    hidden_size=hidden_size,
    lstm_layers=lstm_layers,
    num_attention_heads=num_attention_heads,
    dropout=dropout,
    batch_size=batch_size,
    n_epochs=n_epochs,
    lr=lr,
    grad_clip=grad_clip,
    patience=patience,
    min_delta=min_delta,
    seed=seed,
    sample_sets=3,
)

mae, mape, mse, rmse, r2 = error_metrics(y_true, y_pred)
runtime = (datetime.now() - start).total_seconds()
print("prediction errors")
print(f"MAE={mae:.3f}, MAPE={mape:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}, Time={runtime:.1f}s")

print(y_true)
print(y_pred)




model = out[0]
X_train = out[1]
X_test = out[2]
feature_names = out[6]

plt.figure(figsize=(12, 6))
plt.plot(y_true, marker="o", label="Real")
plt.plot(y_pred, marker="o", label="Prediction")
plt.title("Prediction (original scale)")
plt.xlabel("Weeks")
plt.ylabel(target_col)
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(f"plots/tft_prediction_{target_col}_{test_size}.png", dpi=200)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from darts import TimeSeries

# unpack from out
model, train_ts, val_ts, scaler_y, scaler_cov, epochs_ran, feature_cols = out

# ----------------------------
# 1. Build series & covariates
# ----------------------------
# series/covariates for the df used in training (tmp)
series_tmp = TimeSeries.from_dataframe(
    tmp,
    time_col="FECHA",
    value_cols=target_col,
)
cov_tmp = TimeSeries.from_dataframe(
    tmp,
    time_col="FECHA",
    value_cols=feature_cols,
)

series_tmp_scaled = scaler_y.transform(series_tmp)
cov_tmp_scaled = scaler_cov.transform(cov_tmp)

# ----------------------------
# 2. Forecast the held-out tail
# ----------------------------
horizon = deleted_sample  # same as your test_size

pred_scaled_test = model.predict(
    n=horizon,
    past_covariates=cov_tmp_scaled,
    dataloader_kwargs={"num_workers": 0},
)
pred_test = scaler_y.inverse_transform(pred_scaled_test)
y_pred_test = pred_test.values().flatten()

# ground truth: last deleted_sample rows you cut off
test_df = df.tail(deleted_sample)
y_true_test = test_df[target_col].values
test_dates = test_df["FECHA"].values

# ----------------------------
# 3. History window before the forecast
# ----------------------------
# window_size points right before the first test point
hist_df = df.iloc[-(window_size + deleted_sample):-deleted_sample]
hist_dates = hist_df["FECHA"].values
hist_values = hist_df[target_col].values

# ============================
# 5. Fitted values over window
# ============================

# scaled full series used in training (tmp)
full_series_scaled = series_tmp_scaled
full_cov_scaled = cov_tmp_scaled

# Get indices of the window inside tmp
# hist_df corresponds to the last (window_size + deleted_sample) rows of df,
# but tmp is df without the deleted_sample tail
start_idx = len(tmp) - window_size
end_idx = len(tmp)  # exclusive

# Extract scaled window from tmp
window_series = full_series_scaled[start_idx:end_idx]
window_cov = full_cov_scaled[start_idx:end_idx]

# Now compute fitted values for each step in the window:
# Predict 1-step ahead recursively
y_fitted_window = []

for t in range(window_size):
    # use history up to t (exclusive of the point we want to predict)
    hist_series_t = window_series[:t]
    hist_cov_t = window_cov

    # model needs at least window_size history; skip the first part
    if len(hist_series_t) < window_size:
        y_fitted_window.append(np.nan)
        continue

    pred_scaled = model.predict(
        n=1,
        series=window_series[:t],
        past_covariates=window_cov[:t],
        dataloader_kwargs={"num_workers": 0},
    )
    pred_unscaled = scaler_y.inverse_transform(pred_scaled).values().flatten()[0]
    y_fitted_window.append(pred_unscaled)

y_fitted_window = np.array(y_fitted_window)


# ----------------------------
# 4. Plot window + prediction + ground truth
# ----------------------------
plt.figure(figsize=(12, 6))

# history (what the model "sees")
plt.plot(hist_dates, hist_values, marker="o", label=f"History (last {window_size} weeks)")
plt.plot(test_dates, y_true_test, marker="o", label="Ground truth (test)")
plt.plot(test_dates, y_pred_test, marker="o", linestyle="--", label="TFT prediction")
plt.plot(hist_dates, y_fitted_window, marker=".", linestyle="-.", label="Fitted (in-sample)")


# vertical line at forecast start
plt.axvline(hist_dates[-1], linestyle="--", linewidth=1.0)
plt.text(
    hist_dates[-1],
    max(hist_values.max(), y_true_test.max()),
    " forecast start",
    ha="left",
    va="bottom",
    rotation=90,
    fontsize=9,
)

plt.title(f"TFT forecast for {target_col}: history window + {deleted_sample}-step horizon")
plt.xlabel("Date")
plt.ylabel(target_col)
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(f"plots/tft_window_plus_test_{target_col}.png", dpi=200)




model = out[0]
scaler_y = out[3]
scaler_cov = out[4]
feature_cols = out[6]   # past covariates

# Rebuild full series from the df you actually trained on (tmp)
series_full = TimeSeries.from_dataframe(tmp, time_col="FECHA", value_cols=target_col)
cov_full    = TimeSeries.from_dataframe(tmp, time_col="FECHA", value_cols=feature_cols)

# Use the SAME scalers you used for training
series_scaled = scaler_y.transform(series_full)
cov_scaled    = scaler_cov.transform(cov_full)

# 1) Create explainer
explainer = TFTExplainer(
    model,
    background_series=series_scaled,
    background_past_covariates=cov_scaled,
)

# 2) Get explainability result
expl_res = explainer.explain()   # uses background_* by default

# 3) Get variable selection / feature importances
importances = expl_res.get_feature_importances()
encoder_imp = importances["encoder_importance"]              # pd.DataFrame
decoder_imp = importances["decoder_importance"]              # pd.DataFrame
static_imp  = importances["static_covariates_importance"]    # pd.DataFrame





# VSN

plt.figure(figsize=(12, 4))
encoder_imp.mean(axis=0).plot(kind="bar")
plt.title("TFT Encoder Variable Selection (Average Importance)")
plt.ylabel("Importance")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"plots/tft_VSN_encoder_weights_{target_col}.png", dpi=200)

plt.figure(figsize=(12, 4))
decoder_imp.mean(axis=0).plot(kind="bar")
plt.title("TFT Decoder Variable Selection (Average Importance)")
plt.ylabel("Importance")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"plots/tft_VSN_decoder_weights_{target_col}.png", dpi=200)

# ATTENTION MAPS


# aggregated:
explainer.plot_attention(expl_res)
fig = plt.gcf()
fig.savefig(f"plots/tft_attention_{target_col}.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# --- 2. Get raw attention *before* Darts aggregates heads ---
att_raw = model.model._attn_out_weights.detach().cpu().numpy()
print("att_raw shape:", att_raw.shape)
# usually: (n_series, n_horizons, n_heads, T_total)

_, n_horizons, n_heads, T_total = att_raw.shape

# relative index: -input_chunk_length ... output_chunk_length-1
rel_idx = np.arange(-model.input_chunk_length, model.output_chunk_length)

# sanity check
assert T_total == len(rel_idx), "T_total != encoder+decoder length, adjust indexing if this trips."

# --- 3. Plot one heatmap per head and save ---
for h in range(n_heads):
    head_att = att_raw[0, :, h, :]      # [horizon, time]

    plt.figure(figsize=(6, 4))
    X, Y = np.meshgrid(rel_idx, np.arange(1, n_horizons + 1))
    pcm = plt.pcolormesh(X, Y, head_att, shading="auto")
    plt.colorbar(pcm, label="Attention")
    plt.xlabel("Index relative to first prediction point")
    plt.ylabel("Horizon")
    plt.title(f"TFT attention – head {h+1}")
    plt.tight_layout()
    plt.savefig(f"plots/tft_head{h+1}_attention_{target_col}.png", dpi=300)
    plt.close()




# LIME _____________________________________________-

import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from darts import TimeSeries

model = out[0]
train_ts = out[1]        # TimeSeries (target, *unscaled*)
val_ts   = out[2]        # TimeSeries (target, *unscaled*)
scaler_y = out[3]
scaler_cov = out[4]
feature_cols = out[6]

# Rebuild full target series and covariates (unscaled)
full_series = TimeSeries.from_dataframe(
    df,
    time_col="FECHA",
    value_cols=target_col,
)
full_cov = TimeSeries.from_dataframe(
    df,
    time_col="FECHA",
    value_cols=feature_cols,
)

# Scale them in the SAME way used in training
series_scaled = scaler_y.transform(full_series)
cov_scaled = scaler_cov.transform(full_cov)

window_size = window_size  # already defined
test_size   = test_size
F = 1 + len(feature_cols)  # target + all covariates


# index of the last prediction in the validation horizon
last_idx = len(df) - 1        # last row in df
start_hist = last_idx - test_size + 1 - window_size
end_hist   = last_idx - test_size + 1    # exclusive

# Sanity
assert start_hist >= 0

# Get the scaled window used as history: shape [window_size, 1] for target
start_time = df["FECHA"].iloc[start_hist]
end_time   = df["FECHA"].iloc[start_hist + window_size - 1]

hist_target = series_scaled.slice(start_time, end_time)
hist_cov    = cov_scaled.slice(start_time, end_time)

# Convert to numpy arrays
y_hist_np = hist_target.values(copy=True)        # (window_size, 1)
cov_hist_np = hist_cov.values(copy=True)        # (window_size, len(feature_cols))

# Concatenate along features -> (window_size, F)
hist_full_np = np.concatenate([y_hist_np, cov_hist_np], axis=1)

# Flatten to 1D: (window_size * F,)
x0 = hist_full_np.flatten()

# Build feature names: col_t0 is the most recent, col_t-(window_size-1) oldest
feat_names = []
cols_all = [target_col] + feature_cols
for t in range(window_size):
    offset = window_size - 1 - t  # 0 is oldest, window_size-1 is most recent
    for c in cols_all:
        feat_names.append(f"{c}_t-{offset}")


# Toy "training data" for LIME explainer: jitter x0 a bit
X_lime = np.tile(x0, (200, 1))
noise = np.random.normal(scale=0.01, size=X_lime.shape)
X_lime = X_lime + noise

explainer = LimeTabularExplainer(
    X_lime,
    feature_names=feat_names,
    mode="regression",
    verbose=False
)


def predict_fn(flat_X):
    """
    flat_X: numpy array [n_samples, window_size * F]
    Returns: [n_samples] with the value of the LAST forecast step.
    """
    flat_X = np.array(flat_X)
    n_samples = flat_X.shape[0]
    preds = []

    # time index for the window we are explaining
    time_idx = pd.DatetimeIndex(df["FECHA"].iloc[start_hist:start_hist + window_size])

    for i in range(n_samples):
        arr = flat_X[i].reshape(window_size, F)   # (T, F)

        # Split back into target + cov
        y_win   = arr[:, :1]                # (T, 1)
        cov_win = arr[:, 1:]                # (T, len(feature_cols))

        # build TimeSeries from this window alone
        y_win_ts   = y_win.squeeze()        # (T,)
        cov_win_ts = cov_win                # (T, n_cov)

        ts_y = TimeSeries.from_times_and_values(time_idx, y_win_ts)
        ts_cov = TimeSeries.from_times_and_values(time_idx, cov_win_ts)

        # directly predict from this window
        pred_scaled = model.predict(
            n=test_size,
            series=ts_y,
            past_covariates=ts_cov,
            dataloader_kwargs={"num_workers": 0},
        )
        pred_unscaled = scaler_y.inverse_transform(pred_scaled)

        # take last forecast step
        preds.append(pred_unscaled.values().flatten()[-1])

    return np.array(preds)


exp = explainer.explain_instance(
    x0,
    predict_fn,
    num_features=20  # top features
)

print(exp.as_list())   # text explanation

# Or a plot
fig = exp.as_pyplot_figure()
plt.tight_layout()
plt.savefig(f"plots/tft_lime_last_point_{target_col}.png", dpi=200)






# =======================
# SHAP (KernelExplainer)
# =======================

# Use the same flattened representation and predict_fn as LIME
# X_lime is already a cloud around x0, good as background
background = X_lime[:50]   # keep small for speed

# Create SHAP explainer
explainer_shap = shap.KernelExplainer(
    predict_fn,  # takes [n_samples, window_size * F] -> [n_samples]
    background
)

x0_2d = x0.reshape(1, -1)  # (1, window_size * F)

shap_values = explainer_shap.shap_values(x0_2d, nsamples=50)

# 1) Global-style summary for that point (bar plot)
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    features=x0.reshape(1, -1),
    feature_names=feat_names,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig(f"plots/tft_shap_bar_last_point_{target_col}.png", dpi=200)
plt.close()

# 2) Detailed summary plot (beeswarm-style for this single instance)
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    features=x0.reshape(1, -1),
    feature_names=feat_names,
    show=False
)
plt.tight_layout()
plt.savefig(f"plots/tft_shap_beeswarm_last_point_{target_col}.png", dpi=200)
plt.close()



# Use a small batch from your jittered data as background + evaluation set
X_batch = X_lime[:100]  # shape (100, n_features)

explainer_shap = shap.KernelExplainer(predict_fn, X_batch[:50])
shap_vals = explainer_shap.shap_values(X_batch, nsamples=50)

if isinstance(shap_vals, list):
    shap_vals = shap_vals[0]   # (n_samples, n_features)

from shap.utils import approximate_interactions

# choose a feature index to analyze interactions for
feat_idx = 0  # e.g., first feature; change as needed

interaction_ranking = approximate_interactions(feat_idx, shap_vals, X_batch)
# interaction_ranking is a list of feature indices sorted by interaction strength

print("Top interacting features with", feat_names[feat_idx])
for j in interaction_ranking[:10]:
    print(f"  {feat_names[j]}")




import matplotlib.pyplot as plt
from shap.utils import approximate_interactions

# Pick a feature index to analyze (example: the most important feature)
feat_idx = np.argmax(np.abs(shap_vals).mean(axis=0))

# Compute interaction scores for this feature
interaction_ranking = approximate_interactions(feat_idx, shap_vals, X_batch)

# Take the top K interacting features
K = 10
top_idx = interaction_ranking[:K]
top_scores = [np.abs(np.corrcoef(shap_vals[:, feat_idx], shap_vals[:, j])[0,1]) for j in top_idx]

# Build labels
labels = [feat_names[j] for j in top_idx]

# ---- Plot ----
plt.figure(figsize=(10, 5))
plt.barh(labels, top_scores)
plt.gca().invert_yaxis()
plt.xlabel("Approx. interaction strength")
plt.title(f"Top {K} interactions with {feat_names[feat_idx]}")
plt.tight_layout()
plt.savefig(f"plots/shap_interaction_bar_{target_col}.png", dpi=200)
plt.close()
