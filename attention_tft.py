
from darts.explainability import TFTExplainer

import pandas as pd
from functions import error_metrics, run_darts_tft
import matplotlib.pyplot as plt
from lightning.pytorch import seed_everything
import torch

# ==== 1. Load your data ====
df = pd.read_csv("weekly_chile_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"])
df = df.sort_values("FECHA")



target_col = "FE"


#suppose to be 6.82% mape
test_size = 12  # number of weeks to forecast
window_size = 40
hidden_size = 128
lstm_layers = 3
num_attention_heads = 4
dropout = 0.313791
batch_size = 64
lr = 0.000395
n_epochs = 2000
grad_clip = 1.0

patience = 100
min_delta = 1e-5

seed = 1048596

# ==== 2. Run Darts TFT ====

tmp = df.copy().sort_values("FECHA")
deleted_sample = test_size  # delete the test samples from the end
if deleted_sample > 0:
    tmp = tmp.iloc[:-deleted_sample]

true_vals, pred_vals, out = run_darts_tft(tmp, target_col, test_size, window_size, hidden_size, lstm_layers,
                                           num_attention_heads, dropout,
                                           batch_size, n_epochs, lr, grad_clip, patience=patience, min_delta=min_delta,
                                           seed=seed)

mae, mape, mse, rmse, r2 = error_metrics(true_vals, pred_vals)

plt.figure(figsize=(12, 6))
plt.plot(true_vals, marker="o", label="Real")
plt.plot(pred_vals, marker="o", label="Prediction")
plt.title("Prediction (original scale)")
plt.xlabel("Weeks")
plt.ylabel(target_col)
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("plots/z_darts_prediction.png", dpi=200)

model = out[0]


out

explainer = TFTExplainer(model)

explainer_results = explainer.explain()

explainer.plot_attention(explainer_results, plot_type="time")  # avg over horizons
explainer.plot_attention(explainer_results, plot_type="all")  # per horizon
explainer.plot_attention(explainer_results, plot_type="heatmap")
explainer.plot_variable_selection(explainer_results)

from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt





stl = STL(df[target_col], period=52)  # weekly data → annual seasonality
res = stl.fit()

res.plot()
plt.show()

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(df[target_col], lags=100)
plt.show()



# --- attention (TimeSeries or list[TimeSeries]) ---
attention_ts = explainer_results.get_attention()  # encoder+decoder attention
encoder_imp = explainer_results.get_encoder_importance()  # pd.DataFrame or list[pd.DataFrame]
decoder_imp = explainer_results.get_decoder_importance()
feat_imps = explainer_results.get_feature_importances()  # dict of DataFrames
static_imp = explainer_results.get_static_covariates_importance()

var_imp = explainer_results.get_feature_importances()

# --- encoder (past target + past covs + historic future covs) ---
enc = var_imp.get("encoder_importance")
if isinstance(enc, list):  # if you explained multiple series
    enc = enc[0]

# --- decoder (future part of future covs) ---
dec = var_imp.get("decoder_importance")
if isinstance(dec, list):
    dec = dec[0]

# --- static covariates ---
stat = var_imp.get("static_covariates_importance")
if isinstance(stat, list):
    stat = stat[0]

# Some of these can be None if you didn't use that type
n_encoder = 0 if enc is None else enc.shape[1]
n_decoder = 0 if dec is None else dec.shape[1]
n_static = 0 if stat is None else stat.shape[1]

print("Encoder vars:", n_encoder)
print("Decoder vars:", n_decoder)
print("Static vars:", n_static)
print("Total unique vars:",
      len(
          set(
              ([] if enc is None else enc.columns.tolist())
              + ([] if dec is None else dec.columns.tolist())
              + ([] if stat is None else stat.columns.tolist())
          )
      ))



# 1) Get encoder/decoder importance as DataFrames
enc_imp = explainer_results.get_encoder_importance()
dec_imp = explainer_results.get_decoder_importance()
stat_imp = explainer_results.get_static_covariates_importance()

# If you explained multiple series, you get a list -> take the first one
if isinstance(enc_imp, list):
    enc_imp = enc_imp[0]
if isinstance(dec_imp, list):
    dec_imp = dec_imp[0]
if isinstance(stat_imp, list):
    stat_imp = stat_imp[0]

print(enc_imp.shape)  # (time, n_encoder_features)
print(dec_imp.shape)  # (time, n_decoder_features)

# 2) Aggregate over time -> one weight per variable
enc_mean = enc_imp.mean(axis=0).sort_values(ascending=False)
dec_mean = dec_imp.mean(axis=0).sort_values(ascending=False)

enc_df = enc_mean.reset_index()
enc_df.columns = ["feature", "importance"]
dec_df = dec_mean.reset_index()
dec_df.columns = ["feature", "importance"]

print("Encoder variable selection weights:")
print(enc_df.head(20))

print("Decoder variable selection weights:")
print(dec_df.head(20))