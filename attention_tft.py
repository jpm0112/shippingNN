
from darts.explainability import TFTExplainer

import pandas as pd
from functions import error_metrics, run_darts_tft
import matplotlib.pyplot as plt
from lightning.pytorch import seed_everything
import torch

# ==== 1. Load your data ====
df = pd.read_csv("weekly_chile_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"])
df = df.sort_values("FECHA").reset_index(drop=True)

target_col = "FE"

test_size = 12  # number of weeks to forecast
window_size = 24
hidden_size = 128
lstm_layers = 2
num_attention_heads = 2
dropout = 0.3
batch_size = 32
lr = 0.000920696018319848
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

true_vals, pred_vals, out = run_darts_tft(tmp, target_col, test_size, window_size, hidden_size, lstm_layers,
                                           num_attention_heads, dropout,
                                           batch_size, n_epochs, lr, grad_clip, patience=patience, min_delta=min_delta,
                                           seed=seed)

mae, mape, mse, rmse, r2 = error_metrics(true_vals, pred_vals)


model = out[0]

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








