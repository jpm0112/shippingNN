import pandas as pd
from functions import error_metrics, run_darts_tft
import matplotlib.pyplot as plt
from lightning.pytorch import  seed_everything
import torch



# ==== 1. Load your data ====
df = pd.read_csv("weekly_chile_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"])
df = df.sort_values("FECHA").reset_index(drop=True)





target_col = "NE"

test_size = 12 # number of weeks to forecast
window_size = 26
hidden_size = 64
lstm_layers = 1
num_attention_heads = 1
dropout = 0.184845678
batch_size = 32
lr = 0.000323253143048712
n_epochs = 2000
grad_clip = 3


patience = 100
min_delta = 1e-5

seed = 1048596

# ==== 2. Run Darts TFT ====

deleted_sample = test_size  # delete the test samples from the end
tmp = df.copy().sort_values("FECHA")
if deleted_sample > 0:
        tmp = tmp.iloc[:-deleted_sample]


true_vals, pred_vals, list = run_darts_tft(tmp,target_col,test_size,window_size,hidden_size,lstm_layers,num_attention_heads,dropout,
        batch_size,n_epochs,lr,grad_clip,patience=patience,min_delta=min_delta,seed=seed)

mae, mape_v, mse, rmse, r2 = error_metrics(true_vals, pred_vals)

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




