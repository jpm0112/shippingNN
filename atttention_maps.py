# --- SINGLE RUN VERSION ---
import warnings
warnings.filterwarnings("ignore")
from lightning.pytorch import Trainer
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.data.encoders import GroupNormalizer
import pandas as pd
import torch
import numpy as np
from datetime import datetime
from functions import error_metrics
import os

torch.set_float32_matmul_precision("high")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load data
df = pd.read_csv("data/test_final_kz.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"] + "-5", format="%Y-%W-%w")
df = df.sort_values("FECHA")
df = df.rename(columns=lambda x: x.replace(".", "_"))

# --- single model parameters ---
target_col   = "XSICFENE Index  (R1)"
window_size  = 8
test_size    = 4
d_model      = 8
n_head       = 2
num_layers   = 1
epoch_number = 50
lr           = 0.01
batch_size   = 16
dropout      = 0.5
grad_clip    = 1
seed         = 1048596
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

train_cut = tmp["time_idx"].max() - test_size

# --- TFT function ---
def run_tft(data, target_col, window_size, test_size,
            d_model, n_head, num_layers, epoch_number, lr, batch_size, seed):

    feature_cols = [c for c in data.columns if c not in ["FECHA", target_col, "time_idx", "series"]]
    time_varying_known_reals = ["time_idx", "dow", "month"]
    time_varying_unknown_reals = [target_col] + feature_cols

    training = TimeSeriesDataSet(
        data[data.time_idx <= train_cut],
        time_idx="time_idx",
        target=target_col,
        group_ids=["series"],
        max_encoder_length=window_size,
        min_encoder_length=window_size,
        max_prediction_length=1,
        min_prediction_length=1,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        static_categoricals=["series"],
        target_normalizer=GroupNormalizer(groups=["series"]),
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, data, predict=True, stop_randomization=True, min_prediction_idx=train_cut + 1
    )

    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader   = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=lr,
        hidden_size=d_model,
        attention_head_size=n_head,
        lstm_layers=num_layers,
        dropout=dropout,
        loss=MAE(),
        output_size=1,
        reduce_on_plateau_patience=3,
    )

    trainer = Trainer(
        max_epochs=epoch_number,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
        enable_checkpointing=False,
        enable_model_summary=False,
        gradient_clip_val=grad_clip,
    )

    trainer.fit(tft, train_loader, val_loader)
    preds = tft.predict(val_loader).squeeze(-1).cpu().numpy()

    y_true = []
    for x, y in val_loader:
        if isinstance(y, (tuple, list)):
            y = y[0]
        y_true.append(y.detach().cpu().numpy())
    y_true = np.concatenate(y_true).reshape(-1)

    preds = preds[-test_size:]
    y_true = y_true[-test_size:]

    return y_true, preds, tft, val_loader, training


# --- Run single model ---
start = datetime.now()
y_true, y_pred, tft, val_loader, training = run_tft(
    tmp, target_col, window_size, test_size,
    d_model, n_head, num_layers, epoch_number, lr, batch_size, seed
)
mae, mape, mse, rmse, r2 = error_metrics(y_true, y_pred)
runtime = (datetime.now() - start).total_seconds()

print(f"MAE={mae:.3f}, MAPE={mape:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}, Time={runtime:.1f}s")



# after training
raw_pred, x, *_ = tft.predict(val_loader, return_x=True, mode="raw")
interpret = tft.interpret_output(raw_pred, reduction="none")
A = np.asarray(interpret["attention"])           # attention

print("attention shape:", A.shape)

# Average to "lag importance" (length = enc_len)
if A.ndim == 4:          # [B, dec, enc, heads]
    attn_mean_lag = A.mean(axis=(0,1,3))
elif A.ndim == 3:        # [B, dec, enc]
    attn_mean_lag = A.mean(axis=(0,1))
elif A.ndim == 2:        # [dec, enc]
    attn_mean_lag = A.mean(axis=0)
else:
    raise ValueError(f"Unexpected attention ndim={A.ndim}")

# Quick plots
import matplotlib.pyplot as plt

# heatmap for a single example
if A.ndim == 4:
    heat = A[0].mean(axis=-1)          # avg heads -> [dec, enc]
elif A.ndim == 3:
    heat = A[0]                        # [dec, enc]
else:
    heat = A                           # [dec, enc]

plt.figure(); plt.imshow(heat, aspect="auto", origin="lower")
plt.colorbar(label="attention"); plt.xlabel("encoder steps"); plt.ylabel("decoder step")
plt.title("TFT attention"); plt.show()

# if single-step forecast, also show line over lags
if heat.shape[0] == 1:
    plt.figure(); plt.plot(attn_mean_lag)
    plt.xlabel("encoder step (old→new)"); plt.ylabel("avg attention")
    plt.title("Average attention over encoder lags"); plt.show()

