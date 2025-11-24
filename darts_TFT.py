import pandas as pd
from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape

from functions import error_metrics
import matplotlib.pyplot as plt

import pytorch_lightning as pl  # <-- NEW

# ==== 1. Load your data ====
df = pd.read_csv("weekly_chile_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"])
df = df.sort_values("FECHA").reset_index(drop=True)

target_col = "FE"
df["t"] = df.index


# df = df.drop(columns=["FECHA"])
# df = df.drop(columns=["time_idx"])

series = TimeSeries.from_dataframe(
    df,
    time_col="FECHA",
    value_cols=target_col
)


# ==== 2. Train / test split ====
test_size = 24
input_chunk_length = 48  # lookback window
output_chunk_length = test_size  # forecast horizon
val_size = test_size + input_chunk_length

train = series[:-val_size]
val = series[-val_size:]

# ==== 3. (Optional) Scale ====
scaler = Scaler()
train_scaled = scaler.fit_transform(train)
val_scaled = scaler.transform(val)

# ==== 3.5 Early stopping callback ====
early_stop = pl.callbacks.EarlyStopping(
    monitor="val_loss",  # what to watch
    patience=20,  # epochs with no improvement
    min_delta=1e-4,  # minimum improvement to reset patience
    mode="min"
)

# ==== 4. Define TFT model ====


model = TFTModel(
    input_chunk_length=input_chunk_length,
    output_chunk_length=output_chunk_length,
    hidden_size=128,
    lstm_layers=1,
    num_attention_heads=4,
    dropout=0.1,
    batch_size=32,
    n_epochs=500,  # max, early stopping will cut this
    add_relative_index=True,
    add_encoders={
        "datetime_attribute": {
            "past": ["weekofyear"],
            "future": ["weekofyear"],
        },
        "cyclic": {
            "past": ["weekofyear"],
            "future": ["weekofyear"],
        },
    },
    random_state=42,
    likelihood=None,
    pl_trainer_kwargs={
        "accelerator": "auto",
        "callbacks": [early_stop],  # <-- pass callback
    },
)



# ==== 5. Fit ====
model.fit(train_scaled, val_series=val_scaled, verbose=True)

# ==== GET EPOCHS RAN ====
epochs_ran = model.trainer.current_epoch + 1
print("Epochs effectively run:", epochs_ran)

# ==== 6. Forecast ====
n = test_size
pred_scaled = model.predict(n=n)
pred = scaler.inverse_transform(pred_scaled)

val = val[-n:]


true_vals = val.values().flatten().tolist()
pred_vals = pred.values().flatten().tolist()


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
