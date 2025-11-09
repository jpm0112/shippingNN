import os
import warnings


import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from lightning.pytorch import Trainer
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
from pytorch_forecasting.metrics import MAPE

warnings.filterwarnings("ignore")
os.chdir(os.getcwd())
# Initialize missing variables
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
deleted_sample = None  # Set this to your actual value if needed


# Define error_metrics function
def error_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2


# Initialize results dataframe
results_df = pd.DataFrame(
    columns=[
        "seed",
        "deleted_sample",
        "window_size",
        "test_size",
        "d_model",
        "n_head",
        "num_layers",
        "epoch_number",
        "target_col",
        "lr",
        "mae",
        "mse",
        "rmse",
        "r2",
    ]
)

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
root_dir = os.path.dirname(script_dir)

# Build time features once (outside loops if you want)
df = pd.read_csv(os.path.join(root_dir, "proc", "test_final_kz.csv"))
df["FECHA"] = pd.to_datetime(df["FECHA"] + "-5", format="%Y-%W-%w")
df = df.sort_values("FECHA")
# clean column names (replace "." with "_")
df = df.rename(columns=lambda x: x.replace(".", "_"))
# replace spaces with underscores in column names
df.columns = df.columns.str.replace(" ", "_", regex=False)
tmp = df.copy()
tmp = df.sort_values("FECHA").copy()
tmp["series"] = "kz"  # or your series id
tmp["time_idx"] = tmp.groupby("series").cumcount()
# simple known-future calendar features (always available)
tmp["dow"] = tmp["FECHA"].dt.weekday.astype(int)
tmp["month"] = tmp["FECHA"].dt.month.astype(int)

target_col = "XSICFEUW_Index__(R4)"  # Fixed column name with underscores
window_size = 5  # one year context
test_size = 2  # ~half year test
train_cut = tmp["time_idx"].max() - test_size
d_model = 64  # hidden size
n_head = 2  # attention heads
num_layers = 1  # lstm layers
epoch_number = 100  # train epochs
lr = 1e-3  # learning rate
batch_size = 8  # batch size
seed = 1048596


def run_tft(
    data,
    target_col,
    window_size,
    test_size,
    d_model,
    n_head,
    num_layers,
    epoch_number,
    lr,
    batch_size,
    seed,
):
    pl.seed_everything(seed)

    # split by time
    max_time = data["time_idx"].max()
    train_cutoff = max_time - test_size

    # observed covariates = your other columns (no need to pre-scale)
    feature_cols = [
        c
        for c in data.columns
        if c not in ["FECHA", target_col, "time_idx", "series", "dow", "month"]
    ]
    time_varying_known_reals = ["time_idx", "dow", "month"]  # known into the future
    time_varying_unknown_reals = [
        target_col
    ] + feature_cols  # observed only historically

    print(f"Target column: {target_col}")
    print(f"Feature columns: {feature_cols}")
    print(f"Data shape: {data.shape}")

    training = TimeSeriesDataSet(
        data[data.time_idx <= train_cutoff],
        time_idx="time_idx",
        target=target_col,
        group_ids=["series"],
        max_encoder_length=window_size,
        min_encoder_length=window_size,  # ensure full window
        max_prediction_length=1,
        min_prediction_length=1,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        static_categoricals=["series"],
        target_normalizer=GroupNormalizer(groups=["series"]),
        allow_missing_timesteps=True,
    )

    # start validation AFTER the training cutoff
    validation = TimeSeriesDataSet.from_dataset(
        training,
        data,
        predict=True,
        stop_randomization=True,
        min_prediction_idx=train_cutoff + 1,
    )

    train_loader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_loader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0
    )

    # Model (point forecast using MSE; set QuantileLoss for probabilistic)
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=lr,
        hidden_size=d_model,
        attention_head_size=n_head,
        lstm_layers=num_layers,
        dropout=0.1,
        loss=MAPE(),  # or QuantileLoss() with output_size>1
        output_size=1,
        reduce_on_plateau_patience=3,
    )

    # FIX: Check accelerator availability
    if torch.cuda.is_available():
        accelerator = "gpu"
    elif torch.backends.mps.is_available():
        accelerator = "mps"
    else:
        accelerator = "cpu"

    # FIX: Use compatible trainer settings
    trainer = Trainer(
        max_epochs=epoch_number,
        accelerator=accelerator,
        devices=1,
        log_every_n_steps=10,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,  # Add progress bar
        gradient_clip_val=0.1,  # Add gradient clipping for stability
    )

    # FIX: Ensure model is properly initialized as LightningModule
    # The TFT from pytorch-forecasting should already be a LightningModule
    # If not, check version compatibility
    print(f"Model type: {type(tft)}")
    print(f"Is LightningModule: {isinstance(tft, pl.LightningModule)}")

    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Predict last test_size steps (1-step-ahead rolling from validation set)
    predictions = tft.predict(val_loader, return_y=True, trainer=trainer)

    # Extract predictions and true values
    y_pred = predictions.output.squeeze(-1).cpu().numpy()
    y_true = predictions.y[0].squeeze(-1).cpu().numpy()

    # Keep only the last `test_size` 1-step predictions (matches your prior eval)
    preds = y_pred[-test_size:]
    y_true = y_true[-test_size:]

    return y_true, preds


# ----- use inside your innermost loop, replacing your TransformerForecast block -----
try:
    y_test_true, y_test_pred = run_tft(
        data=tmp,
        target_col=target_col,
        window_size=window_size,
        test_size=test_size,
        d_model=d_model,
        n_head=n_head,
        num_layers=num_layers,
        epoch_number=epoch_number,
        lr=lr,
        batch_size=batch_size,
        seed=seed,
    )

    mae, mse, rmse, r2 = error_metrics(y_test_true, y_test_pred)
    results_df.loc[len(results_df)] = [
        seed,
        deleted_sample,
        window_size,
        test_size,
        d_model,
        n_head,
        num_layers,
        epoch_number,
        target_col,
        lr,
        mae,
        mse,
        rmse,
        r2,
    ]
    results_df.to_csv(
        os.path.join("results", f"model_results_{timestamp}.csv"), index=False
    )

    print("\nResults:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

except Exception as e:
    print(f"Error occurred: {e}")
    import traceback

    traceback.print_exc()
