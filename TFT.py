# --- NEW: Temporal Fusion Transformer with pytorch-forecasting ---
import warnings
warnings.filterwarnings("ignore")
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAPE
from pytorch_forecasting.data.encoders import GroupNormalizer
import pandas as pd
import torch

# Build time features once (outside loops if you want)
df = pd.read_csv("test_final_kz.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"] + "-5", format="%Y-%W-%w")
df = df.sort_values('FECHA')
# clean column names (replace "." with "_")
df = df.rename(columns=lambda x: x.replace(".", "_"))


tmp = df.copy()
tmp = df.sort_values("FECHA").copy()
tmp["series"] = "kz"  # or your series id
tmp["time_idx"] = tmp.groupby("series").cumcount()
# simple known-future calendar features (always available)
tmp["dow"] = tmp["FECHA"].dt.weekday.astype(int)
tmp["month"] = tmp["FECHA"].dt.month.astype(int)

target_col = "XSICFEUW Index  (R4)"
window_size  = 5     # one year context
test_size    = 2     # ~half year test
train_cut = tmp["time_idx"].max() - test_size
d_model      = 64     # hidden size
n_head       = 2      # attention heads
num_layers   = 1      # lstm layers
epoch_number = 100    # train epochs
lr           = 1e-3   # learning rate
batch_size   = 8      # batch size
seed         = 1048596

def run_tft(
    data, target_col, window_size, test_size,
    d_model, n_head, num_layers, epoch_number, lr, batch_size, seed
):
    pl.seed_everything(seed)

    # split by time
    max_time = data["time_idx"].max()
    train_cutoff = max_time - test_size

    # observed covariates = your other columns (no need to pre-scale)
    feature_cols = [c for c in data.columns if c not in ["FECHA", target_col, "time_idx", "series"]]
    time_varying_known_reals = ["time_idx", "dow", "month"]          # known into the future
    time_varying_unknown_reals = [target_col] + feature_cols         # observed only historically

    print(data)


    training = TimeSeriesDataSet(
        tmp[tmp.time_idx <= train_cut],
        time_idx="time_idx",
        target=target_col,
        group_ids=["series"],
        max_encoder_length=window_size,
        min_encoder_length=window_size,   # ensure full window
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
        training, tmp, predict=True, stop_randomization=True, min_prediction_idx=train_cut + 1
    )





    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader   = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    # Model (point forecast using MSE; set QuantileLoss for probabilistic)
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=lr,
        hidden_size=d_model,
        attention_head_size=n_head,
        lstm_layers=num_layers,
        dropout=0.1,
        loss=MAPE(),          # or QuantileLoss() with output_size>1
        output_size=1,
        reduce_on_plateau_patience=3,
    )

    trainer = pl.Trainer(
        max_epochs=epoch_number,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    trainer.fit(tft, train_loader, val_loader)

    # Predict last test_size steps (1-step-ahead rolling from validation set)
    preds = tft.predict(val_loader, trainer=trainer).squeeze(-1).cpu().numpy()

    # True values aligned with preds:
    y_true = []
    for batch in iter(val_loader):
        # batch[1] is target in pytorch-forecasting dataloader
        y_true.append(batch[1].cpu().numpy())
    y_true = np.concatenate(y_true).reshape(-1)

    # Keep only the last `test_size` 1-step predictions (matches your prior eval)
    preds = preds[-test_size:]
    y_true = y_true[-test_size:]

    return y_true, preds

# ----- use inside your innermost loop, replacing your TransformerForecast block -----
y_test_true, y_test_pred = run_tft(
    data=tmp, target_col=target_col, window_size=window_size, test_size=test_size,
    d_model=d_model, n_head=n_head, num_layers=num_layers,
    epoch_number=epoch_number, lr=lr, batch_size=batch_size, seed=seed
)

mae, mse, rmse, r2 = error_metrics(y_test_true, y_test_pred)
results_df.loc[len(results_df)] = [
    seed, deleted_sample, window_size, test_size, d_model, n_head, num_layers, epoch_number, target_col, lr,
    mae, mse, rmse, r2
]
results_df.to_csv(os.path.join("results", f"model_results_{timestamp}.csv"), index=False)
