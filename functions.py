import numpy as np
import torch
from lightning.pytorch import Trainer, seed_everything
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE, QuantileLoss, SMAPE
from pytorch_forecasting.data.encoders import GroupNormalizer

def error_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    # Basic errors
    err = y_true - y_pred
    mse = np.mean(err ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(err))

    # R² with constant-target handling
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        r2 = 1.0 if mse == 0 else 0.0   # sklearn convention for constant y_true
    else:
        r2 = 1 - ss_res / ss_tot

    # MAPE that ignores zeros in y_true (or use epsilon if you prefer)
    nonzero = y_true != 0
    if nonzero.any():
        mape = np.mean(np.abs(err[nonzero] / y_true[nonzero])) * 100
    else:
        mape = np.nan  # undefined if all y_true are zero

    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²: {r2:.4f}")
    return(mae,mape,mse,rmse,r2)


def run_tft(data, target_col, window_size, test_size, grad_clip,
            d_model, n_head, num_layers, epoch_number, lr, batch_size, seed, train_cut):
    seed_everything(seed)
    feature_cols = [c for c in data.columns if c not in ["FECHA", target_col, "time_idx", "series"]]
    # time_varying_known_reals = ["time_idx", "dow", "month"]
    time_varying_known_reals = ["time_idx"]
    time_varying_unknown_reals = [target_col] + feature_cols

    q = [0.1, 0.5, 0.9]  # choose your quantiles
    output_size = len(q)
    loss = QuantileLoss(quantiles=q)

    # loss = SMAPE()
    # output_size = 1

    training = TimeSeriesDataSet(
        data[data.time_idx <= train_cut],  # only the training slice (no future leakage)
        time_idx="time_idx",  # column representing the time ordering (0,1,2,...)
        target=target_col,  # the variable you want to forecast
        group_ids=["series"],  # identifies each time series (you have one: "kz")
        max_encoder_length=window_size,  # how many past steps the model sees as input
        min_encoder_length=window_size,  # force this length (no variable window)
        max_prediction_length=test_size,  # how many future steps to predict
        min_prediction_length=test_size,  # force prediction length
        time_varying_known_reals=time_varying_known_reals,
        # features known for all time (e.g., time index, calendar info)
        time_varying_unknown_reals=time_varying_unknown_reals,
        # features that are only known up to “now” (target and lagged features)
        static_categoricals=["series"],  # series label — constant across time
        target_normalizer=GroupNormalizer(groups=["series"]),  # normalize per series (mean/std or quantiles)
        allow_missing_timesteps=True,  # let the dataset handle gaps in time_idx
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, data, predict=True, stop_randomization=True, min_prediction_idx=train_cut + 1
    )

    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=lr,
        hidden_size=d_model,
        attention_head_size=n_head,
        lstm_layers=num_layers,
        dropout=0.3,
        loss = loss,
        output_size=output_size,
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
    preds = tft.predict(val_loader, mode="prediction").cpu().numpy()
    preds = preds[0]






    y_true = []
    for x, y in val_loader:
        if isinstance(y, (tuple, list)):
            y = y[0]
        y_true.append(y.detach().cpu().numpy())
    y_true = np.concatenate(y_true).reshape(-1)



    # preds = preds[-test_size:]
    # y_true = y_true[-test_size:]

    return y_true, preds, tft, val_loader, training
