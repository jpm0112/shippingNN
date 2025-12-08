import numpy as np
import torch, gc
from lightning.pytorch import Trainer, seed_everything
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE, QuantileLoss, SMAPE
from pytorch_forecasting.data.encoders import GroupNormalizer
import torch.nn.init as init
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl  # <-- NEW

from omegaconf import OmegaConf
import tft_torch
from tft_torch.tft import TemporalFusionTransformer
import tft_torch.loss as tft_loss
from tqdm.auto import tqdm  # progress bar

from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape
import copy


def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
        for names in m._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(m, name)
                n = bias.size(0)
                bias.data[:n // 3].fill_(-1.)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
class DictDataset(Dataset):
    """
    Simple dataset that holds a dict of numpy arrays and returns
    a dict of tensors for each sample.
    """
    def __init__(self, arrays_dict):
        self.arrays = {}
        for k, v in arrays_dict.items():
            if isinstance(v, np.ndarray):
                if np.issubdtype(v.dtype, np.floating):
                    self.arrays[k] = torch.from_numpy(v.astype(np.float32))
                elif np.issubdtype(v.dtype, np.integer):
                    self.arrays[k] = torch.from_numpy(v.astype(np.int64))
                else:
                    # Shouldn't really happen for data fed to the model
                    self.arrays[k] = torch.from_numpy(v)
            else:
                # Fallback, but usually everything here is ndarray
                v = np.asarray(v)
                if np.issubdtype(v.dtype, np.floating):
                    self.arrays[k] = torch.from_numpy(v.astype(np.float32))
                elif np.issubdtype(v.dtype, np.integer):
                    self.arrays[k] = torch.from_numpy(v.astype(np.int64))
                else:
                    self.arrays[k] = torch.from_numpy(v)

        self.keys = list(self.arrays.keys())

    def __len__(self):
        return self.arrays[self.keys[0]].shape[0]

    def __getitem__(self, idx):
        return {k: self.arrays[k][idx] for k in self.keys}
def run_tft_tftorch(
    data,
    target_col,
    window_size,
    test_size,
    grad_clip,
    d_model,
    n_head,
    num_layers,
    epoch_number,
    lr,
    batch_size,
    seed,
    train_cut
):
    """
    data: pandas DataFrame with columns:
        - 'time_idx' (int)
        - 'series'   (id of the time series)
        - target_col
        - 'dow', 'month'
        - other numeric features (optional)
    """

    # -------------------- 0. Setup & feature definitions --------------------
    seed_everything(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



    df = data.copy().sort_values(["series", "time_idx"])

    # same feature split as your pytorch_forecasting setup
    feature_cols = [
        c for c in df.columns
        if c not in ["FECHA", target_col, "time_idx", "series"]
    ]

    time_varying_known_reals = [c for c in ["time_idx", "dow", "month"] if c in df.columns]
    time_varying_unknown_reals = [target_col] + [
        c for c in feature_cols if c not in ["dow", "month"]
    ]

    history_len = window_size
    future_len = test_size

    # we will treat all time-varying reals as historical numerics,
    # and only "known" ones as future numerics
    hist_numeric_cols = time_varying_known_reals + time_varying_unknown_reals
    future_numeric_cols = time_varying_known_reals  # known in future

    # encode series as a categorical static feature
    df["series_id"] = df["series"].astype("category").cat.codes.astype(np.int32)
    n_series = df["series_id"].nunique()

    # -------------------- 1. Build sliding windows (train + val) --------------------
    # We mirror what TimeSeriesDataSet does conceptually:
    # - For each series, build all windows whose *last forecast step* <= train_cut  --> train
    # - For each series, build ONE window whose encoder ends at train_cut          --> val
    train_samples = []
    val_samples = []

    for sid, df_s in df.groupby("series_id"):
        df_s = df_s.sort_values("time_idx").reset_index(drop=True)
        times = df_s["time_idx"].to_numpy()
        n = len(df_s)

        if n < history_len + future_len:
            continue

        # ---------- TRAIN WINDOWS ----------
        # All windows where the last horizon step is <= train_cut
        # hist: [t, ..., t+history_len-1], fut: [t+history_len, ..., t+history_len+future_len-1]
        for start in range(0, n - history_len - future_len + 1):
            hist_start = start
            hist_end = start + history_len - 1
            fut_start = hist_end + 1
            fut_end = hist_end + future_len

            last_future_time = times[fut_end]
            if last_future_time > train_cut:
                # this window's forecast reaches into validation/test period -> skip for training
                continue

            hist_slice = slice(hist_start, hist_end + 1)
            fut_slice = slice(fut_start, fut_end + 1)
            hist_values = df_s.iloc[hist_slice][hist_numeric_cols].to_numpy(dtype=np.float32)
            fut_values = df_s.iloc[fut_slice][future_numeric_cols].to_numpy(dtype=np.float32)


            sample = {
                # meta (not used by the model but OK to have)
                "time_index": df_s.loc[hist_end, "time_idx"],  # this is fine, it's a single row
                # static
                "static_feats_numeric": np.zeros((0,), dtype=np.float32),
                "static_feats_categorical": np.array([sid], dtype=np.int32),
                # historical
                "historical_ts_numeric": hist_values,
                "historical_ts_categorical": np.zeros((history_len, 0), dtype=np.int32),
                # future (known reals only)
                "future_ts_numeric": fut_values,
                "future_ts_categorical": np.zeros((future_len, 0), dtype=np.int32),
                # target for all future steps
                "target": df_s.iloc[fut_slice][target_col].to_numpy(dtype=np.float32),
            }
            train_samples.append(sample)

        # ---------- VALIDATION WINDOW ----------
        # One window per series ending exactly at train_cut (if possible)
        idx_candidates = np.where(times == train_cut-1)[0]
        if len(idx_candidates) == 0:
            continue

        hist_end = idx_candidates[-1]
        hist_start = hist_end - history_len + 1
        fut_start = hist_end + 1
        fut_end = hist_end + future_len

        if hist_start < 0 or fut_end >= n:
            continue

        hist_slice = slice(hist_start, hist_end + 1)
        fut_slice = slice(fut_start, fut_end + 1)

        hist_values = df_s.iloc[hist_slice][hist_numeric_cols].to_numpy(dtype=np.float32)
        fut_values = df_s.iloc[fut_slice][future_numeric_cols].to_numpy(dtype=np.float32)

        val_sample = {
            "time_index": df_s.loc[hist_end, "time_idx"],
            "static_feats_numeric": np.zeros((0,), dtype=np.float32),
            "static_feats_categorical": np.array([sid], dtype=np.int32),
            "historical_ts_numeric": hist_values,
            "historical_ts_categorical": np.zeros((history_len, 0), dtype=np.int32),
            "future_ts_numeric": fut_values,
            "future_ts_categorical": np.zeros((future_len, 0), dtype=np.int32),
            "target": df_s.iloc[fut_slice][target_col].to_numpy(dtype=np.float32),
        }
        val_samples.append(val_sample)

    if len(train_samples) == 0 or len(val_samples) == 0:
        raise RuntimeError("No train/val samples were created – check time_idx/train_cut/window_size.")

    # -------------------- 2. Simple Dataset wrapper returning tensors --------------------
    class TFTDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            s = self.samples[idx]
            out = {}
            for k, v in s.items():
                if "categorical" in k:
                    out[k] = torch.as_tensor(v, dtype=torch.long)
                elif k == "time_index":
                    # keep as scalar tensor (not used by model)
                    out[k] = torch.as_tensor(v)
                else:
                    out[k] = torch.as_tensor(v, dtype=torch.float32)
            return out

    train_ds = TFTDataset(train_samples)
    val_ds = TFTDataset(val_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(train_ds)),
        shuffle=True,
        drop_last=False,
    )
    print("num train samples:", len(train_ds))
    print("num val samples:", len(val_ds))
    print("num train batches:", len(train_loader))

    val_loader = DataLoader(
        val_ds,
        batch_size=min(batch_size, len(val_ds)),
        shuffle=False,
        drop_last=False,
    )

    # -------------------- 3. Build TFT config --------------------
    q = [0.1, 0.5, 0.9]

    structure = {
        "num_historical_numeric": len(hist_numeric_cols),
        "num_historical_categorical": 0,
        "num_static_numeric": 0,
        "num_static_categorical": 1,
        "num_future_numeric": len(future_numeric_cols),
        "num_future_categorical": 0,
        "historical_categorical_cardinalities": [],
        "static_categorical_cardinalities": [n_series + 1],
        "future_categorical_cardinalities": [],
    }

    configuration = {
        "optimization": {
            "batch_size": {"training": batch_size, "inference": batch_size},
            "learning_rate": lr,
            "max_grad_norm": float(grad_clip) if grad_clip is not None else 0.0,
        },
        "model": {
            "dropout": 0.0,
            "state_size": d_model,
            "output_quantiles": q,
            "lstm_layers": num_layers,
            "attention_heads": n_head,
        },
        "task_type": "regression",
        "target_window_start": None,
        "data_props": structure,
    }

    config = OmegaConf.create(configuration)
    model = TemporalFusionTransformer(config=config)
    model.apply(weight_init)

    # # (optional but recommended) weight initialization from the tutorial
    # def weight_init(m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.xavier_normal_(m.weight.data)
    #         if m.bias is not None:
    #             nn.init.normal_(m.bias.data)
    #     elif isinstance(m, nn.LSTM):
    #         for param in m.parameters():
    #             if param.data.ndimension() >= 2:
    #                 nn.init.orthogonal_(param.data)
    #             else:
    #                 nn.init.normal_(param.data)

    # model.apply(weight_init)

    # -------------------- 4. Move to device & optimizer --------------------
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    quantiles_tensor = torch.tensor(q, dtype=torch.float32, device=device)

    # -------------------- 5. Training loop with metrics & progress bar --------------------
    def process_batch(batch, model):
        # move to device
        if is_cuda:
            for k in list(batch.keys()):
                batch[k] = batch[k].to(device)

        outputs = model(batch)
        labels = batch["target"]  # [B, future_len]
        predicted_quantiles = outputs["predicted_quantiles"]  # [B, future_len, Q]

        q_loss, q_risk, _ = tft_torch.loss.get_quantiles_loss_and_q_risk(
            outputs=predicted_quantiles,
            targets=labels,
            desired_quantiles=quantiles_tensor,
        )
        return q_loss, q_risk

    for epoch in range(epoch_number):
        # ----- TRAIN -----
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epoch_number} [train]"):
            optimizer.zero_grad()
            loss, _ = process_batch(batch, model)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        # ----- VALIDATION -----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                loss, _ = process_batch(batch, model)
                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses))

        print(f"Epoch {epoch+1}/{epoch_number} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    # -------------------- 6. Inference on validation set --------------------
    model.eval()
    with torch.no_grad():
        # assuming single series -> one sample in val_ds
        batch = next(iter(val_loader))
        print("sample targets:", batch["target"][0])
        if is_cuda:
            for k in list(batch.keys()):
                batch[k] = batch[k].to(device)
        outputs = model(batch)
        pred_all_q = outputs["predicted_quantiles"].cpu().numpy()  # [B, future_len, Q]
    sid0 = batch["static_feats_categorical"][0].item()
    # take sample 0, median quantile (index 1)
    y_pred = pred_all_q[0, :, 1]
    df_val = df[df["series_id"] == sid0].sort_values("time_idx").reset_index(drop=True)
    times  = df_val["time_idx"].to_numpy()
    idx_train_cut = np.where(times == train_cut)[0][-1]
    fut_start = idx_train_cut + 1
    fut_end   = idx_train_cut + future_len
    y_true = df_val.iloc[fut_start:fut_end+1][target_col].to_numpy()

    # y_true from original df (unscaled), same as in your pytorch_forecasting version
    # y_true = df.loc[
    #     (df["series_id"] == sid0) &
    #     (df.time_idx > train_cut) &
    #     (df.time_idx <= train_cut + future_len),
    #     target_col,
    # ].to_numpy()

    return y_true, y_pred, model, train_loader, val_loader, train_ds

# def select_device(prefer: str | None = None) -> torch.device:
#     """
#     Choose the best available device.
#     Priority: user preference (if available) > CUDA > MPS (Apple Silicon) > CPU.
#
#     Args:
#         prefer: Optional string preference: "cuda", "mps", or "cpu".
#                 If that backend isn't available, falls back automatically.
#
#     Returns:
#         torch.device
#     """
#     # Helper checks
#     cuda_ok = torch.cuda.is_available()
#     mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
#
#     # Respect user preference if possible
#     if prefer is not None:
#         pref = prefer.lower()
#         if pref == "cuda" and cuda_ok:
#             return torch.device("cuda")
#         if pref == "mps" and mps_ok:
#             return torch.device("mps")
#         if pref == "cpu":
#             return torch.device("cpu")
#         # If preferred isn't available, continue to fallbacks
#
#     # Auto selection
#     if cuda_ok:
#         return torch.device("cuda")
#     if mps_ok:
#         return torch.device("mps")
#     return torch.device("cpu")


def device_info(dev: torch.device) -> str:
    import torch
    """Nice human-readable summary."""
    if dev.type == "cuda":
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        cap = torch.cuda.get_device_capability(idx)
        return f"CUDA[{idx}] {name} (cc {cap[0]}.{cap[1]})"
    if dev.type == "mps":
        return "Apple Metal (MPS)"
    return "CPU"

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
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    from pytorch_forecasting.data import GroupNormalizer

    target_normalizer = GroupNormalizer(
        groups=["series"],
        method="robust",
        transformation="softplus",
    )



    seed_everything(seed)


    feature_cols = [c for c in data.columns
                    if c not in ["FECHA", target_col, "time_idx", "series"]]

    time_varying_known_reals = ["time_idx", "dow", "month"]
    time_varying_unknown_reals = [target_col] + [
        c for c in feature_cols if c not in ["dow", "month"]
    ]

    q = [0.1, 0.5, 0.9]

    loss = QuantileLoss(quantiles=q)
    output_size = len(q)
    lags = {
    target_col: [1, 2, 4, 8, 13, 26, 52],  # short-term and medium-term memory
    # you could also lag other real covariates if they’re in the time-varying lists
    # "some_other_feature": [1, 2, 3]
    }


    # ---------- TRAIN DATASET ----------
    training = TimeSeriesDataSet(
        data[data.time_idx <= train_cut],
        time_idx="time_idx",
        target=target_col,
        group_ids=["series"],
        max_encoder_length=window_size,
        min_encoder_length=8,
        max_prediction_length=test_size,
        min_prediction_length=test_size,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,

        static_categoricals=["series"],
        target_normalizer=target_normalizer,
        allow_missing_timesteps=True,
        add_relative_time_idx=True,
        add_target_scales=True,
        randomize_length=False,
        add_encoder_length=True,
        lags=lags,

    )
    validation = TimeSeriesDataSet.from_dataset(
    training,
    data,
    predict=False,
    # min_prediction_idx=train_cut + 1,
    stop_randomization=True
    )
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=15, persistent_workers=True)

    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=15, persistent_workers=True)

    # ---------- MODEL ----------
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=lr,
        hidden_size=d_model,
        attention_head_size=n_head,
        lstm_layers=num_layers,
        dropout=0.0,
        loss=loss,
        output_size=output_size,
        reduce_on_plateau_patience=3,
        hidden_continuous_size=8,
        share_single_variable_networks=True,
    )

    trainer = Trainer(
        max_epochs=epoch_number,
        accelerator="gpu" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
        enable_checkpointing=False,
        enable_model_summary=False,
        gradient_clip_val=grad_clip
    )

    trainer.fit(tft, train_loader, val_loader)

    # ---------- PREDICT ----------
    # shape: [1, test_size, len(q)]
    pred_all_q = tft.predict(
        val_loader,
        return_index=True,
        trainer_kwargs={"accelerator": "gpu", "devices": 1},
        mode="raw",
    )


    preds = pred_all_q.output.prediction[0,:,1].cpu().numpy()   # length = test_size

    #y_true directly from original df, unnormalized
    y_true = data.loc[
        (data.time_idx > train_cut) & (data.time_idx <= train_cut + test_size),
        target_col,
    ].to_numpy()

    return y_true, preds, tft, val_loader, training

# receives the data as a pandas dataframe as shown in the transformers.py file



def run_transformer2(df, target_col, window_size, test_size, batch_size, d_model, n_head, num_layers, epoch_number, lr,
                    device, seed, optimizer_type='adam',weight_decay=1e-4):

    seed_everything(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # device = select_device(device if isinstance(device, str) else None)
    print(f"Using device: {device_info(device)}")
    feature_cols = [col for col in df.columns if col not in ['FECHA', target_col, 'series']]

    # Split train/test
    train_df = df[:-test_size]
    test_df = df[-(test_size + window_size):]

    # Escalamiento manual
    train_features = train_df[feature_cols].values
    train_target = train_df[target_col].values
    min_vals = train_features.min(axis=0)
    max_vals = train_features.max(axis=0)
    target_min = train_target.min()
    target_max = train_target.max()

    scaled_train = (train_features - min_vals) / (max_vals - min_vals + 1e-8)
    scaled_target = (train_target - target_min) / (target_max - target_min + 1e-8)

    # Crear ventanas para entrenamiento
    X_train, y_train = [], []
    for i in range(len(train_df) - window_size):
        X_train.append(scaled_train[i:i + window_size])
        y_train.append(scaled_target[i + window_size])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Preparar test
    test_features = test_df[feature_cols].values
    test_target = test_df[target_col].values
    scaled_test = (test_features - min_vals) / (max_vals - min_vals + 1e-8)
    scaled_test_target = (test_target - target_min) / (target_max - target_min + 1e-8)

    X_test, y_test = [], []
    for i in range(test_size):
        X_test.append(scaled_test[i:i + window_size])
        y_test.append(scaled_test_target[i + window_size])

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Tensores
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

    # DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size, shuffle=False)

    # Transformer model

    class TransformerForecast(nn.Module):
        def __init__(self, input_size, d_model=d_model, nhead=n_head, num_layers=num_layers, max_len=window_size):
            super().__init__()
            self.input_linear = nn.Linear(input_size, d_model)

            # learned positional encodings
            self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.normal_(self.positional_encoding, std=0.02)

            # single (clean) definition
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, 1)

        def forward(self, x):
            x = self.input_linear(x)  # (B, T, D)
            # >>> add positions <<<
            x = x + self.positional_encoding[:, :x.size(1), :]
            x = self.transformer(x)
            out = x[:, -1, :]
            return self.fc(out)

    model = TransformerForecast(input_size=X_train.shape[2]).to(device)

    criterion = nn.MSELoss()
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    if optimizer_type == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)


    # Entrenamiento
    for epoch in range(epoch_number):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    # Evaluación
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test_tensor).squeeze().cpu().numpy()

    preds = preds_scaled * (target_max - target_min + 1e-8) + target_min
    real = np.array(y_test) * (target_max - target_min + 1e-8) + target_min

    return real, preds, [model, X_test, X_train, feature_cols]


def run_transformer(df, target_col, window_size, test_size, batch_size, d_model, n_head, num_layers, epoch_number, lr,
                    device, seed, optimizer_type='adam', weight_decay=1e-4, early_stop=True, patience=200, min_delta=1e-5):

    seed_everything(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Using device: {device_info(device)}")

    feature_cols = [col for col in df.columns if col not in ['FECHA', target_col, 'series']]

    # ===== Split train/test =====
    horizon = test_size
    val_size = test_size
    train_df = df[:-horizon]
    test_df = df[-horizon:]

    # ===== Manual scaling =====
    train_features = train_df[feature_cols].values
    train_target = train_df[target_col].values
    min_vals = train_features.min(axis=0)
    max_vals = train_features.max(axis=0)
    target_min = train_target.min()
    target_max = train_target.max()

    scaled_train = (train_features - min_vals) / (max_vals - min_vals + 1e-8)
    scaled_target = (train_target - target_min) / (target_max - target_min + 1e-8)

    # Make val windows
    split_idx = len(train_df) - val_size
    X_tr, y_tr = [], []
    for i in range(0, split_idx - window_size - horizon + 1):
        X_tr.append(scaled_train[i:i+window_size])
        y_tr.append(scaled_target[i+window_size:i+window_size+horizon])

    X_va, y_va = [], []
    # allow input window to reach slightly before split_idx, but targets must be in val region
    for i in range(split_idx - window_size, len(train_df) - window_size - horizon + 1):
        X_va.append(scaled_train[i:i+window_size])
        y_va.append(scaled_target[i+window_size:i+window_size+horizon])

    X_tr, y_tr = np.array(X_tr), np.array(y_tr)
    X_va, y_va = np.array(X_va), np.array(y_va)

    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32)
    ), batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(TensorDataset(
        torch.tensor(X_va, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32)
    ), batch_size=batch_size, shuffle=False)

    # ===== Model =====
    class TransformerForecast(nn.Module):
        def __init__(self, input_size, d_model=d_model, nhead=n_head,
                     num_layers=num_layers, max_len=window_size):
            super().__init__()
            self.input_linear = nn.Linear(input_size, d_model)

            # learned positional encodings
            self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.normal_(self.positional_encoding, std=0.02)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, horizon)

        def forward(self, x):
            x = self.input_linear(x)  # (B, T, D)
            x = x + self.positional_encoding[:, :x.size(1), :]
            x = self.transformer(x)
            out = x[:, -1, :]
            return self.fc(out)

    model = TransformerForecast(input_size=X_tr.shape[2]).to(device)

    criterion = nn.MSELoss()
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    if optimizer_type == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)

    import copy

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    # ===== Training =====
    for epoch in range(epoch_number):
        # ---- train epoch ----
        model.train()
        running_loss = 0.0
        n_samples = 0

        for batch_X, batch_y in train_loader:        # <- use train_loader
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            bs_actual = batch_X.size(0)
            running_loss += loss.item() * bs_actual
            n_samples += bs_actual

        train_loss = running_loss / max(n_samples, 1)
        epochs_run = epoch + 1

        # ---- validation epoch (THIS GOES HERE) ----
        model.eval()
        val_running = 0.0
        val_samples = 0
        with torch.no_grad():
            for val_X, val_y in val_loader:          # <- use val_loader
                val_X, val_y = val_X.to(device), val_y.to(device)
                val_out = model(val_X)
                vloss = criterion(val_out, val_y)

                bs = val_X.size(0)
                val_running += vloss.item() * bs
                val_samples += bs

        val_loss = val_running / max(val_samples, 1)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

        # ---- early stopping using VAL loss (THIS GOES HERE) ----
        if early_stop:
            if best_val_loss - val_loss > min_delta:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (best_val_loss={best_val_loss:.4f})")
                break

    # Restore best weights BEFORE final evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    # ===== Evaluation (true 24-ahead from ONE cutoff) =====
    X0 = scaled_train[-window_size:]  # (window_size, n_features)

    X0_tensor = torch.tensor(X0, dtype=torch.float32).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X0_tensor).squeeze(0).cpu().numpy()  # (horizon,)

    preds = preds_scaled * (target_max - target_min + 1e-8) + target_min
    real  = test_df[target_col].values  # (horizon,)

    # ---- compatibility outputs ----
    X_test_compat  = np.repeat(X0[np.newaxis, :, :], horizon, axis=0)  # (horizon, window_size, n_features)
    X_train_compat = X_tr                                           # (N_train_windows, window_size, n_features)

    return real, preds, [model, X_test_compat, X_train_compat, feature_cols, epochs_run]


def run_lstm2(df, target_col, window_size, test_size, batch_size, hidden_size, num_layers, epoch_number, lr, device, seed, patience, min_delta):


    seed_everything(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # device = select_device(device if isinstance(device, str) else None)
    print(f"Using device: {device_info(device)}")
    feature_cols = [col for col in df.columns if col not in ['FECHA', target_col, 'series']]
    # Split train/test
    train_df = df[:-test_size]
    test_df = df[-(test_size + window_size):]

    # Escalamiento manual
    train_features = train_df[feature_cols].values
    train_target = train_df[target_col].values
    min_vals = train_features.min(axis=0)
    max_vals = train_features.max(axis=0)
    target_min = train_target.min()
    target_max = train_target.max()

    scaled_train = (train_features - min_vals) / (max_vals - min_vals + 1e-8)
    scaled_target = (train_target - target_min) / (target_max - target_min + 1e-8)

    # Crear ventanas para entrenamiento
    X_train, y_train = [], []
    for i in range(len(train_df) - window_size):
        X_train.append(scaled_train[i:i + window_size])
        y_train.append(scaled_target[i + window_size])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Preparar test
    test_features = test_df[feature_cols].values
    test_target = test_df[target_col].values
    scaled_test = (test_features - min_vals) / (max_vals - min_vals + 1e-8)
    scaled_test_target = (test_target - target_min) / (target_max - target_min + 1e-8)

    X_test, y_test = [], []
    for i in range(test_size):
        X_test.append(scaled_test[i:i + window_size])
        y_test.append(scaled_test_target[i + window_size])

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Tensores
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

    # DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size, shuffle=False)


    # LSTM model
    class LSTMForecast(nn.Module):
        def __init__(self, input_size, hidden_size=hidden_size, num_layers=num_layers):
            super(LSTMForecast, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            return self.fc(out)


    model = LSTMForecast(input_size=X_train.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-4)

    # Entrenamiento
    for epoch in range(epoch_number):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")



    # Evaluación
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test_tensor).squeeze().cpu().numpy()

    preds = preds_scaled * (target_max - target_min + 1e-8) + target_min
    real = np.array(y_test) * (target_max - target_min + 1e-8) + target_min

    return real, preds


def run_lstm(df, target_col, window_size, test_size, batch_size,
             hidden_size, num_layers, epoch_number, lr,
             device, seed, patience, min_delta):

    seed_everything(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Using device: {device_info(device)}")

    feature_cols = [col for col in df.columns if col not in ['FECHA', target_col, 'series']]

    horizon = test_size  # Direct multi-horizon prediction

    # Split train/test
    train_df = df[:-test_size]
    test_df = df[-test_size:]

    # Escalamiento manual (solo con train)
    train_features = train_df[feature_cols].values
    train_target = train_df[target_col].values

    min_vals = train_features.min(axis=0)
    max_vals = train_features.max(axis=0)
    target_min = train_target.min()
    target_max = train_target.max()

    scaled_train = (train_features - min_vals) / (max_vals - min_vals + 1e-8)
    scaled_target = (train_target - target_min) / (target_max - target_min + 1e-8)

    # Crear ventanas para entrenamiento (train + val)
    # Each sample predicts the next `horizon` values directly
    X_all, y_all = [], []
    for i in range(len(train_df) - window_size - horizon + 1):
        X_all.append(scaled_train[i:i + window_size])
        y_all.append(scaled_target[i + window_size:i + window_size + horizon])

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    # Split train/val respetando orden temporal (último 20% como val)
    num_samples = X_all.shape[0]
    val_size = max(1, int(0.2 * num_samples))
    train_size = num_samples - val_size

    X_train = X_all[:train_size]
    y_train = y_all[:train_size]
    X_val = X_all[train_size:]
    y_val = y_all[train_size:]

    # Preparar test (misma escala)
    # Use last window_size rows of train as input to predict all test_size values
    test_input_features = train_df[feature_cols].values[-window_size:]
    scaled_test_input = (test_input_features - min_vals) / (max_vals - min_vals + 1e-8)

    X_test = np.array([scaled_test_input])  # Shape: (1, window_size, n_features)

    # Ground truth for test
    test_target = test_df[target_col].values
    y_test = (test_target - target_min) / (target_max - target_min + 1e-8)

    # Tensores
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    # DataLoader
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # LSTM model with multi-horizon output
    class LSTMForecast(nn.Module):
        def __init__(self, input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=horizon):
            super(LSTMForecast, self).__init__()
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=0.3 if num_layers > 1 else 0.0
            )
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            return self.fc(out)

    model = LSTMForecast(input_size=X_train.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Early stopping vars
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    for epoch in range(epoch_number):
        model.train()
        running_loss = 0.0
        num_train_samples = 0

        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            bs = batch_X.size(0)
            running_loss += loss.item() * bs
            num_train_samples += bs

        epoch_train_loss = running_loss / max(1, num_train_samples)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_tensor)
            val_loss = criterion(val_output, y_val_tensor).item()
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epoch_number} - "
                  f"Train Loss: {epoch_train_loss:.6f} - Val Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    epochs_ran = len(train_losses)

    # Evaluación en test
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test_tensor).squeeze().cpu().numpy()

    preds = preds_scaled * (target_max - target_min) + target_min
    real = y_test * (target_max - target_min) + target_min

    # "scalers"
    scaler_cov = {"min": min_vals, "max": max_vals}
    scaler_y = {"min": target_min, "max": target_max}

    out_list = [model, train_losses, val_losses, scaler_y, scaler_cov, epochs_ran]

    return real, preds, out_list


def run_sarima(df, target_col, test_size, p, d, q, P, D, Q, m, seed = 1048596):




    feature_cols = [
        col for col in df.columns
        if col not in ['FECHA', target_col, 'series'] and target_col not in col
    ]

    # Split

    train_df = df[:-test_size].copy()
    test_df  = df[-test_size:].copy()

    # Scale exogenous
    exog_scaler = StandardScaler()
    exog_train = exog_scaler.fit_transform(train_df[feature_cols]) if feature_cols else None
    exog_test  = exog_scaler.transform(test_df[feature_cols]) if feature_cols else None

    # Scale target
    y_train = train_df[target_col].values.astype(float)
    y_test  = test_df[target_col].values.astype(float)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1,1)).ravel()

    try:
        model = SARIMAX(
            endog=y_train_scaled,
            exog=exog_train,
            order=(p, d, q),
            seasonal_order=(P, D, Q, m),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res = model.fit(disp=False)
        fc_scaled = res.predict(
            start=len(train_df),
            end=len(df)-1,
            exog=exog_test
        )
        y_pred = y_scaler.inverse_transform(np.asarray(fc_scaled).reshape(-1,1)).ravel()
        y_true = y_test
        return y_true, y_pred
    except Exception:
        return [], []

def run_dnn(df, target_col, window_size, test_size, batch_size, epoch_number, lr, hidden_sizes, dropout, weight_decay, device, seed):

    seed_everything(seed)
    feature_cols = [col for col in df.columns if col not in ['FECHA', target_col]]
    # Split
    train_df = df[:-test_size]
    test_df = df[-(test_size + window_size):]

    # Normalize
    train_features = train_df[feature_cols].values
    train_target = train_df[target_col].values
    min_vals = train_features.min(axis=0)
    max_vals = train_features.max(axis=0)
    target_min = train_target.min()
    target_max = train_target.max()

    scaled_train = (train_features - min_vals) / (max_vals - min_vals + 1e-8)
    scaled_target = (train_target - target_min) / (target_max - target_min + 1e-8)

    X_train, y_train = [], []
    for i in range(len(train_df) - window_size):
        X_train.append(scaled_train[i:i + window_size])
        y_train.append(scaled_target[i + window_size])
    X_train, y_train = np.array(X_train), np.array(y_train)

    test_features = test_df[feature_cols].values
    test_target = test_df[target_col].values
    scaled_test = (test_features - min_vals) / (max_vals - min_vals + 1e-8)
    scaled_test_target = (test_target - target_min) / (target_max - target_min + 1e-8)

    X_test, y_test = [], []
    for i in range(test_size):
        X_test.append(scaled_test[i:i + window_size])
        y_test.append(scaled_test_target[i + window_size])
    X_test, y_test = np.array(X_test), np.array(y_test)

    # Flatten inputs for DNN
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    # DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size, shuffle=False)

    # DNN Model
    class DNNForecast(nn.Module):
        def __init__(self, input_size, hidden_sizes=hidden_sizes):
            super(DNNForecast, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_sizes[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_sizes[1], 1)
            )

        def forward(self, x):
            return self.layers(x)

    model = DNNForecast(input_size=X_train.shape[1]).to(device)

    # Training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)

    print("Running model")
    for epoch in range(epoch_number):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test_tensor).squeeze().cpu().numpy()

    preds = preds_scaled
    real = np.array(y_test)

    return real, preds


def get_attention_maps(model, sample):
    """
    sample: tensor of shape [B, T, input_size]
    returns: list of attn_maps, one per layer, each [B, n_heads, T, T]
    """
    attn_maps = []

    # same prep as in forward()
    x = model.input_linear(sample)
    x = x + model.positional_encoding[:, :x.size(1), :]

    # manually go through encoder layers and capture attention
    for layer in model.transformer.layers:
        # self-attention with need_weights=True
        attn_output, attn_weights = layer.self_attn(
            x, x, x,
            attn_mask=None,
            key_padding_mask=None,
            need_weights=True,
            average_attn_weights=False,
        )
        attn_maps.append(attn_weights.detach().cpu())  # [B, n_heads, T, T]

        # rest of TransformerEncoderLayer.forward (PyTorch)
        x = x + layer.dropout1(attn_output)
        x = layer.norm1(x)

        ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
        x = x + layer.dropout2(ff)
        x = layer.norm2(x)

    return attn_maps


def run_darts_tft2(df,
                  target_col,
                  test_size,
                  window_size,
                  hidden_size,
                  lstm_layers,
                  num_attention_heads,
                  dropout,
                  batch_size,
                  n_epochs,
                  lr,
                  grad_clip=1.0,
                  patience=20,
                  min_delta=1e-4,
                  seed=1048596,

                  ):
    """
    Trains a Darts TFT model on a univariate weekly series and returns
    model, predictions, and some metadata.
    """
    seed_everything(seed, workers=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        print("Running with cuda")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')


    output_chunk_length = test_size

    # ---- 1. Build series ----
    series = TimeSeries.from_dataframe(
        df,
        time_col="FECHA",
        value_cols=target_col,
    )



    # ---- 2. Train/val split ----
    val_size = test_size + window_size
    train = series[:-val_size]
    val = series[-val_size:]



    # ---- 3. Scaling ----
    scaler = Scaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)

    # ---- 4. Early stopping ----
    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        min_delta=min_delta,
        mode="min",
    )

    # ---- 5. Model ----
    model = TFTModel(
        input_chunk_length=window_size,
        output_chunk_length=output_chunk_length,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        batch_size=batch_size,
        n_epochs=n_epochs,
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
        random_state=seed,
        likelihood=None,
        optimizer_kwargs={"lr": lr},

        pl_trainer_kwargs={
            "accelerator": "gpu" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
            "callbacks": [early_stop],
            "gradient_clip_val": grad_clip,
            "gradient_clip_algorithm": "norm",
        },
    )

    # ---- 6. Fit ----
    model.fit(train_scaled, val_series=val_scaled, verbose=True, dataloader_kwargs={"num_workers": 0})

    # ---- 7. Epochs actually run ----
    epochs_ran = model.trainer.current_epoch + 1

    # ---- 8. Forecast ----
    pred_scaled = model.predict(n=test_size, dataloader_kwargs={"num_workers": 0})
    pred = scaler.inverse_transform(pred_scaled)

    val_last = val[-test_size:]

    true_vals = val_last.values().flatten().tolist()
    pred_vals = pred.values().flatten().tolist()
    list = [model, train, val, scaler, epochs_ran]

    return true_vals, pred_vals, list


def run_darts_tft(df,
                  target_col,
                  test_size,
                  window_size,
                  hidden_size,
                  lstm_layers,
                  num_attention_heads,
                  dropout,
                  batch_size,
                  n_epochs,
                  lr,
                  grad_clip=1.0,
                  patience=20,
                  min_delta=1e-4,
                  seed=1048596):
    """
    Trains a Darts TFT model using target + all other columns as past covariates.
    Returns true_vals, pred_vals, [model, train, val, scaler_target, scaler_cov, epochs_ran].
    """
    seed_everything(seed, workers=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        print("Running with cuda")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')

    output_chunk_length = test_size

    # ---- 1. Build target series ----
    series = TimeSeries.from_dataframe(
        df,
        time_col="FECHA",
        value_cols=target_col,
    )


    # ---- 1b. Build past covariates from all other columns ----
    feature_cols = [c for c in df.columns if c not in ["FECHA", target_col]]
    past_cov = TimeSeries.from_dataframe(
        df,
        time_col="FECHA",
        value_cols=feature_cols,
    )

    # ---- 2. Train/val split ----
    val_size = test_size + window_size
    train = series[:-val_size]
    val = series[-val_size:]

    train_past = past_cov[:-val_size]
    # full past covariates (for prediction later)
    full_past = past_cov

    # ---- 3. Scaling ----
    scaler_y = Scaler()
    train_scaled = scaler_y.fit_transform(train)
    val_scaled = scaler_y.transform(val)

    scaler_cov = Scaler()
    train_past_scaled = scaler_cov.fit_transform(train_past)
    full_past_scaled = scaler_cov.transform(full_past)
    val_past_scaled = full_past_scaled[-val_size:]

    # ---- 4. Early stopping ----
    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        min_delta=min_delta,
        mode="min",
    )

    # ---- 5. Model ----
    model = TFTModel(
        input_chunk_length=window_size,
        output_chunk_length=output_chunk_length,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        batch_size=batch_size,
        n_epochs=n_epochs,
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
        random_state=seed,
        likelihood=None,
        optimizer_kwargs={"lr": lr},
        pl_trainer_kwargs={
            "accelerator": "gpu" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu",
            "callbacks": [early_stop],
            "gradient_clip_val": grad_clip,
            "gradient_clip_algorithm": "norm",
        },
    )

    # ---- 6. Fit ----
    model.fit(
        train_scaled,
        past_covariates=train_past_scaled,
        val_series=val_scaled,
        val_past_covariates=val_past_scaled,
        verbose=True,
        dataloader_kwargs={"num_workers": 0},
    )

    # ---- 7. Epochs actually run ----
    epochs_ran = model.trainer.current_epoch + 1

    # ---- 8. Forecast ----
    # Use full_past_scaled so model has covariates over history + horizon
    pred_scaled = model.predict(
        n=test_size,
        past_covariates=full_past_scaled,
        dataloader_kwargs={"num_workers": 0},
    )
    pred = scaler_y.inverse_transform(pred_scaled)

    val_last = val[-test_size:]

    true_vals = val_last.values().flatten().tolist()
    pred_vals = pred.values().flatten().tolist()
    out_list = [model, train, val, scaler_y, scaler_cov, epochs_ran, feature_cols]

    return true_vals, pred_vals, out_list


def run_darts_tft_with_for(df,
              target_col,
              test_size,
              window_size,
              hidden_size,
              lstm_layers,
              num_attention_heads,
              dropout,
              batch_size,
              n_epochs,
              lr,
              grad_clip=1.0,
              patience=20,
              min_delta=1e-4,
              seed=1048596):
    mae_values = []
    mape_values = []
    mse_values = []
    rmse_values = []
    r2_values = []
    n_epochs_values = []

    for i in range(3):



         # Run model
        tmp = df.copy().sort_values("FECHA")
        deleted_sample = test_size*(i+1)  # delete the test samples from the end
        if deleted_sample > 0:
            tmp = tmp.iloc[:-deleted_sample]
        y_true, y_pred, out = run_darts_tft(
            tmp,
            target_col,
            test_size,
            window_size,
            hidden_size,
            lstm_layers,
            num_attention_heads,
            dropout,
            batch_size,
            n_epochs,
            lr,
            grad_clip,
            patience,
            min_delta,
            seed
        )

        mae, mape, mse, rmse, r2 = error_metrics(y_true, y_pred)

        mae_values.append(mae)
        mape_values.append(mape)
        mse_values.append(mse)
        rmse_values.append(rmse)
        r2_values.append(r2)
        n_epochs_values.append(out[5])
    print(mape_values)
    mean_epochs = np.mean(n_epochs_values)  # dummy values for errors
    return (np.mean(mae_values), np.mean(mape_values), np.mean(mse_values),
            np.mean(rmse_values), np.mean(r2_values), mean_epochs, np.std(mape_values))



def clean_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
