import numpy as np
import torch
from lightning.pytorch import Trainer, seed_everything
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE, QuantileLoss, SMAPE
from pytorch_forecasting.data.encoders import GroupNormalizer

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

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
    data["trend"] = np.arange(len(data))
    feature_cols = [c for c in data.columns if c not in ["FECHA", target_col, "time_idx", "series"]]

    time_varying_known_reals = ["time_idx", "dow", "month", "trend"]





    time_varying_unknown_reals = [target_col] + feature_cols

    # everything you want lagged must be "unknown" (past-only at predict time)
    laggable_feats = [c for c in data.columns if c not in ["FECHA", "series", target_col] and c not in time_varying_known_reals]
    time_varying_unknown_reals = [target_col] + laggable_feats

    # lags for target + ALL other variables (choose ranges you can afford)
    lags_dict = {target_col: list(range(1, window_size + 1))}


    for c in laggable_feats:
        lags_dict[c] = [1, 2, 3, 6, 12]



    # for c in laggable_feats:
    #     # option A: full window (heavy)
    #     lags_dict[c] = list(range(1, window_size + 1))
    #     # option B: lighter set (uncomment to use)
    #     # lags_dict[c] = [1,2,3,6,12,24]  # e.g., short + seasonal


    # OUTPUT SHAPE
    q = [0.5]  # choose your quantiles
    output_size = len(q)
    loss = QuantileLoss(quantiles=q)

    loss = SMAPE()
    output_size = 1



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
        allow_missing_timesteps=False,  # let the dataset handle gaps in time_idx
        lags=lags_dict,  # ← THIS GIVES THE MODEL MEMORY
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,

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

# receives the data as a pandas dataframe as shown in the transformers.py file
def run_transformer2(df, target_col, window_size, test_size, batch_size, d_model, n_head, num_layers, epoch_number, lr, device, seed):

    seed_everything(seed)
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
        def __init__(self, input_size, d_model=d_model, nhead=n_head, num_layers=num_layers):
            super(TransformerForecast, self).__init__()
            self.input_linear = nn.Linear(input_size, d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, 1)

        def forward(self, x):
            x = self.input_linear(x)
            x = self.transformer(x)
            out = x[:, -1, :]
            return self.fc(out)


    model = TransformerForecast(input_size=X_train.shape[2]).to(device)

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


def run_transformer(df, target_col, window_size, test_size, batch_size, d_model, n_head, num_layers, epoch_number, lr,
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

    return real, preds, [model, X_test]


def run_lstm(df, target_col, window_size, test_size, batch_size, hidden_size, num_layers, epoch_number, lr, device, seed):

    seed_everything(seed)
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

def run_dnn(df, target_col, window_size, test_size, batch_size, epoch_number, lr, hidden_sizes, dropout, weight_decay, device):


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