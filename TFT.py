# --- NEW: Temporal Fusion Transformer with pytorch-forecasting ---
import warnings
warnings.filterwarnings("ignore")
# import pytorch_lightning as pl
from lightning.pytorch import Trainer, seed_everything
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAPE, MAE
from pytorch_forecasting.data.encoders import GroupNormalizer
import pandas as pd
import torch
import numpy as np
from functions import error_metrics
import os
from datetime import datetime

torch.set_float32_matmul_precision("high")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Build time features once (outside loops if you want)
df = pd.read_csv("data/test_final_kz.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"] + "-5", format="%Y-%W-%w")
df = df.sort_values('FECHA')
# clean column names (replace "." with "_")
df = df.rename(columns=lambda x: x.replace(".", "_"))




target_col = "XSICFEUW Index  (R4)"
window_size  = 5     # one year context
test_size    = 4     # ~half year test

d_model      = 64     # hidden size
n_head       = 2      # attention heads
num_layers   = 1      # lstm layers
epoch_number = 5    # train epochs
lr           = 1e-3   # learning rate
batch_size   = 16      # batch size
seed         = 1048596
deleted_sample = 0





# context window (encoder length)
window_sizes = [8]

# prediction horizon / holdout
test_sizes = [4]

# hidden sizes (model capacity)
d_models = [32, 64]

# attention heads (must divide hidden size)
n_heads = [2, 4]

# LSTM layers depth
num_layerss = [2, 3]

# training epochs (start small, scale later)
epoch_numbers = [50, 100]

# learning rates
lrs = [1e-1, 1e-5]

# seeds (reproducibility)
seeds = [1048596]

# optional: drop some samples for robustness
deleted_samples = [0,4,8,12]
n_folds = len(deleted_samples)

# targets (already defined)
target_cols = [
    # 'NE','SE','NAE','NAW','SAE','SAW',
    'XSICFEUW Index  (R4)','XSICFENE Index  (R1)',
    'XSICFESE Index  (L4)','XSICNEFE Index  (R2)',
    'XSICNESE Index  (L2)','XSICUENE Index  (R1)'
]

target_cols = [
    'XSICFENE Index  (R1)','XSICNEFE Index  (R2)','XSICUENE Index  (R1)'
]


results_df = pd.DataFrame(columns=[
    "seed","deleted_samples","window_size","test_size","hidden_size","n_head",
    "num_layers","epoch_number","Target","LR","MAE","MAPE","MSE","RMSE","r2"
])


total = (
    len(seeds) *
    len(window_sizes) *
    len(test_sizes) *
    len(d_models) *
    len(n_heads) *
    len(num_layerss) *
    len(epoch_numbers) *
    len(target_cols)
)
iteration_counter = 0

for seed in seeds:
    for window_size in window_sizes:
        for test_size in test_sizes:
            for d_model in d_models:
                for n_head in n_heads:
                    for num_layers in num_layerss:
                        for epoch_number in epoch_numbers:
                            for target_col in target_cols:
                                for lr in lrs:

                                    iteration_counter = iteration_counter +1
                                    print("Iteration: " + str(iteration_counter) + "/" + str(total))

                                    sum_mae = 0
                                    sum_mape = 0 
                                    sum_mse = 0
                                    sum_rmse = 0
                                    sum_r2 = 0


                                    for deleted_sample in deleted_samples:


                                        tmp = df.copy()
                                        tmp = df.sort_values("FECHA").copy()
                                        tmp["series"] = "kz"  # or your series id
                                        tmp["time_idx"] = tmp.groupby("series").cumcount()
                                        # simple known-future calendar features (always available)
                                        tmp["dow"] = tmp["FECHA"].dt.weekday.astype(int)
                                        tmp["month"] = tmp["FECHA"].dt.month.astype(int)
                                        if deleted_sample > 0:
                                            tmp = tmp.iloc[:-deleted_sample]
                                        
                                        train_cut = tmp["time_idx"].max() - test_size 


                                        print(f"Running with seed={seed}, window_size={window_size}, test_size={test_size}, "
                                              f"d_model={d_model}, num_layers={num_layers}, epoch_number={epoch_number}, "
                                              f"target_col={target_col}, lr={lr}, deleted_sample={deleted_sample}")

                                        
                                        def run_tft(
                                            data, target_col, window_size, test_size,
                                            d_model, n_head, num_layers, epoch_number, lr, batch_size, seed
                                        ):
                                            # pl.seed_everything(seed)

                                            # split by time
                                            max_time = data["time_idx"].max()
                                            train_cutoff = max_time - test_size

                                            # observed covariates = your other columns (no need to pre-scale)
                                            feature_cols = [c for c in data.columns if c not in ["FECHA", target_col, "time_idx", "series"]]
                                            time_varying_known_reals = ["time_idx", "dow", "month"]          # known into the future
                                            time_varying_unknown_reals = [target_col] + feature_cols         # observed only historically


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

                                            # validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)




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
                                                loss=MAE(),          # or QuantileLoss() with output_size>1
                                                output_size=1,
                                                reduce_on_plateau_patience=3,
                                                gradient_clip_val = 0.1
                                            )

                                            trainer = Trainer(
                                                max_epochs=epoch_number,
                                                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                                                devices=1,
                                                # precision="16-mixed",   # <<< use FP16 3090
                                                log_every_n_steps=10,
                                                enable_checkpointing=False,
                                                enable_model_summary=False,
                                            )

                                            trainer.fit(tft, train_loader, val_loader)

                                            # Predict last test_size steps (1-step-ahead rolling from validation set)
                                            # preds = tft.predict(val_loader, trainer=trainer).squeeze(-1).cpu().numpy()

                                            preds = tft.predict(val_loader).squeeze(-1).cpu().numpy()

                                            # True values aligned with preds:
                                            y_true = []
                                            # for batch in iter(val_loader):
                                            #     # batch[1] is target in pytorch-forecasting dataloader
                                            #     y_true.append(batch[1].cpu().numpy())

                                            for x, y in val_loader:
                                                # if (target, weight), keep only target
                                                if isinstance(y, (tuple, list)):
                                                    y = y[0]
                                                # move to cpu + numpy
                                                y_true.append(y.detach().cpu().numpy())

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

                                        mae,mape, mse, rmse, r2 = error_metrics(y_test_true, y_test_pred)

                                        sum_mae  += mae
                                        sum_mape += mape
                                        sum_mse  += mse
                                        sum_rmse += rmse
                                        sum_r2   += r2


                                    avg_mae  = sum_mae  / n_folds
                                    avg_mape = sum_mape / n_folds
                                    avg_mse  = sum_mse  / n_folds
                                    avg_rmse = sum_rmse / n_folds
                                    avg_r2   = sum_r2   / n_folds

                                    results_df.loc[len(results_df)] = [
                                        seed, deleted_sample, window_size, test_size, d_model, n_head, num_layers, epoch_number, target_col, lr,
                                        avg_mae, avg_mape, avg_mse, avg_rmse, avg_r2
                                    ]
                                    results_df.to_csv(os.path.join("results", f"model_results_TFT_{timestamp}.csv"), index=False)

                                    

