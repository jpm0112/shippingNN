import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import torch
import numpy as np
from datetime import datetime
# from functions import error_metrics, run_transformer_xai, get_attention_maps, run_transformer_with_for_xai
from functions import error_metrics, run_transformer_xai, run_transformer_with_for_xai
from functions import prettify, rename_specific
import os

import matplotlib.pyplot as plt

# torch.set_float32_matmul_precision("high")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df = pd.read_csv("weekly_chile_data.csv")
df["FECHA"] = pd.to_datetime(df["FECHA"])
tmp = df.copy().sort_values("FECHA")

# BEST FE 4
test_size = 4
target_col = "FE"
window_size = 42
batch_size = 128
d_model = 64
n_head = 2
num_layers = 4  # lstm layers
epoch_number = 2000
lr = 0.000340605
dropout = 0.069862972
weight_decay = 0.000187754

# BEST NAE 4

test_size = 4
target_col = "NAE"
window_size = 51
batch_size = 32
d_model = 64
n_head = 4
num_layers = 4  # lstm layers
epoch_number = 2000
lr = 0.000397956
dropout = 0.223329688
weight_decay = 0.000103901

# BEST NAW 4
test_size = 4
target_col = "NAW"
window_size = 41
batch_size = 64
d_model = 128
n_head = 4
num_layers = 4  # lstm layers
epoch_number = 2000
lr = 6.47E-05
dropout = 0.121422183
weight_decay = 6.10E-06

# BEST NE 4
test_size = 4
target_col = "NE"
window_size = 30
batch_size = 128
d_model = 32
n_head = 8
num_layers = 4  # lstm layers
epoch_number = 2000
lr = 9.28E-05
dropout = 0.483863325
weight_decay = 0.000197146

# BEST SE 4
test_size = 4
target_col = "SE"
window_size = 36
batch_size = 32
d_model = 256
n_head = 4
num_layers = 4  # lstm layers
epoch_number = 2000
lr = 0.000555638
dropout = 0.312582405
weight_decay = 0.000179268

# BEST SAW 4 udpated with real values from csv
test_size = 4
target_col = "SAW"
window_size = 33
batch_size = 128
d_model = 32
n_head = 2
num_layers = 2  # lstm layers
epoch_number = 2000
lr = 0.000153194
dropout = 0.575846442

weight_decay = 0.0000808

# # # BEST SAE 4 (updated with real values from the csv)
# test_size = 4
# target_col = "SAE"
# window_size = 32
# batch_size = 32
# d_model = 128
# n_head = 2
# num_layers = 4  # lstm layers
# epoch_number = 2000
# lr = 0.000482690969879744
# dropout = 0.517106545291414
# weight_decay = 7.75222194438743E-06
#
# # BEST FE 12
# test_size = 12
# target_col = "FE"
# window_size = 28
# batch_size = 32
# d_model = 128
# n_head = 2
# num_layers = 4  # lstm layers
# epoch_number = 2000
# lr = 0.001
# dropout = 0.332641601
# weight_decay = 0.001
#
# # BEST NAE 12
# test_size = 12
# target_col = "NAE"
# window_size = 26
# batch_size = 32
# d_model = 64
# n_head = 2
# num_layers = 4  # lstm layers
# lr = 0.000187546
# dropout = 0.273919743
# weight_decay = 7.55E-06
#
# # BEST NAW 12
# test_size = 12
# target_col = "NAW"
# window_size = 51
# batch_size = 128
# d_model = 256
# n_head = 8
# num_layers = 4  # lstm layers
# lr = 0.00029803
# dropout = 0.165152428
# weight_decay = 5.62E-06
#
# # BEST NE 12
# test_size = 12
# target_col = "NE"
# window_size = 47
# batch_size = 32
# d_model = 64
# n_head = 2
# num_layers = 4  # lstm layers
# lr = 0.0004815
# dropout = 0.318221159
# weight_decay = 4.00E-05
#
# # BEST SE 12
# test_size = 12
# target_col = "SE"
# window_size = 50
# batch_size = 32
# d_model = 64
# n_head = 8
# num_layers = 4  # lstm layers
# lr = 0.000427606
# dropout = 0.444859662
# weight_decay = 1.50E-05
#
# BEST SAW 12
# test_size = 12
# target_col = "SAW"
# window_size = 49
# batch_size = 128
# d_model = 256
# n_head = 4
# num_layers = 4
# lr = 0.000551659
# dropout = 0.426466938
# weight_decay = 6.90E-05
# #
# # # BEST SAE 12 updated values from csv
# test_size = 12
# target_col = "SAE"
# window_size = 45
# batch_size = 64
# d_model = 256
# n_head = 8
# num_layers = 3  # lstm layers
# lr = 0.000496421949598074
# dropout = 0.506098106834949
# weight_decay = 0.000001


# # combination SAW 12
# test_size = 12
# target_col = "SAW"
# window_size = 4
# batch_size = 64
# d_model = 256
# n_head = 4
# num_layers = 3
# lr = 0.000166341101818292
# dropout = 0.292323124414817
# weight_decay = 7.86040820898082E-06
#
# # combination SAE 12
# test_size = 12
# target_col ="SAE"
# window_size = 4
# batch_size = 128
# d_model = 256
# n_head = 4
# num_layers = 3
# lr = 0.000169317234009676
# dropout = 0.447946858052264
# weight_decay = 0.0000167817374557847

# # combination SAW 4
# test_size = 4
# target_col = "SAW"
# window_size = 3
# batch_size = 64
# d_model = 64
# n_head = 2
# num_layers = 2
# lr = 0.000396541130496921
# dropout = 0.392386896101136
# weight_decay = 0.000159288598534597
#
# # combination SAE 4
# test_size = 4
# target_col = "SAE"
# window_size = 3
# batch_size = 128
# d_model = 64
# n_head = 8
# num_layers = 4
# lr = 0.000171389801980269
# dropout = 0.262536763586103
# weight_decay = 0.000169851561630298

# ==============================

# General parameters
seed = 1048596
patience = 200
min_delta = 1e-5
epoch_number = 2000

tmp = df.copy().sort_values("FECHA")

avg_mae, avg_mape, avg_mse, avg_rmse, avg_r2, avg_epochs_ran, sd, out, y_true, y_pred = run_transformer_with_for_xai(
    tmp, target_col, window_size, test_size, batch_size, d_model, n_head,
    num_layers, epoch_number, lr, dropout,
    device, seed, optimizer_type='adam', weight_decay=weight_decay, early_stop=True,
    patience=patience,
    min_delta=min_delta,
    n_runs=3,
)

mae, mape, mse, rmse, r2 = error_metrics(y_true, y_pred)

plt.figure(figsize=(12, 6))
plt.plot(y_true, marker="o", label="Real")
plt.plot(y_pred, marker="o", label="Prediction")
plt.title("Prediction (original scale)")
plt.xlabel("Weeks")
plt.ylabel(target_col)
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(f"plots/transformers_prediction_{target_col}_{test_size}.png", dpi=200)

H = test_size  # 4

model = out[0]
train_ts = out[1]
val_ts = out[2]
scaler_y = out[3]
scaler_cov = out[4]
feature_cols = out[6]

import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from darts import TimeSeries

# AUDIT FIX: make sure the model is in inference mode for XAI (dropout OFF)
model.to(device)
model.eval()
torch.set_grad_enabled(False)

# Optional guard: if you truly intend a covariates-only model, the target must NOT be in feature_cols
if target_col in feature_cols:
    print(
        f"⚠️ WARNING: target_col '{target_col}' is present inside feature_cols. This leaks the target into the model inputs and will invalidate feature importance.")

# Ensure datetime (CRITICAL)
df["FECHA"] = pd.to_datetime(df["FECHA"])

# Rebuild full target series and covariates
full_series = TimeSeries.from_dataframe(df, "FECHA", target_col)
full_cov = TimeSeries.from_dataframe(df, "FECHA", feature_cols)

series_scaled = scaler_y.transform(full_series)
cov_scaled = scaler_cov.transform(full_cov)

F = 1 + len(feature_cols)

# last forecast point
last_idx = len(df) - 1
start_hist = last_idx - test_size + 1 - window_size
assert start_hist >= 0

start_time = df["FECHA"].iloc[start_hist]
end_time = df["FECHA"].iloc[start_hist + window_size - 1]

hist_target = series_scaled.slice(start_time, end_time)
hist_cov = cov_scaled.slice(start_time, end_time)

y_hist_np = hist_target.values(copy=True)
cov_hist_np = hist_cov.values(copy=True)

hist_full_np = np.concatenate([y_hist_np, cov_hist_np], axis=1)
x0 = hist_full_np.flatten()

# Feature names
feat_names = []
cols_all = [target_col] + feature_cols
for t in range(window_size):
    lag = window_size - 1 - t
    for c in cols_all:
        feat_names.append(f"{c}_t-{lag}")
N_ORIGINS = 4
# LIME _____________________________________________-


# ============================================================


#
#
# # ATTENTION MAPS FOR TRANSFORMER
# # ============================================================
#
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # ------------------------------------------------------------
# # CONFIG
# # ------------------------------------------------------------
# LAYER_TO_PLOT = 0  # last Transformer layer
# HEAD_TO_PLOT = 0  # specific head
# SAVE_DIR = "plots"
#
# os.makedirs(SAVE_DIR, exist_ok=True)
#
# T = window_size
# F = 1 + len(feature_cols)
#
# # ------------------------------------------------------------
# # 1) ATTENTION FOR SINGLE PREDICTION (x0)
# # ------------------------------------------------------------
#
# # rebuild covariate window (Transformer input)
# arr = x0.reshape(T, F)
# cov_win = arr[:, 1:]  # Transformer only sees covariates
#
# X_attn = torch.tensor(
#     cov_win,
#     dtype=torch.float32
# ).unsqueeze(0).to(device)
#
# # get attention maps
# attn_maps = get_attention_maps(model, X_attn)
#
# # expected shape: attn_maps[layer] -> (heads, T, T)
# print("Attention map shapes:")
# for l, A in enumerate(attn_maps):
#     print(f"Layer {l}: {A.shape}")
#
# # ------------------------------------------------------------
# # 2) PLOT SINGLE HEAD ATTENTION
# # ------------------------------------------------------------
#
# A_head = attn_maps[LAYER_TO_PLOT][0, HEAD_TO_PLOT].cpu().numpy()
#
# plt.figure(figsize=(8, 6))
# sns.heatmap(
#     A_head,
#     cmap="viridis",
#     xticklabels=False,
#     yticklabels=False
# )
# plt.title(
#     f"Attention Map – Layer {LAYER_TO_PLOT}, Head {HEAD_TO_PLOT}"
# )
# plt.xlabel("Key time step (past)")
# plt.ylabel("Query time step")
# plt.tight_layout()
# plt.savefig(
#     f"{SAVE_DIR}/attention_single_layer{LAYER_TO_PLOT}_head{HEAD_TO_PLOT}_{target_col}.png",
#     dpi=200
# )
# plt.show()
#
# # ============================================================
# # 4) GLOBAL ATTENTION (AVERAGED OVER MANY WINDOWS)  [FIXED]
# # ============================================================
# # AUDIT FIX: the original code averaged attention over ALL-ZERO windows.
# # Here we reuse the same valid window logic used above (no leakage).
#
# N_GLOBAL = 80
# end_min = T - 1
# end_max = len(series_scaled) - H - 1
# valid_ends = np.arange(end_min, end_max + 1)
#
# rng = np.random.default_rng(seed)
# attn_ends = rng.choice(valid_ends, size=min(N_GLOBAL, len(valid_ends)), replace=False)
#
# ATTN_GLOBAL = []
#
# for end in attn_ends:
#     start = end - T + 1
#     y_win = series_scaled.values(copy=True)[start:end + 1]
#     cov_win = cov_scaled.values(copy=True)[start:end + 1]
#
#     arr = np.concatenate([y_win, cov_win], axis=1)  # (T,F)
#     cov_in = arr[:, 1:]  # model sees covariates only
#
#     X_attn = torch.tensor(cov_in, dtype=torch.float32).unsqueeze(0).to(device)
#
#     with torch.no_grad():
#         attn = get_attention_maps(model, X_attn)
#
#     ATTN_GLOBAL.append(
#         attn[LAYER_TO_PLOT][0].mean(axis=0).cpu().numpy()
#     )
#
# ATTN_GLOBAL_MEAN = np.mean(ATTN_GLOBAL, axis=0)  # (T,T)
#
# plt.figure(figsize=(8, 6))
# sns.heatmap(
#     ATTN_GLOBAL_MEAN,
#     cmap="viridis",
#     xticklabels=False,
#     yticklabels=False
# )
# plt.title(
#     f"Global Mean Attention – Layer {LAYER_TO_PLOT}"
# )
# plt.xlabel("Key time step (past)")
# plt.ylabel("Query time step")
# plt.tight_layout()
# plt.savefig(
#     f"{SAVE_DIR}/attention_global_mean_layer{LAYER_TO_PLOT}_{target_col}.png",
#     dpi=200
# )
# plt.show()
#
# # ============================================================
# # GLOBAL TIME × TIME ATTENTION (LAST LAYER, HEADS AVERAGED)
# # ============================================================
#
# LAYER = -1
# T = window_size
# F = 1 + len(feature_cols)
#
# ATTN = []
#
# for i in range(N_GLOBAL):
#     arr = X_global[i].reshape(T, F)
#     cov_win = arr[:, 1:]
#
#     X = torch.tensor(cov_win, dtype=torch.float32).unsqueeze(0).to(device)
#
#     with torch.no_grad():
#         attn_maps = get_attention_maps(model, X)
#         # attn_maps[layer]: (heads, T, T)
#         A = attn_maps[LAYER][0].mean(dim=0)
#     # avg over heads → (T,T)
#
#     ATTN.append(A.cpu().numpy())
#
# ATTN_GLOBAL = np.mean(ATTN, axis=0)  # avg over windows → (T,T)
#
# plt.figure(figsize=(8, 6))
# sns.heatmap(ATTN_GLOBAL, cmap="viridis")
# plt.xlabel("Key time step (past)")
# plt.ylabel("Query time step")
# plt.title("Global Mean Attention (Time × Time)")
# plt.tight_layout()
# plt.show()
#
# # # ============================================================
# # HEAD SPECIALIZATION OVER TIME
# n_layers = len(attn_maps)
#
# for layer in range(n_layers):
#     A = attn_maps[layer][0].cpu().numpy()  # (heads, T, T)
#     head_importance = A.mean(axis=2)  # (heads, T)
#
#     plt.figure(figsize=(8, 4))
#     sns.heatmap(
#         head_importance,
#         cmap="viridis",
#         yticklabels=[f"Head {i}" for i in range(head_importance.shape[0])],
#         xticklabels=False
#     )
#     plt.xlabel("Past time step")
#     plt.ylabel("Attention head")
#     plt.title(f"Head specialization – Layer {layer}")
#     plt.tight_layout()
#     plt.savefig(
#         f"plots/attention_head_specialization_layer{layer}_{target_col}_h{test_size}.png",
#         dpi=200,
#         bbox_inches="tight"
#     )
#     plt.show()
#     plt.close()
#
# LAYER = 1
# n_heads = attn_maps[LAYER].shape[1]
# for h in range(n_heads):
#     A = attn_maps[LAYER][0, h].cpu().numpy()  # (T_query, T_key)
#     var_query = np.mean(np.var(A, axis=0))  # varies by query
#     var_key = np.mean(np.var(A, axis=1))  # varies by key
#     print(
#         f"Head {h}: "
#         f"query-var = {var_query:.4e}, "
#         f"key-var = {var_key:.4e}"
#     )
#
# head_stats = []
#
# n_layers = len(attn_maps)
#
# for L in range(n_layers):
#     n_heads = attn_maps[L].shape[1]
#
#     for h in range(n_heads):
#         A = attn_maps[L][0, h].cpu().numpy()  # (T_query, T_key)
#
#         query_var = np.mean(np.var(A, axis=0))
#         key_var = np.mean(np.var(A, axis=1))
#         total_var = query_var + key_var
#
#         head_stats.append({
#             "layer": L,
#             "head": h,
#             "query_var": query_var,
#             "key_var": key_var,
#             "total_var": total_var
#         })
#
# df_heads = (
#     pd.DataFrame(head_stats)
#     .sort_values("total_var", ascending=False)
#     .reset_index(drop=True)
# )
#
# print(df_heads.head(10))
#
# # ONE HEAD ATTENTION OVER TIME OF ONE LAYER
# LAYER = 1  # which Transformer layer
# HEAD = 6  # which attention head
#
# # take ONE window (e.g., the first global window)
# arr = X_global[0].reshape(T, F)
# cov_win = arr[:, 1:]
#
# X = torch.tensor(cov_win, dtype=torch.float32).unsqueeze(0).to(device)
#
# with torch.no_grad():
#     attn_maps = get_attention_maps(model, X)
#     # attn_maps[layer]: (heads, T, T)
#     A_head = attn_maps[LAYER][0, HEAD].cpu().numpy()  # (T, T)
#
# plt.figure(figsize=(8, 6))
# sns.heatmap(
#     A_head,
#     cmap="viridis",
#     linewidths=0,  # <<< removes white grid lines
#     linecolor=None,
#     cbar=True
# )
#
# # thin ticks
# step = 5  # show one tick every 5 steps
# plt.xticks(
#     ticks=np.arange(0, T, step),
#     labels=np.arange(0, T, step),
#     rotation=0
# )
# plt.yticks(
#     ticks=np.arange(0, T, step),
#     labels=np.arange(0, T, step),
#     rotation=0
# )
#
# plt.xlabel("Key time step (past)")
# plt.ylabel("Query time step")
# plt.title(f"Attention – Layer {LAYER}, Head {HEAD}")
# plt.tight_layout()
# plt.show()

# ============================================================
# ============================================================
# BUILD GLOBAL WINDOWS FOR ATTENTION
# ============================================================

N_GLOBAL = 104
T = window_size
H = test_size
F = 1 + len(feature_cols)

end_min = T - 1
end_max = len(series_scaled) - H - 1
valid_ends = np.arange(end_min, end_max + 1)
non_overlap_ends = valid_ends[::test_size]

rng = np.random.default_rng(seed)
chosen_ends = rng.choice(
    non_overlap_ends,
    size=min(N_GLOBAL, len(non_overlap_ends)),
    replace=False
)

X_global = np.zeros((len(chosen_ends), T * F), dtype=np.float32)

for i, end in enumerate(chosen_ends):
    start = end - T + 1

    y_win = series_scaled.values(copy=True)[start:end + 1]
    cov_win = cov_scaled.values(copy=True)[start:end + 1]

    arr = np.concatenate([y_win, cov_win], axis=1)  # (T, F)
    X_global[i] = arr.flatten()


# LAYER = -1
# T = window_size
# F = 1 + len(feature_cols)
#
# ATTN = []
#
# for i in range(len(X_global)):
#     arr = X_global[i].reshape(T, F)
#     cov_win = arr[:, 1:]
#
#     X = torch.tensor(cov_win, dtype=torch.float32).unsqueeze(0).to(device)
#
#     with torch.no_grad():
#         attn_maps = get_attention_maps(model, X)
#         A = attn_maps[LAYER][0].mean(dim=0)  # avg over heads
#
#     ATTN.append(A.cpu().numpy())
#
# ATTN_GLOBAL = np.mean(ATTN, axis=0)
#
#
# # ============================================================
#
#
# # mean of one layer's attention heads over many windows
#
#
# plt.figure(figsize=(8, 6))
# plt.imshow(
#     ATTN_GLOBAL,
#     cmap="viridis",
#     aspect="auto",
#     interpolation="nearest"  # ← KEY
# )
# plt.xlabel("Key time step (past)")
# plt.ylabel("Query time step")
# plt.title("Global Mean Attention (Time × Time)")
# plt.colorbar()
# plt.tight_layout()
# plt.show()


def get_attention_maps(model, x):
    """
    Returns a list (per layer) of attention weights with shape:
      attn_maps[L] -> (B, n_head, T, T)
    Works for your CustomEncoderLayer which stores TransformerEncoderLayer in `.layer`.
    """
    attn_maps = []
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, inp, out):
            # out is (attn_output, attn_weights) when need_weights=True
            # attn_weights is either (B, T, T) if average_attn_weights=True
            # or (B, n_head, T, T) if average_attn_weights=False
            attn_w = out[1].detach()
            attn_maps[layer_idx] = attn_w

        return hook_fn

    # pre-allocate list
    n_layers = len(model.layers)
    attn_maps = [None] * n_layers

    # register hook on each layer's self_attn module
    for L in range(n_layers):
        mha = model.layers[L].layer.self_attn
        hooks.append(mha.register_forward_hook(make_hook(L)))

    # run a forward pass
    _ = model(x)

    # remove hooks
    for h in hooks:
        h.remove()

    return attn_maps


# ============================================================

ATTN = []

for i in range(len(X_global)):
    arr = X_global[i].reshape(T, F)
    cov_win = arr[:, 1:]
    X = torch.tensor(cov_win, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        attn_maps = get_attention_maps(model, X)

        A_layers = []
        for L in range(len(attn_maps)):
            A_L = attn_maps[L][0].mean(dim=0)  # avg over heads
            A_layers.append(A_L)

        A_mean_layers = torch.stack(A_layers).mean(dim=0)  # avg over layers

    ATTN.append(A_mean_layers.cpu().numpy())

ATTN_GLOBAL_ALL_LAYERS = np.mean(ATTN, axis=0)

plt.figure(figsize=(8, 6))
plt.imshow(
    ATTN_GLOBAL_ALL_LAYERS,
    cmap="viridis",
    aspect="auto",
    interpolation="nearest"
)
plt.xlabel("Key time step (past)")
plt.ylabel("Query time step")
plt.title("Global Mean Attention (All Layers)")
plt.colorbar()
plt.tight_layout()
plt.savefig(
    f"plots/global_mean_attention_all_layers_{target_col}_{test_size}.png",
    dpi=200,
    bbox_inches="tight"
)
plt.show()

import seaborn as sns

# ONE HEAD ATTENTION OVER TIME OF ONE LAYER (GLOBAL WINDOW)
LAYER = -1
HEAD = 0  # change based on variance ranking if you want

A_head = attn_maps[LAYER][0, HEAD].cpu().numpy()

plt.figure(figsize=(8, 6))
sns.heatmap(
    A_head,
    cmap="viridis",
    linewidths=0,
    cbar=True
)

step = 5
plt.xticks(np.arange(0, T, step), np.arange(0, T, step))
plt.yticks(np.arange(0, T, step), np.arange(0, T, step))

plt.xlabel("Key time step (past)")
plt.ylabel("Query time step")
plt.title(f"Attention – Layer {LAYER}, Head {HEAD}")
plt.tight_layout()
plt.show()

ATTN_BY_LAYER = []

n_layers = len(attn_maps)  # number of Transformer layers

for L in range(n_layers):

    ATTN_L = []

    for i in range(len(X_global)):
        arr = X_global[i].reshape(T, F)
        cov_win = arr[:, 1:]

        X = torch.tensor(cov_win, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            attn_maps = get_attention_maps(model, X)
            A = attn_maps[L][0].mean(dim=0)  # mean over heads

        ATTN_L.append(A.cpu().numpy())

    ATTN_GLOBAL_L = np.mean(ATTN_L, axis=0)  # mean over windows

    # ------------------ PLOT ------------------
    plt.figure(figsize=(8, 6))
    plt.imshow(
        ATTN_GLOBAL_L,
        cmap="viridis",
        aspect="auto",
        interpolation="nearest"
    )
    plt.xlabel("Key time step (past)")
    plt.ylabel("Query time step")
    plt.title(f"Global Mean Attention – Layer {L + 1}")
    plt.colorbar()
    plt.tight_layout()

    plt.savefig(
        f"plots/global_mean_attention_layer{L}_{target_col}_{test_size}.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()

attn_list = out[7]  # list of layers
A = attn_list[0][0]  # layer 0, sample 0: (n_head, T, T)
A = A.mean(0)  # avg heads -> (T, T)
last = A[-1].cpu().numpy()  # attention used to make prediction


# simple attention plot
# ------------------------------------------------

def get_attention_maps(model, x):
    """
    Returns a list (per layer) of attention weights with shape:
      attn_maps[L] -> (B, n_head, T, T)
    Works for your CustomEncoderLayer which stores TransformerEncoderLayer in `.layer`.
    """
    attn_maps = []
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, inp, out):
            # out is (attn_output, attn_weights) when need_weights=True
            # attn_weights is either (B, T, T) if average_attn_weights=True
            # or (B, n_head, T, T) if average_attn_weights=False
            attn_w = out[1].detach()
            attn_maps[layer_idx] = attn_w

        return hook_fn

    # pre-allocate list
    n_layers = len(model.layers)
    attn_maps = [None] * n_layers

    # register hook on each layer's self_attn module
    for L in range(n_layers):
        mha = model.layers[L].layer.self_attn
        hooks.append(mha.register_forward_hook(make_hook(L)))

    # run a forward pass
    _ = model(x)

    # remove hooks
    for h in hooks:
        h.remove()

    return attn_maps


all_attn = []

T = window_size
H = test_size
LAST_YEAR = 104  # 2 years if weekly, adjust if needed

end_min = T - 1
end_max = len(series_scaled) - H - 1

valid_ends = np.arange(end_min, end_max + 1)

# windows whose END is inside last year
last_year_start = len(series_scaled) - LAST_YEAR - 1
last_year_ends = valid_ends[valid_ends >= last_year_start]

n_cov = len(feature_cols)
N = len(last_year_ends)

X_batch = np.zeros((N, T, n_cov), dtype=np.float32)

for i, end in enumerate(last_year_ends):
    start = end - T + 1
    X_batch[i] = cov_scaled.values(copy=True)[start:end + 1]

X_batch = torch.tensor(X_batch, dtype=torch.float32).to(device)

print("Batch shape:", X_batch.shape)  # (N, T, n_cov)

with torch.no_grad():
    _, attn_list = model(X_batch, return_attn=True)

print("Layers:", len(attn_list))
print("Layer 0 shape:", attn_list[0].shape)
# should be (N, n_head, T, T)


n_layers = len(attn_list)
T = window_size

import matplotlib.pyplot as plt
import numpy as np

n_layers = len(attn_list)
T = window_size

for L in range(n_layers):

    A = attn_list[L]  # (N, n_head, T, T)
    N, n_head, _, _ = A.shape

    A_mean = A.mean(dim=0)  # (n_head, T, T)

    plt.figure(figsize=(8, 6))

    # better distinct colors
    colors = plt.cm.tab10(np.linspace(0, 1, n_head))

    for h in range(n_head):
        colors = [
            "#e41a1c",  # red
            "#377eb8",  # blue
            "#4daf4a",  # green
            "#984ea3",  # purple
            "#ff7f00",  # orange
            "#a65628",  # brown
            "#f781bf",  # pink
            "#999999",  # gray
        ]

        last_row = A_mean[h, -1, :].cpu().numpy()[::-1]
        plt.plot(last_row, color=colors[h % len(colors)], linewidth=2, label=f"Attention Head {h + 1}")

    # plt.title(f"Layer {L + 1}", fontsize=14)
    plt.xlabel("Time lag (0 = most recent)", fontsize=14)
    plt.ylabel("Attention weight", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(f"plots/attention_{target_col}_H{test_size}_layer{L + 1}.png",
                dpi=300, bbox_inches="tight")
    plt.show()

plt.figure(figsize=(8, 6))

layer_colors = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange
]

for L in range(n_layers):
    A = attn_list[L]  # (N, n_head, T, T)

    # mean over samples and heads
    A_mean = A.mean(dim=0).mean(dim=0)  # (T, T)

    # last query token
    last_row = A_mean[-1, :].cpu().numpy()[::-1]

    plt.plot(
        last_row,
        color=layer_colors[L % len(layer_colors)],
        linewidth=2.5,
        label=f"Layer {L + 1}"
    )

plt.xlabel("Time lag (0 = most recent)", fontsize=14)
plt.ylabel("Attention weight", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(f"plots/attention_layers_{target_col}_H{test_size}.png",
            dpi=300, bbox_inches="tight")
plt.show()