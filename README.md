# shippingNN

Code for the manuscript *Predictive Models for Interpretable Maritime Container Freight Pricing* (Morande, Ziliaskopoulos, Silva, González-Ramírez, Smith). Auburn University, Universidad de los Andes Chile, University of Alabama.

## What this repo does

Forecasts mean container freight rates with Transformer-based time-series models and explains the predictions with three complementary XAI techniques. The headline case study covers containerized imports to the ports of Chile.

- Models: Transformer, Temporal Fusion Transformer (TFT), LSTM, deep MLP, LASSO, SARIMA.
- Hyperparameter search: Bayesian Optimization per model.
- Explainability: SHAP, LIME, integrated gradients, attention-map inspection.
- Validation: blocked ANOVA across random seeds, multi-horizon comparison against baselines.

## Layout

| File or folder | Purpose |
|---|---|
| `data_processing_chile.py`, `data_processing_large.py` | Build the Chilean-imports feature panel from raw monthly trade and macro data |
| `data_preprocessing_uruguay.py` | Uruguay panel for the secondary case |
| `TFT.py`, `LSTM.py`, `attention_transformers.py`, `dnn.py`, `sarima.py`, `lasso.py` | Per-model training and evaluation |
| `bayesopt_*.py` | Bayesian Optimization hyperparameter sweeps |
| `attention_maps.py`, `integrated_gradients.py` | Post-hoc explanation routines |
| `TFT_block_anova.py` | Statistical comparison across blocks and seeds |
| `chile_data.csv`, `processed_df.csv` | Cleaned monthly panels for the Chile case |
| `notebooks/` | Result analysis and plot generation |
| `results/`, `plots/`, `lightning_logs/` | Training outputs and figures |

## Reproducing the headline experiment

Python 3.10 or newer. Set up the env:

```bash
python -m pip install -r requirements.txt
```

Re-run the Chile case:

```bash
python data_processing_chile.py     # build the feature panel
python bayesopt_tft.py              # tune the TFT
python TFT.py                       # train and evaluate the tuned TFT
python attention_maps.py            # generate the XAI artifacts
```

The full reproduction (all baselines, all horizons, all XAI artifacts) lives across `bayesopt_*.py`, the per-model trainers, and the analysis notebooks in `notebooks/`.

## Findings (paper)

The Transformer family delivers competitive multi-horizon accuracy against ARIMA, SARIMA, LSTM, and MLP baselines. The XAI layer surfaces two non-obvious drivers of Chilean import rates: Chinese macro indicators show up even on intra-Pacific lanes, and Chilean port congestion measurably shifts spot rates.

## Status

Research code from a manuscript in preparation. Stable for the experiments reported in the paper; not maintained as a production package.

## Citation

Pending. Will be updated on acceptance.

## Contact

Juan Pablo Morande, [jpm0112@auburn.edu](mailto:jpm0112@auburn.edu)
