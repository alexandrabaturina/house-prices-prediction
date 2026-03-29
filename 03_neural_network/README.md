# House Price Prediction: Neural Network

This project explores whether a feedforward neural network can outperform classical regression methods on housing dataset, building directly on the [Nonlinear modeling project](https://github.com/alexandrabaturina/house-price-prediction/tree/main/02_nonlinear_modeling).

## Methodology

1. **Default Model** — feedforward neural network with a fixed pyramidal architecture (128 → 64 → 32)
2. **Sensitivity Analysis** — the default model is trained with multiple fixed seeds to quantify the effect of random initialization
3. **Hyperparameter Tuning** — random search over learning rate, batch size, and early stopping patience

## Architecture

| Layer | Details |
|---|---|
| Input | number of input features |
| Dense | 128 neurons, ReLU |
| Dropout | 0.3 |
| Dense | 64 neurons, ReLU |
| Dropout | 0.2 |
| Dense | 32 neurons, ReLU |
| Output | 1 neuron |

## Results

| Model | Validation RMSE |
|---|---|
| Lasso + Splines (baseline) | 0.1150 |
| Default NN | 0.7211 |
| NN + Random Search | 0.8110 |

RMSE is measured in log(SalePrice) units.

> Hyperparameter tuning did not improve over the default model in this run, demonstrating the instability of neural networks on small datasets.

## Key Findings

- The neural network performs substantially worse than the Lasso baseline, with a validation RMSE several times higher.
- The default model shows severe overfitting and high sensitivity to random initialization — the same architecture with different seeds produces substantially different results across runs.
- Hyperparameter tuning improves training stability but does not close the gap with the baseline.
- On a small tabular dataset (~1,500 samples), a simple regularized linear model generalizes better than a deep learning approach.

## Note on Reproducibility

Due to the stochastic nature of neural network training, results may vary slightly between runs even with fixed seeds. All results documented in the notebook correspond to a single fixed run.

## Libraries
- `pandas`, `numpy` — data manipulation
- `tensorflow`, `keras` — neural network training
- `scikit-learn` — preprocessing, train/val split
- `matplotlib` — visualization
- `joblib`, `pickle` — loading pipeline components from the previous project
