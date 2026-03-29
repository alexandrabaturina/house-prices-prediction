# House Price Prediction

A series of three projects exploring progressively complex modeling approaches on the housing dataset: 1460 training and 1459 test records describing residential properties. The target variable is `SalePrice`.

## Structure
```
├── 01_linear_models/
├── 02_nonlinear_modeling/
├── 03_neural_network/
├── artifacts/          # trained pipelines and column lists
├── data/
│   ├── raw/            # original datasets
│   ├── processed/      # cleaned datasets
│   └── data_description.txt
└── requirements.txt
```

## Projects

### [01. Linear Models](01_linear_models/)
- EDA, feature engineering, and regularized regression.
- Lasso with cross-validated alpha selection achieved a validation RMSE of 0.1160.
- A nonlinear relationship between `HouseAge` and `SalePrice` was identified as a motivation for the follow-up project.

### [02. Nonlinear Modeling](02_nonlinear_modeling/)
- Polynomial Regression, Step Functions, and Regression Splines compared across three features.
- Spline transformations for `HouseAge` and `TotalBsmtSF` were integrated into the Lasso pipeline, improving validation RMSE to 0.1150 with a re-tuned alpha.

### [03. Neural Network](03_neural_network/)
- Feedforward neural network compared against the Lasso + Splines baseline.
- On this dataset (~1,500 samples), the regularized linear model outperforms deep learning — the neural network achieved a validation RMSE of 0.7211 vs 0.1150 for the baseline.

## Results Summary

| Model | Validation RMSE | Gap |
|---|---|---|
| Linear Regression | 0.1289 | 0.041 |
| Ridge (α=19.47) | 0.1171 | 0.0160 |
| Lasso (α=0.00066) | 0.1160 | 0.0120 |
| Lasso + Splines (α=0.000762) | 0.1150 | 0.0107 |
| Neural Network (default) | 0.7211 | — |
| Neural Network + Random Search | 0.8110 | — |

RMSE is measured in log(SalePrice) units.

## Key Takeaways
- On small tabular datasets, well-regularized linear models generalize better than deep learning.
- Nonlinear feature transformations can improve model performance, but the gains are modest when the transformed features are not among the strongest predictors.

## Setup
```bash
pip install -r requirements.txt
```
