# 2019-2020 Transformer Risk and Pseudo-Anomaly Prediction

This project is a machine learning study for pseudo-labelled transformer risk/anomaly prediction using transformer and distribution-network data from 2019 and 2020. The raw datasets include the columns `Burned transformers 2019` and `Burned transformers 2020`; however, these columns are not considered reliable field labels. Therefore, they are removed from the modelling pipeline, and the target variable is regenerated as a pseudo label using KMeans and Isolation Forest.

In this study, the `target` variable is not a verified failure record. It is an algorithmically generated risk/anomaly indicator:

| Label | Meaning |
|---:|---|
| `0` | Normal or lower-risk observation |
| `1` | Pseudo-anomalous or higher-risk observation |

This distinction is essential: model performance metrics should not be interpreted as direct field failure prediction performance. They should be interpreted as the model's ability to learn the target structure produced by the pseudo-labelling mechanism.

## Project Objective

The main objective of this study is to construct an alternative target variable that can represent transformer risk without relying on unreliable direct failure labels, and then compare different model families using this target variable. In this scope:

- Unreliable burn/failure columns are removed from the raw Excel files.
- Numerical and categorical transformer/network variables are cleaned.
- New risk-related features are engineered.
- Pseudo labelling is performed with KMeans and Isolation Forest.
- The 2019 dataset is used for training/validation, and the 2020 dataset is used as a forward-year test set.
- Random Forest, XGBoost, and Autoencoder approaches are compared.

## Project Structure

```text
.
├── data/
│   ├── raw/
│   │   ├── Dataset_Year_2019.xlsx
│   │   └── Dataset_Year_2020.xlsx
│   └── processed/
│       ├── processed_dataset_2019.csv
│       ├── processed_dataset_2020.csv
│       ├── 2019_pseudo.csv
│       ├── 2020_pseudo.csv
│       ├── normalized_dataset_2019.csv
│       └── normalized_dataset_2020.csv
├── results/
│   ├── 2019 Correlation Matrix.png
│   ├── 2020 Correlation Matrix.png
│   ├── tsne_2019.png
│   ├── tsne_2020.png
│   ├── RandomForest-ConfusionMatrix.png
│   ├── XG-Boost-ConfusionMatrix.png
│   ├── AutoEncoder-ConfusionMatrix.png
│   ├── burned_transformers_2019_correlation.png
│   └── burned_transformers_2020_correlation.png
├── src/
│   ├── preprocessing.py
│   ├── labelling.py
│   ├── normalization.py
│   ├── check_distrubition.py
│   └── models/
│       ├── random_forrest.py
│       ├── XG_Boost.py
│       └── autoencoder_anomally.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Datasets

Raw data files:

- `data/raw/Dataset_Year_2019.xlsx`
- `data/raw/Dataset_Year_2020.xlsx`

Both raw files contain `15873` observations. The datasets include transformer and distribution-network characteristics such as transformer power, protection status, lightning density, client type, installation type, number of users, network length, and related operational attributes.

Raw columns:

```text
LOCATION
POWER
SELF-PROTECTION
Average earth discharge density DDT [Rays/km^2-ano]
Maximum ground discharge density DDT [Rays/km^2-ano]
Burning rate [Failures/year]
Criticality according to previous study for ceramics level
Removable connectors
Type of clients
Number of users
Electric power not supplied EENS [kWh]
Type of installation
Air network
Circuit Queue
km of network LT
Burned transformers 2019 / Burned transformers 2020
```

The original `Burned transformers` columns are not used as target variables in this study. They are considered misleading and are removed directly during preprocessing.

### Processed Data Files

| File | Shape | Content | Target Status |
|---|---:|---|---|
| `processed_dataset_2019.csv` | `15873 x 29` | Cleaned and feature-engineered 2019 feature set | None |
| `processed_dataset_2020.csv` | `15873 x 29` | Cleaned and feature-engineered 2020 feature set | None |
| `2019_pseudo.csv` | `15873 x 30` | 2019 feature set with pseudo `target` | Present |
| `2020_pseudo.csv` | `15873 x 30` | 2020 feature set with pseudo `target` | Present |
| `normalized_dataset_2019.csv` | `15873 x 30` | Final normalized 2019 model dataset | Present |
| `normalized_dataset_2020.csv` | `15873 x 30` | Final normalized 2020 model dataset | Present |

Pseudo-label distribution:

| Dataset | `target=0` | `target=1` | Total | `target=1` Ratio |
|---|---:|---:|---:|---:|
| 2019 pseudo | 12736 | 3137 | 15873 | 19.76% |
| 2020 pseudo | 12778 | 3095 | 15873 | 19.50% |

This distribution differs from the original burn-label distribution because the new target is not a verified burn/failure record. It is an algorithmic risk/anomaly label.

## Preprocessing

Preprocessing is implemented in [src/preprocessing.py](src/preprocessing.py).

Main steps:

- Column names are converted to lowercase.
- Special characters are removed.
- Spaces are replaced with `_`.
- `Burned transformers 2019` and `Burned transformers 2020` are removed.
- Categorical variables are transformed with one-hot encoding.
- Some low-usability or unnecessary columns are excluded from modelling.
- Correlation matrices are generated for numerical variables.

Dropped columns:

```text
location
maximum_ground_discharge_density_ddt_rayskm2ao
electric_power_not_supplied_eens_kwh
circuit_queue
air_network
```

Engineered features:

| Feature | Description |
|---|---|
| `power_per_user` | Transformer power normalized by the number of users |
| `lightning_risk_score` | Interaction between average lightning density and self-protection |
| `network_density` | LT network length expressed relative to transformer power |
| `historical_risk_index` | Interaction between historical burning rate and criticality level |

## Pseudo Labelling

Pseudo labelling is implemented in [src/labelling.py](src/labelling.py). At this stage, `processed_dataset_2019.csv` and `processed_dataset_2020.csv` are used; therefore, the algorithms never see the old burn-label columns.

Pseudo-labelling process:

1. The 2019 and 2020 feature sets are loaded.
2. Boolean columns are converted to numerical type.
3. `StandardScaler` is fitted on the 2019 feature set and applied to the 2020 feature set.
4. KMeans is trained on the 2019 data.
5. The 2020 data is labelled using the KMeans model trained on 2019.
6. For KMeans, the cluster with the higher average distance from its centroid is treated as the anomalous cluster.
7. Isolation Forest is trained on the 2019 data and used to predict anomalies for 2019 and 2020.
8. The final pseudo target is generated with the following rule:

```text
target = 1 if KMeans or Isolation Forest marks the observation as anomalous/risky
target = 0 otherwise
```

This approach combines cluster-structure information and isolation-based anomaly signals instead of relying on a single algorithm. However, although pseudo labels provide a practical modelling target, they should not be treated as verified field failure labels.

## Normalization

Normalization is implemented in [src/normalization.py](src/normalization.py). It is applied after pseudo labelling, using `2019_pseudo.csv` and `2020_pseudo.csv`.

`StandardScaler` is fitted only on the 2019 dataset and the same transformation is applied to the 2020 dataset. This design prevents information leakage from the forward-year test set into the training process.

Normalized numerical variables:

```text
power
average_earth_discharge_density_ddt_rayskm2ao
burning_rate__failuresyear
number_of_users
km_of_network_lt
power_per_user
lightning_risk_score
network_density
historical_risk_index
```

One-hot encoded categorical variables and the pseudo `target` column are not normalized.

## Experimental Design

This project uses a time-based experimental design:

| Role | Data |
|---|---|
| Training and validation | 2019 data |
| Forward-year test | 2020 data |

In this setup, the model learns from the structure of the previous year and is evaluated on the pseudo risk/anomaly labels of the following year. This design provides a more realistic generalization scenario than a random train-test split.

## Model Analyses

### Random Forest

File: [src/models/random_forrest.py](src/models/random_forrest.py)

The Random Forest model is trained on the normalized 2019 dataset and tested on the normalized 2020 dataset. The model produces class probabilities, and then `precision_recall_curve` is used to select the threshold that maximizes the F1 score.

Model settings:

```text
n_estimators = 100
max_depth = 10
random_state = 42
```

Test result:

| Metric | Value |
|---|---:|
| Best threshold | 0.3651 |
| Best F1 score | 0.9770 |
| Accuracy | 0.9911 |

Confusion matrix:

```text
[[12714    64]
 [   78  3017]]
```

Class-wise report:

| Class | Precision | Recall | F1-score | Support |
|---:|---:|---:|---:|---:|
| 0 | 0.99 | 0.99 | 0.99 | 12778 |
| 1 | 0.98 | 0.97 | 0.98 | 3095 |

Top 10 most important features:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `type_of_clients_STRATUM 1` | 0.2889 |
| 2 | `type_of_installation_POLE` | 0.1244 |
| 3 | `power` | 0.1031 |
| 4 | `type_of_clients_STRATUM 2` | 0.0735 |
| 5 | `type_of_installation_MACRO WITHOUT ANTI-FRAUD NET` | 0.0648 |
| 6 | `number_of_users` | 0.0613 |
| 7 | `lightning_risk_score` | 0.0442 |
| 8 | `average_earth_discharge_density_ddt_rayskm2ao` | 0.0377 |
| 9 | `removable_connectors` | 0.0357 |
| 10 | `network_density` | 0.0304 |

Interpretation: Random Forest separates the pseudo labels with very high performance. The prominence of client type, installation type, and transformer power suggests that the pseudo-labelling mechanism is strongly associated with these structural variables. However, this performance should not be interpreted as validated field failure prediction performance.

### XGBoost

File: [src/models/XG_Boost.py](src/models/XG_Boost.py)

XGBoost is used to learn the pseudo target with gradient-boosted decision trees. The model is trained on 2019 data and tested on 2020 data.

Model settings:

```text
n_estimators = 500
max_depth = 8
learning_rate = 0.01
random_state = 42
```

Test result:

| Metric | Value |
|---|---:|
| Accuracy | 0.9915 |

Confusion matrix:

```text
[[12755    23]
 [  112  2983]]
```

Class-wise report:

| Class | Precision | Recall | F1-score | Support |
|---:|---:|---:|---:|---:|
| 0 | 0.99 | 1.00 | 0.99 | 12778 |
| 1 | 0.99 | 0.96 | 0.98 | 3095 |

Interpretation: XGBoost produces fewer false positives than Random Forest, but it has slightly more false negatives for `target=1`. This suggests that XGBoost is more selective when predicting the pseudo risk class.

### Autoencoder Anomaly Detection

File: [src/models/autoencoder_anomally.py](src/models/autoencoder_anomally.py)

The Autoencoder approach differs from the two supervised classifiers. Instead of directly learning both `target` classes, it learns to reconstruct normal observations (`target=0`) from the 2019 training data. Observations with high reconstruction error are then classified as anomalous/risky.

Architecture:

```text
Input -> Dense(32) -> Dense(16) -> Latent(4)
      -> Dense(16) -> Dense(32) -> Output
```

Training settings:

```text
loss = mse
optimizer = adam
epochs = 50
batch_size = 32
early stopping patience = 5
```

Anomaly decision rule:

```text
if reconstruction_error > threshold_p95, then target=1
if reconstruction_error <= threshold_p95, then target=0
```

Final values observed in the latest run:

| Value | Result |
|---|---:|
| Final training loss | 0.01142 |
| Validation error mean | 0.01146 |
| Threshold p95 | 0.04877 |
| Test error mean | 0.05129 |
| Accuracy | 0.9214 |

Confusion matrix:

```text
[[11673  1105]
 [  143  2952]]
```

Class-wise report:

| Class | Precision | Recall | F1-score | Support |
|---:|---:|---:|---:|---:|
| 0 | 0.9879 | 0.9135 | 0.9493 | 12778 |
| 1 | 0.7276 | 0.9538 | 0.8255 | 3095 |

Interpretation: The Autoencoder produces high recall for the `target=1` pseudo risk class. This indicates a strong tendency to avoid missing observations marked as risky/anomalous. However, its precision is lower than Random Forest and XGBoost, meaning that it marks more normal observations as risky. In risk-screening problems, high recall may be desirable, but it also introduces a higher false-positive cost.

## Model Comparison

| Model | Accuracy | Target=1 Precision | Target=1 Recall | Target=1 F1 | Interpretation |
|---|---:|---:|---:|---:|---|
| Random Forest | 0.9911 | 0.98 | 0.97 | 0.98 | Balanced and high performance |
| XGBoost | 0.9915 | 0.99 | 0.96 | 0.98 | Lower false positives, slightly higher false negatives |
| Autoencoder | 0.9214 | 0.7276 | 0.9538 | 0.8255 | Strong at capturing the risky class, with more false positives |

Overall, the supervised models, Random Forest and XGBoost, learn the pseudo labels with very high performance. The Autoencoder has lower overall accuracy, but provides high recall for `target=1`. Therefore, the Autoencoder is more suitable as a risk-screening or prioritization method than as a strict classifier.

## Results Folder Outputs

The `results/` folder contains model and data-analysis outputs.

| File | Description | Current Pipeline Status |
|---|---|---|
| `2019 Correlation Matrix.png` | Numerical correlation matrix for the 2019 feature set | Current |
| `2020 Correlation Matrix.png` | Numerical correlation matrix for the 2020 feature set | Current |
| `tsne_2019.png` | Two-dimensional t-SNE projection of the normalized 2019 pseudo dataset | Current |
| `tsne_2020.png` | Two-dimensional t-SNE projection of the normalized 2020 pseudo dataset | Current |
| `RandomForest-ConfusionMatrix.png` | Random Forest test confusion matrix plot | Current |
| `XG-Boost-ConfusionMatrix.png` | XGBoost test confusion matrix plot | Current |
| `AutoEncoder-ConfusionMatrix.png` | Autoencoder test confusion matrix plot | Current |
| `burned_transformers_2019_correlation.png` | Older correlation analysis using the original target | Legacy/previous experiment |
| `burned_transformers_2020_correlation.png` | Older correlation analysis using the original target | Legacy/previous experiment |

Note: `burned_transformers_2019_correlation.png` and `burned_transformers_2020_correlation.png` were produced in an earlier analysis using the old target columns. Since the current pipeline removes the original `Burned transformers` columns, these two figures should not be interpreted as part of the current modelling results.

## How to Run

The project runs with Python `>=3.12`. Dependencies are defined in [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock).

Install the environment:

```bash
uv sync
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Recommended execution order:

```bash
python src/preprocessing.py
python src/labelling.py
python src/normalization.py
python src/check_distrubition.py
python src/models/random_forrest.py
python src/models/XG_Boost.py
python src/models/autoencoder_anomally.py
```

Direct execution in the current local environment:

```bash
.venv/bin/python src/preprocessing.py
.venv/bin/python src/labelling.py
.venv/bin/python src/normalization.py
.venv/bin/python src/check_distrubition.py
.venv/bin/python src/models/random_forrest.py
.venv/bin/python src/models/XG_Boost.py
.venv/bin/python src/models/autoencoder_anomally.py
```

If TensorFlow cannot find GPU libraries, it will continue running on CPU. Messages such as `Cannot dlopen some GPU libraries` or `Skipping registering GPU devices` indicate that GPU acceleration is not being used; they do not prevent the model from running on CPU.

## Dependencies

Main libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `imbalanced-learn`
- `xgboost`
- `tensorflow`
- `matplotlib`
- `seaborn`
- `openpyxl`

## Academic Interpretation and Limitations

This study uses a pseudo-labelling approach for transformer risk/anomaly prediction instead of relying on unreliable direct failure labels. This provides a consistent modelling target, but the generated `target` column should not be considered a field-validated failure label.

Main limitations:

- `target=1` does not mean a confirmed failure or burn event; it indicates an algorithmic anomaly/risk signal.
- Model metrics are computed against the pseudo target.
- Before the results are interpreted as real field failure prediction, they should be calibrated with expert validation or reliable failure records.
- The dataset is not a DGA time-series dataset; therefore, it is not sufficient for direct RUL or "days until failure" forecasting.
- The 2019-2020 forward-year setup is more suitable for temporal generalization than a random split, but long-term generalization remains limited because only two years are available.
- The scripts use absolute file paths. For execution in a different environment, paths can be refactored relative to the project root.

In conclusion, this project is an experimental risk modelling study that combines pseudo labelling, classical supervised models, and autoencoder-based anomaly detection on transformer data. Random Forest and XGBoost learn the pseudo labels with high performance, while the Autoencoder provides a more sensitive anomaly-detection alternative focused on capturing the risky class.
