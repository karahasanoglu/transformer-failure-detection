# Predictive Maintenance for Distribution Transformers

This project is an academic predictive maintenance study that investigates fault/risk behavior in distribution transformers using data mining and machine learning methods. The workflow derives engineering features from 2019 and 2020 transformer data, generates pseudo labels with unsupervised methods, and classifies these labels with an SVM model.

Reference problem domain:

`Vita, V.; Fotis, G.; Chobanov, V.; Pavlatos, C.; Mladenov, V. Predictive Maintenance for Distribution System Operators in Increasing Transformers' Reliability. Electronics 2023, 12, 1356.`

This repository is not a one-to-one reproduction of the paper. In the current codebase, the `Burned transformers 2019/2020` columns in the raw Excel files are not used as the model target. They are removed during preprocessing to reduce possible target leakage and direct label dependency. After that, `KMeans` and `IsolationForest` are used to generate pseudo risk labels.

## Project Goal

The main goal of the project is to build an interpretable experimental pipeline that can separate potentially risky transformer records from technical and environmental variables. In this scope:

- raw 2019 and 2020 Excel files are cleaned,
- categorical variables are converted into numerical features,
- new features are produced for transformer load, lightning risk, network density, and historical risk interaction,
- unsupervised pseudo labeling is performed after removing the real `burned_transformers` columns,
- RBF-kernel SVM models are trained on normalized data,
- outputs are evaluated with confusion matrix plots and classification reports.

Therefore, this project should be considered an academic/prototype-level predictive maintenance modeling study, not a production system.

## Dataset

Raw data is stored under `data/raw/`:

```text
data/raw/
├── Dataset_Year_2019.xlsx
└── Dataset_Year_2020.xlsx
```

Both files contain `15873` rows and `16` raw columns. The raw columns are:

- `LOCATION`
- `POWER`
- `SELF-PROTECTION`
- `Average earth discharge density DDT [Rays/km^2-año]`
- `Maximum ground discharge density DDT [Rays/km^2-año]`
- `Burning rate  [Failures/year]`
- `Criticality according to previous study for ceramics level`
- `Removable connectors`
- `Type of clients`
- `Number of users`
- `Electric power not supplied EENS [kWh]`
- `Type of installation`
- `Air network`
- `Circuit Queue`
- `km of network LT:`
- `Burned transformers 2019` or `Burned transformers 2020`

In the code, column names are first converted to lowercase, special characters are removed, and spaces are replaced with `_`. For example, `SELF-PROTECTION` becomes `selfprotection`; `km of network LT:` becomes `km_of_network_lt`; and `Burned transformers 2019` becomes `burned_transformers_2019`.

## Preprocessing

The preprocessing workflow is defined in [src/preprocessing.py](/home/emirfurkan/Desktop/transformer_predictive_maintenance/src/preprocessing.py:1). The script reads both raw Excel files, standardizes the column names, and removes the year-specific real burned-transformer labels:

- `burned_transformers_2019` for 2019
- `burned_transformers_2020` for 2020

This choice is important: the current modeling pipeline does not perform supervised fault prediction with the real target labels. Instead, after removing the real target columns, it generates a pseudo risk label from the remaining features.

Engineered features created during preprocessing:

- `power_per_user = power / (number_of_users + 1)`
- `lightning_risk_score = average_earth_discharge_density_ddt_rayskm2ao * (2 - selfprotection)`
- `network_density = km_of_network_lt / (power + 1)`
- `historical_risk_index = burning_rate__failuresyear * criticality_according_to_previous_study_for_ceramics_level`

Then, `type_of_clients` and `type_of_installation` are converted into one-hot encoded columns using `pd.get_dummies`. The following columns are dropped at the end of preprocessing:

- `location`
- `maximum_ground_discharge_density_ddt_rayskm2ao`
- `electric_power_not_supplied_eens_kwh`
- `circuit_queue`
- `air_network`

In the 2020 dataset, `type_of_installation_POLE WITH ANTI-FRAU NET` is also renamed to `type_of_installation_POLE WITH ANTI-FRAUD NET`. This keeps the processed 2019 and 2020 datasets aligned under the same column schema.

Preprocessing outputs:

```text
data/processed/
├── processed_dataset_2019.csv
└── processed_dataset_2020.csv
```

Both processed files have shape `15873 x 29` and do not yet contain a `target` column.

## Correlation Analysis

The preprocessing script generates correlation matrices for numerical variables and saves them under `results/`:

```text
results/
├── 2019 Correlation Matrix.png
└── 2020 Correlation Matrix.png
```

These plots are used to inspect linear relationships among features. They are especially useful for checking how strongly the engineered features relate to the raw variables. However, a correlation matrix alone is not evidence of causality or model success; it should be interpreted only as an exploratory data analysis output.

## Pseudo Labeling

The pseudo-labeling workflow is defined in [src/labelling.py](/home/emirfurkan/Desktop/transformer_predictive_maintenance/src/labelling.py:1). In this step, `processed_dataset_2019.csv` and `processed_dataset_2020.csv` are loaded, boolean columns are converted into `0/1` integers, and all features are scaled with `StandardScaler`.

Scaling in this stage is performed as follows:

- the scaler is learned with `fit_transform` on the processed 2019 data,
- the same scaler is applied to the processed 2020 data with `transform` only.

This means the 2020 data is evaluated using the 2019 reference distribution. From a temporal perspective, this is reasonable because it avoids relearning the scaler from a future year's statistics.

The pseudo target is generated by combining two unsupervised signals:

1. `KMeans(n_clusters=2, random_state=42, n_init=10)`
2. `IsolationForest(contamination=0.05, random_state=42)`

For KMeans, the model is fitted on the scaled 2019 data. Cluster predictions for 2020 are obtained with the same KMeans model. In the 2019 data, the average distance to cluster centers is computed for each cluster; the cluster with the higher mean distance is treated as the anomaly/risk cluster.

For IsolationForest, the model is also fitted on the scaled 2019 data and then used to predict on the 2020 data. Observations with IsolationForest output `-1` are considered anomalies.

The final pseudo target is:

```text
target = 1 if KMeans_anomaly == 1 or IsolationForest_anomaly == 1 else 0
```

In other words, if either method marks an observation as risky, the final pseudo label becomes `1`.

Pseudo-labeling outputs:

```text
data/processed/
├── 2019_pseudo.csv
└── 2020_pseudo.csv
```

Pseudo target distributions:

| Dataset | target=0 | target=1 | Risk rate |
|---|---:|---:|---:|
| 2019 | 12736 | 3137 | 19.76% |
| 2020 | 12778 | 3095 | 19.50% |

These rates should not be interpreted as real fault rates. They represent the proportion of observations that the unsupervised model considers risky/anomalous.

## Normalization

The normalization workflow is defined in [src/normalization.py](/home/emirfurkan/Desktop/transformer_predictive_maintenance/src/normalization.py:1). This script loads the pseudo-labeled files and normalizes only a selected set of numerical variables.

Normalized numerical columns:

- `power`
- `average_earth_discharge_density_ddt_rayskm2ao`
- `burning_rate__failuresyear`
- `number_of_users`
- `km_of_network_lt`
- `power_per_user`
- `lightning_risk_score`
- `network_density`
- `historical_risk_index`

The method used is `sklearn.preprocessing.StandardScaler`. The standardization formula is:

```text
z = (x - mean) / standard_deviation
```

In this script, the scaler is applied to the 2019 pseudo data with `fit_transform`; the 2020 pseudo data is transformed with the same scaler. Thus, 2020 features are standardized relative to the 2019 distribution.

During normalization, boolean one-hot columns are converted into `0/1` integers. The `target` column is not normalized and is preserved as the class label.

Normalization outputs:

```text
data/processed/
├── normalized_dataset_2019.csv
└── normalized_dataset_2020.csv
```

Both files have shape `15873 x 30`: `29` features and `1` pseudo target column.

## Modeling

Model scripts are located under `src/models/`:

```text
src/models/
├── SVM_2019.py
├── SVM_2020.py
└── SVM-2019-2020.py
```

All model scripts use the normalized CSV files. The modeling setup is:

- target variable: `target`
- features: all columns except `target`
- model: `sklearn.svm.SVC`
- kernel: `rbf`
- class balancing: `class_weight="balanced"`
- `C=1.0`
- `gamma="scale"`; in the combined 2019-2020 script, gamma is not explicitly passed, so the sklearn default is still `scale`
- train/test split: `train_test_split(test_size=0.2, random_state=42)`

Note: The current model scripts do not use `stratify=y`. Since the pseudo target ratio is around 20%, the resulting split still appears balanced; however, adding a stratified split is recommended for a more controlled academic evaluation.

### 2019 SVM Output

[src/models/SVM_2019.py](/home/emirfurkan/Desktop/transformer_predictive_maintenance/src/models/SVM_2019.py:1) runs on `normalized_dataset_2019.csv`.

Confusion matrix:

```text
[[2490   63]
 [   3  619]]
```

Classification report:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 | 0.9988 | 0.9753 | 0.9869 | 2553 |
| 1 | 0.9076 | 0.9952 | 0.9494 | 622 |

Overall accuracy is `0.9792`, macro F1 is `0.9682`, and weighted F1 is `0.9796`.

Generated plot:

```text
results/2019-SVM-confusion_matrix.png
```

### 2020 SVM Output

[src/models/SVM_2020.py](/home/emirfurkan/Desktop/transformer_predictive_maintenance/src/models/SVM_2020.py:1) runs on `normalized_dataset_2020.csv`.

Confusion matrix:

```text
[[2491   65]
 [   6  613]]
```

Classification report:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 | 0.9976 | 0.9746 | 0.9859 | 2556 |
| 1 | 0.9041 | 0.9903 | 0.9453 | 619 |

Overall accuracy is `0.9776`, macro F1 is `0.9656`, and weighted F1 is `0.9780`.

Generated plot:

```text
results/2020-SVM-confusion_matrix.png
```

### Combined 2019-2020 SVM Output

[src/models/SVM-2019-2020.py](/home/emirfurkan/Desktop/transformer_predictive_maintenance/src/models/SVM-2019-2020.py:1) combines the 2019 and 2020 normalized datasets. The script temporarily adds a `yil` column, but removes both `target` and `yil` from the feature matrix during model training.

Confusion matrix:

```text
[[4995   95]
 [   9 1251]]
```

Classification report:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 | 1.00 | 0.98 | 0.99 | 5090 |
| 1 | 0.93 | 0.99 | 0.96 | 1260 |

Overall accuracy is approximately `0.98`, macro F1 is approximately `0.97`, and weighted F1 is approximately `0.98`.

Generated plot:

```text
results/2019-2020-SVM-confusion_matrix.png
```

## Results Directory

The current `results/` directory contains:

```text
results/
├── 2019 Correlation Matrix.png
├── 2020 Correlation Matrix.png
├── 2019-SVM-confusion_matrix.png
├── 2020-SVM-confusion_matrix.png
└── 2019-2020-SVM-confusion_matrix.png
```

The correlation matrices are preprocessing/exploratory analysis outputs. The SVM confusion matrix plots summarize classification behavior on the test split. In a confusion matrix, rows represent the true class and columns represent the predicted class:

- top-left: true negative
- top-right: false positive
- bottom-left: false negative
- bottom-right: true positive

In this project, the positive class is `target=1`, meaning a transformer record considered risky/anomalous by the pseudo-labeling process.

## Running Order

The current codebase is script-based. For a clean workflow, the recommended execution order is:

```bash
.venv/bin/python src/preprocessing.py
.venv/bin/python src/labelling.py
.venv/bin/python src/normalization.py
.venv/bin/python src/models/SVM_2019.py
.venv/bin/python src/models/SVM_2020.py
.venv/bin/python src/models/SVM-2019-2020.py
```

If the `python` command is not available on the system, use `.venv/bin/python` or `python3` in a properly configured environment.

## Installation

The project defines Python `>=3.12` and the following main dependencies in `pyproject.toml`:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `openpyxl`
- `imblearn`
- `tensorflow`

If the virtual environment already exists:

```bash
.venv/bin/python -m pip install -e .
```

Alternatively, the core packages can be installed directly:

```bash
python3 -m pip install pandas numpy scikit-learn matplotlib seaborn openpyxl imbalanced-learn
```

## Directory Structure

```text
transformer_predictive_maintenance/
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
│   ├── 2019-SVM-confusion_matrix.png
│   ├── 2020-SVM-confusion_matrix.png
│   └── 2019-2020-SVM-confusion_matrix.png
├── src/
│   ├── preprocessing.py
│   ├── labelling.py
│   ├── normalization.py
│   └── models/
│       ├── SVM_2019.py
│       ├── SVM_2020.py
│       └── SVM-2019-2020.py
├── pyproject.toml
├── readme.md
└── readme_en.md
```

## Methodological Assessment

The model results look strong, but they should not be interpreted as direct real fault-prediction performance. The main reason is that the SVM model learns the pseudo `target` labels generated by KMeans and IsolationForest, not the real `Burned transformers` labels. Therefore, the SVM largely behaves as a classifier that reproduces the unsupervised anomaly definition.

This approach is academically useful because it:

- tests the idea of generating a risk group without relying on real labels,
- shows whether unsupervised labels are separable by a supervised classifier,
- helps examine whether different years share a similar risk structure in the same feature space.

However, the limitations are also clear:

- pseudo labels are not sufficient for operational decisions unless validated against real field failures,
- `train_test_split` does not account for temporal dependency,
- the model scripts do not currently use stratified splitting,
- all file paths are hardcoded as absolute paths; they should be converted to relative paths for portability,
- there is no separate production pipeline for direct prediction on a new year such as 2021,
- critical predictive maintenance variables such as temperature, load history, oil analysis, maintenance logs, and environmental time series are not included in the dataset.

## Conclusion

This project treats predictive maintenance for distribution transformers as an end-to-end experimental pipeline: from raw data to feature engineering, pseudo labeling, normalization, and SVM-based classification. The current outputs show that the pseudo risk class can be separated with high performance by the SVM model.

Still, the results should be evaluated as pseudo anomaly/risk classification rather than real fault prediction. For real-world field use, the model must be revalidated with real failure labels, time-based validation, and richer operational sensor/maintenance data.
