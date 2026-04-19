# Power Transformer RUL and FDD Prediction

This project is a data science and machine learning study developed to support predictive maintenance in power transformers. Its primary objective is to address two critical tasks simultaneously by using Dissolved Gas Analysis (DGA) time-series data: Fault Detection and Diagnosis (FDD) and Remaining Useful Life (RUL) prediction.

In industrial energy systems, transformer failures may lead to substantial maintenance costs, unplanned downtime, and reduced operational reliability. Within this context, early warning and lifetime estimation play a crucial role in enabling data-driven maintenance strategies. This repository brings together both classical machine learning methods and deep learning-based time-series approaches under a unified experimental framework.

## Scope of the Study

The project focuses on the following two analytical tasks:

1. `FDD`: Identification of the fault category based on the observed gas behavior of the transformer.
2. `RUL`: Estimation of the remaining useful lifetime of the equipment from its operational history.

Accordingly, the repository includes a modular workflow covering data loading, validation, fixed-length sequence generation, feature extraction, class distribution analysis, model training, and evaluation.

## Dataset and Problem Definition

The dataset consists of CSV files, each representing a time-series sample belonging to a single transformer. Every sample contains four core DGA variables:

| Variable | Description |
| --- | --- |
| `H2` | Hydrogen |
| `CO` | Carbon Monoxide |
| `C2H4` | Ethylene |
| `C2H2` | Acetylene |

Each raw sample is a multivariate time series measured over time. During preprocessing, all samples are converted into a common representation:

```text
(sequence_length, feature_count) = (200, 4)
```

The raw data and label structure are organized as follows:

```text
data/raw/
├── data_train/
├── data_test/
└── data_labels/
```

The label files include two target variables:

| Label | Description |
| --- | --- |
| `FDD label` | Fault class |
| `RUL label` | Remaining useful life |

## Dataset Size

The repository currently contains the following data scale:

| Split | Number of samples |
| --- | ---: |
| Raw training files | 2100 |
| Raw test files | 900 |
| Processed training set | 1680 |
| Processed validation set | 420 |
| Processed test set | 900 |

The processed tensor shapes are as follows:

```text
train: (1680, 200, 4)
val  : (420, 200, 4)
test : (900, 200, 4)
```

The observed RUL range in the training set is:

```text
min = 362.0
max = 1093.0
mean = 785.89526
```

## Data Processing and Experimental Pipeline

The overall workflow adopted in the project can be summarized as follows:

```text
Raw CSV files
-> Data loading
-> Data validation
-> Sequence standardization
-> Feature extraction / normalization
-> Model training
-> Evaluation
```

Although the two prediction tasks rely on different modeling strategies, both are built upon a shared data preparation pipeline.

## Project Architecture

The codebase is organized as follows:

```text
src/
├── data_preprocessing/
│   ├── load_data.py
│   ├── validate_data.py
│   └── build_sequences.py
└── models/
    ├── check_class_distrubiton.py
    ├── fdd_prediction_models/
    │   ├── train_grnn_fdd.py
    │   └── train_random_forrest_fdd.py
    └── rul_predictiction_model/
        ├── train_gru.py
        └── train_lstm.py
```

## Preprocessing Components

### `load_data.py`

This module reads the raw CSV files and label files and creates a structured data object for each sample. Each object contains:

- file name,
- split information (`train` or `test`),
- operating condition,
- transformer identifier,
- time-series dataframe,
- FDD label,
- RUL label.

The operating condition and transformer identifier are extracted from the file names, after which the labels are aligned with each sample.

### `validate_data.py`

This module performs systematic data quality checks, including:

- whether the data object has the expected type,
- whether the file name has a valid format,
- whether the dataframe is empty,
- whether all required gas columns are present,
- whether the gas columns are numeric,
- whether missing values exist,
- whether both FDD and RUL labels are present and numeric.

This step is important for supporting reproducibility and preventing silent data-related errors during experimentation.

### `build_sequences.py`

This module converts all time series into fixed-length sequences. If a sample contains fewer than `200` time steps, zero-padding is added to the beginning. If it is longer, only the last `200` time steps are preserved. In this way, all samples are transformed into a common tensor representation suitable for model training.

In addition, the training set is split into a validation set using `train_test_split`:

- training ratio: `80%`
- validation ratio: `20%`
- random seed: `42`

The outputs are stored in `.npy` format in the following directories:

```text
data/processed_merged/
├── train_set/
├── val_set/
└── test_set/
```

## Class Imbalance Analysis

For the FDD task, the training set exhibits a substantial class imbalance. This distribution is reported in `check_class_distrubiton.py`. The processed training set currently follows the distribution below:

| Class | Sample count | Percentage |
| --- | ---: | ---: |
| 1 | 1367 | 81.37 |
| 2 | 69 | 4.11 |
| 3 | 94 | 5.60 |
| 4 | 150 | 8.93 |

This observation indicates that FDD evaluation should not rely solely on overall accuracy; class-wise reports and confusion matrices are also essential.

## FDD Models

The FDD task is formulated as a multi-class classification problem. In both FDD models, the raw time series are not used directly; instead, sample-level feature extraction is performed.

### Feature Set

For each gas variable, the following statistical summaries are computed:

- mean,
- standard deviation (`std`),
- maximum,
- minimum,
- last observed value.

Across four gases, this produces `20` base features. In addition, four ratio-based features are derived from the last observations:

```text
R1 = H2 / CO
R2 = C2H2 / C2H4
R3 = H2 / C2H4
R4 = CO / C2H2
```

Thus, the FDD pipeline uses a total of `24` features. These ratios are consistent with commonly used diagnostic relationships in the DGA literature.

### `train_grnn_fdd.py`

This module implements a kernel-based classifier inspired by a General Regression Neural Network. Its main steps are:

- loading all CSV files from the training and test folders,
- extracting statistical and ratio-based features,
- standardizing the features using `StandardScaler`,
- training the GRNN-like classifier with `sigma = 0.5`,
- reporting the confusion matrix and classification report.

This model serves as a useful baseline, particularly for relatively compact and structured feature spaces.

### `train_random_forrest_fdd.py`

This module adopts a tree-ensemble strategy that is more robust to class imbalance. The main configuration is:

- model: `RandomForestClassifier`
- number of trees: `300`
- `class_weight`: `balanced_subsample`
- `random_state`: `42`
- `n_jobs`: `-1`

In addition to the classification report, the model also provides feature importance scores. Therefore, it offers not only predictive performance but also a degree of interpretability regarding which gas-derived variables contribute most to the decision process.

## RUL Models

The RUL task is formulated as a regression problem in which a continuous target is predicted from multivariate time series. Unlike the FDD branch, the sequential structure of the observations is preserved here.

### Shared Preprocessing Logic

In both deep learning models:

- input data are loaded as `float32`,
- z-normalization is applied using training-set statistics,
- `EarlyStopping` is used to reduce overfitting risk,
- `MAE` and `RMSE` are computed for evaluation.

### `train_lstm.py`

This module builds an LSTM-based regression model using all `200` time steps.

Architecture summary:

- `Input(shape=(200, 4))`
- `LSTM(64)`
- `Dropout(0.2)`
- `Dense(32, activation="relu")`
- `Dense(1)`

Training configuration:

- loss function: `mse`
- metric: `mae`
- epochs: `50`
- batch size: `32`
- early stopping patience: `8`

In this setup, the RUL labels are normalized by dividing by the maximum value observed in the training set.

### `train_gru.py`

This module builds a more compact GRU-based sequential model that focuses on the last `50` time steps. This design tests the assumption that recent temporal behavior may be more informative for lifetime prediction.

Architecture summary:

- `Input(shape=(50, 4))`
- `GRU(64)`
- `Dropout(0.2)`
- `Dense(32, activation="relu")`
- `Dense(1)`

Training configuration:

- loss function: `mae`
- metric: `mae`
- epochs: `50`
- batch size: `32`
- early stopping patience: `8`
- RUL clipping threshold: `500.0`

In this model, the target variable is first clipped to the `0-500` range and then normalized. This limits the influence of very large target values on training dynamics.

## Experimental Findings

This section summarizes representative experimental outputs obtained within the project. The reported results provide a comparative view of the models for both the classification and regression tasks.

### FDD Results

For the FDD task, the GRNN and Random Forest models were evaluated on the test set. The results indicate that the Random Forest approach outperforms the GRNN model across the major performance metrics.

#### Overall comparison

| Model | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| GRNN | 0.92 | 0.80 | 0.92 |
| Random Forest | 0.96 | 0.91 | 0.96 |

#### GRNN FDD classification report

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.96 | 0.97 | 0.96 | 731 |
| 2 | 0.81 | 0.79 | 0.80 | 38 |
| 3 | 0.76 | 0.65 | 0.70 | 49 |
| 4 | 0.72 | 0.73 | 0.73 | 82 |
| Macro Avg | 0.81 | 0.79 | 0.80 | 900 |
| Weighted Avg | 0.92 | 0.92 | 0.92 | 900 |

Confusion matrix for the GRNN model:

```text
[[707   0   9  15]
 [  2  30   0   6]
 [ 15   0  32   2]
 [ 14   7   1  60]]
```

These results show that the GRNN model performs strongly on the dominant `Class 1`, while its performance decreases for minority classes such as `Class 3` and `Class 4`.

#### Random Forest FDD classification report

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.98 | 0.99 | 0.98 | 731 |
| 2 | 0.90 | 0.97 | 0.94 | 38 |
| 3 | 0.93 | 0.88 | 0.91 | 49 |
| 4 | 0.85 | 0.82 | 0.83 | 82 |
| Macro Avg | 0.92 | 0.91 | 0.91 | 900 |
| Weighted Avg | 0.96 | 0.96 | 0.96 | 900 |

Confusion matrix for the Random Forest model:

```text
[[721   0   2   8]
 [  0  37   0   1]
 [  3   0  43   3]
 [ 10   4   1  67]]
```

The Random Forest model achieves more balanced performance across minority classes. In particular, the high precision and recall values for `Class 2` and `Class 3` suggest that the model captures discriminative patterns more effectively despite class imbalance.

#### Random Forest feature importance

The top 15 feature importance scores produced by the Random Forest model are given below:

| Rank | Feature | Importance |
| --- | --- | ---: |
| 1 | `R1_H2_CO` | 0.124147 |
| 2 | `CO_last` | 0.121790 |
| 3 | `CO_max` | 0.099025 |
| 4 | `R3_H2_C2H4` | 0.096098 |
| 5 | `CO_mean` | 0.082340 |
| 6 | `C2H4_max` | 0.067118 |
| 7 | `C2H4_last` | 0.065514 |
| 8 | `CO_min` | 0.048301 |
| 9 | `C2H4_mean` | 0.043669 |
| 10 | `C2H2_last` | 0.031746 |
| 11 | `C2H4_min` | 0.029722 |
| 12 | `R4_CO_C2H2` | 0.025785 |
| 13 | `CO_std` | 0.020042 |
| 14 | `C2H2_max` | 0.019722 |
| 15 | `C2H4_std` | 0.014879 |

This ranking suggests that `CO`-based features and DGA ratio variables play a particularly important role in fault classification. The high ranking of ratio-based variables also supports the usefulness of the adopted feature engineering strategy.

### RUL Results

For the RUL task, two deep learning models were evaluated: an LSTM-based model and a GRU-based model. Under the current setup, the GRU model yields substantially lower error values than the LSTM model.

#### Overall comparison

| Model | Validation MAE | Validation RMSE | Test MAE | Test RMSE |
| --- | ---: | ---: | ---: | ---: |
| GRU | 9.981 | 15.052 | 10.438 | 15.789 |
| LSTM | 221.588 | 245.454 | 218.608 | 243.766 |

#### GRU evaluation

Validation and test results for the GRU model are as follows:

```text
VALIDATION
MAE : 9.981
RMSE: 15.052

TEST
MAE : 10.438
RMSE: 15.789
```

The example predictions indicate that the model produces estimates that are quite close to the target values, especially around the `500` level:

```text
Validation examples:
true=500.00   pred=484.15
true=500.00   pred=504.38
true=500.00   pred=502.12

Test examples:
true=500.00   pred=492.91
true=430.00   pred=499.29
true=468.00   pred=503.13
```

This strong performance suggests that focusing on the last `50` time steps and clipping the target at `500` substantially reduces the reported error metrics. However, this result should be interpreted carefully, since the model is not learning the full original RUL range directly but rather a clipped target space.

#### LSTM evaluation

Validation and test results for the LSTM model are as follows:

```text
VALIDATION
MAE : 221.588
RMSE: 245.454

TEST
MAE : 218.608
RMSE: 243.766
```

The first example predictions show clear deviations from the ground-truth values:

```text
Validation examples:
true=810.00    pred=905.85
true=1093.00   pred=777.49
true=572.00    pred=789.86

Test examples:
true=693.00    pred=876.49
true=430.00    pred=751.91
true=502.00    pred=840.33
```

These results suggest that the current LSTM setup does not model the data distribution effectively enough and tends to produce predictions concentrated around a limited range. In particular, the model appears less capable of separating low and high RUL levels.

#### Interpretation of the findings

Taken together, the present experiments support the following conclusions:

- `Random Forest` is the best-performing approach for the FDD task.
- The GRNN model provides an acceptable baseline but remains weaker on minority classes.
- For the RUL task, the `GRU` model clearly outperforms the `LSTM` model under the current experimental design.
- However, direct one-to-one comparison between GRU and LSTM should be made cautiously, since the two models use different target scaling and temporal window strategies.

## Dependencies

The project defines the following core dependencies in `pyproject.toml`:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow`

Required Python version:

```text
Python >= 3.12
```

## Installation

An example setup using a virtual environment is shown below:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Alternatively, the dependencies may be installed directly:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

## Execution Steps

The recommended experimental order is as follows:

### 1. Data loading and validation

```bash
python3 src/data_preprocessing/load_data.py
python3 src/data_preprocessing/validate_data.py
```

### 2. Building training, validation, and test sequences

```bash
python3 src/data_preprocessing/build_sequences.py
```

### 3. Inspecting class distribution

```bash
python3 src/models/check_class_distrubiton.py
```

### 4. Training FDD models

```bash
python3 src/models/fdd_prediction_models/train_grnn_fdd.py
python3 src/models/fdd_prediction_models/train_random_forrest_fdd.py
```

### 5. Training RUL models

```bash
python3 src/models/rul_predictiction_model/train_lstm.py
python3 src/models/rul_predictiction_model/train_gru.py
```

## Evaluation Strategy

Different evaluation metrics are used depending on the task type:

### For FDD

- confusion matrix,
- class-wise `precision`, `recall`, and `f1-score`,
- overall classification report.

### For RUL

- `MAE` (Mean Absolute Error),
- `RMSE` (Root Mean Squared Error),
- numerical comparison of example predictions.

This combination enables a more comprehensive assessment of both the classification and regression tasks.

## Strengths

The main strengths of this repository may be summarized as follows:

- joint treatment of both FDD and RUL tasks on the same dataset,
- comparable presentation of classical and deep learning-based approaches,
- modular data preparation pipeline that supports reproducibility,
- domain-informed feature engineering based on DGA diagnostic ratios,
- explicit consideration of the class imbalance problem.

## Current Limitations

The current implementation still presents several areas for improvement:

- some file paths are written as absolute paths; converting them to relative paths would improve portability,
- there are naming inconsistencies in certain directories and modules (`predictiction`, `forrest`, `distrubiton`),
- full experimental reporting and version-controlled results would further strengthen reproducibility,
- FDD performance could be improved through cross-validation, hyperparameter tuning, and additional imbalance-aware techniques,
- for RUL, attention-based models, TCNs, or Transformer-based architectures may be explored in future work.

## Future Work

The project can be extended in several directions for stronger academic or industrial relevance:

- data augmentation and synthetic sample generation,
- multi-task learning to address FDD and RUL jointly within a shared model,
- explainable AI methods for interpreting model decisions,
- integration into online monitoring infrastructures,
- evaluation of generalization under different transformer operating conditions.

## Conclusion

This repository provides an integrated experimental environment for data-driven fault diagnosis and lifetime prediction in power transformers. By leveraging DGA measurements, it combines data preparation, feature extraction, classification, and regression within a single study framework and offers a solid starting point for predictive maintenance research.
