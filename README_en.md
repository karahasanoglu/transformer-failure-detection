# Transformer Failure Detection

This repository is a multi-track research portfolio focused on transformer fault detection, condition monitoring, predictive maintenance, fault classification, and remaining useful life estimation. The `main` branch is designed not as a standalone experimental branch, but as an entry point that presents a structured overview of the projects developed across the other branches.

The studies collected in this repository share a common objective: improving reliability in power systems through early warning, maintenance prioritization, and data-driven decision support. At the same time, each branch addresses a different problem definition, data structure, and modeling strategy. For that reason, this README provides a clear and academically framed branch-level synthesis of the repository.

## Repository Structure and Access

The main research branches in this repository are listed below:

- [`DGA_Analysis`](https://github.com/karahasanoglu/transformer-failure-detection/tree/DGA_Analysis)
- [`EFRI`](https://github.com/karahasanoglu/transformer-failure-detection/tree/EFRI)
- [`Transformer_Monitoring`](https://github.com/karahasanoglu/transformer-failure-detection/tree/Transformer_Monitoring)
- [`predictive_maintenance`](https://github.com/karahasanoglu/transformer-failure-detection/tree/predictive_maintenance)
- [`rul_and_failureType_prediction`](https://github.com/karahasanoglu/transformer-failure-detection/tree/rul_and_failureType_prediction)

Each branch represents a distinct research problem and an independent experimental workflow. The following sections summarize these branches in a systematic and academically interpretable form.

## Branch-Level Academic Analysis

### 1. `DGA_Analysis`

Branch link:

- https://github.com/karahasanoglu/transformer-failure-detection/tree/DGA_Analysis

#### Objective

This branch focuses on multi-class fault type classification in power transformers using Dissolved Gas Analysis (DGA) data. The main objective is to merge DGA datasets from different sources into a shared label space and perform transformer fault diagnosis using gas concentration variables and ratio-based features.

#### Problem Definition

The task is formulated as a 7-class fault diagnosis problem. The target classes are:

- `PD`: Partial Discharge
- `D1`: Low Energy Discharge
- `D2`: High Energy Discharge
- `T1`: Thermal Fault < 300C
- `T2`: Thermal Fault 300C - 700C
- `T3`: Thermal Fault > 700C
- `NF`: No Fault / Normal

#### Datasets Used

This branch uses two separate DGA datasets:

- `data/raw/DGA-dataset.csv`
- `data/raw/DGA_dataset2.csv`

These datasets are merged during preprocessing and standardized into a common labeling framework. This logic is implemented in `src/data_preprocessing.py`.

#### Input Variables

The primary gas variables are:

- `H2`
- `CH4`
- `C2H6`
- `C2H4`
- `C2H2`

In addition, domain-informed ratio features are generated:

- `R1 = CH4 / H2`
- `R2 = C2H2 / C2H4`
- `R4 = C2H6 / CH4`
- `R5 = C2H4 / C2H6`

Instead of conventional standard deviation-based normalization, the branch applies IQR-based robust scaling.

#### Models Used

This branch includes a comparative evaluation of the following models:

- Decision Tree
- Random Forest
- Support Vector Machine
- Dense Neural Network

Code-level verification shows the main implementations are based on:

- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `SVC`
- `keras.Sequential`

#### Reported Findings

Based on the branch README and the available code outputs:

- Decision Tree achieves approximately `0.896` test accuracy.
- Random Forest achieves approximately `0.894` test accuracy.
- SVM remains around `0.793` test accuracy.
- The Dense Neural Network performs in the `0.75` range.

#### Academic Interpretation

This branch shows that, for DGA-based transformer fault diagnosis, feature engineering combined with classical machine learning remains highly effective. In particular, tree-based models appear more stable than neural networks for structured and interpretable tabular data.

### 2. `EFRI`

Branch link:

- https://github.com/karahasanoglu/transformer-failure-detection/tree/EFRI

#### Objective

This branch aims to classify electrical fault types in power systems using current and voltage measurements. Its core purpose is to build a classification framework capable of distinguishing multiple fault conditions from three-phase electrical signals.

#### Problem Definition

This branch addresses a multi-class electrical fault classification problem. The target classes are:

- `No Fault`
- `LG Fault`
- `LL Fault`
- `LLG Fault`
- `LLL Fault`
- `GLLL Fault`

The labels are derived from combinations of four fault indicator variables:

- `G`
- `C`
- `B`
- `A`

#### Dataset Used

This branch uses a single main dataset:

- `data/classData.csv`

#### Input Variables

Code inspection indicates that the predictive input variables are the following six continuous measurements:

- `Ia`
- `Ib`
- `Ic`
- `Va`
- `Vb`
- `Vc`

The preprocessing pipeline includes:

- scaling with `StandardScaler`,
- label encoding with `LabelEncoder`,
- categorical target conversion for the neural network workflow.

#### Models Used

The following models are evaluated in this branch:

- Decision Tree
- Logistic Regression
- Neural Network
- Random Forest
- Support Vector Machine

The primary model classes identified in the code are:

- `DecisionTreeClassifier`
- `LogisticRegression`
- `RandomForestClassifier`
- `SVC`
- `tensorflow.keras.Sequential`

#### Reported Findings

According to the branch README:

- Random Forest: approximately `0.8709` accuracy
- Neural Network: approximately `0.856` test accuracy
- SVM: approximately `0.8531` accuracy
- Decision Tree: approximately `0.8442` accuracy
- Logistic Regression: approximately `0.3223` accuracy

#### Academic Interpretation

This branch suggests that ensemble methods and kernel-based approaches are more suitable than linear models for this type of electrical fault classification task. The weak performance of Logistic Regression indicates that the class boundaries are likely highly nonlinear.

### 3. `Transformer_Monitoring`

Branch link:

- https://github.com/karahasanoglu/transformer-failure-detection/tree/Transformer_Monitoring

#### Objective

This branch aims to predict alarm-like or abnormal operating conditions from IoT-based distribution transformer monitoring data. The target variable is `MOG_A`, representing a magnetic oil gauge alarm signal.

#### Problem Definition

The task is formulated as a binary classification problem. The goal is to predict the `MOG_A` alarm state from operational transformer measurements.

#### Datasets Used

This branch combines two timestamped data files:

- `data/Overview.csv`
- `data/CurrentVoltage.csv`

At the code level, these files are merged through the `DeviceTimeStamp` field.

#### Input Variables

The main variables from `CurrentVoltage.csv` are:

- `VL1`
- `VL2`
- `VL3`
- `IL1`
- `IL2`
- `IL3`
- `VL12`
- `VL23`
- `VL31`
- `INUT`

The main variables from `Overview.csv` are:

- `OTI`
- `WTI`
- `ATI`
- `OLI`
- `OTI_A`
- `OTI_T`

Target variable:

- `MOG_A`

The preprocessing pipeline uses `MinMaxScaler`.

#### Models Used

This branch includes the following models:

- K-Nearest Neighbors
- Gaussian Naive Bayes
- Logistic Regression
- Random Forest
- XGBoost Classifier

Code-level verification shows the following model classes:

- `KNeighborsClassifier`
- `GaussianNB`
- `LogisticRegression`
- `RandomForestClassifier`
- `xgboost.XGBClassifier`

#### Reported Findings

According to the README:

- Random Forest: approximately `0.9883` accuracy
- XGBoost: approximately `0.9883` accuracy
- KNN: approximately `0.9538` accuracy
- Logistic Regression: approximately `0.9370` accuracy
- GaussianNB: approximately `0.8480` accuracy

#### Academic Interpretation

This branch demonstrates that ensemble methods are highly effective for IoT-based transformer monitoring data. The strong performance of Random Forest and XGBoost suggests that the task contains robust nonlinear patterns embedded in structured operational data.

### 4. `predictive_maintenance`

Branch link:

- https://github.com/karahasanoglu/transformer-failure-detection/tree/predictive_maintenance

#### Objective

This branch is a predictive maintenance prototype designed to estimate annual transformer failure risk. It reconstructs a literature-inspired problem setting while adopting a more transparent supervised learning framework aligned with the available labels.

#### Problem Definition

The core problem is binary classification of transformers as `burned` or `not burned`. In addition, the branch produces a fleet-level risk projection for the year 2021 using the 2019 and 2020 datasets.

Accordingly, this branch combines two analytical layers:

- binary failure risk classification,
- future-year fleet-level risk projection.

#### Datasets Used

This branch relies on two Excel datasets:

- `data/raw/Dataset_Year_2019.xlsx`
- `data/raw/Dataset_Year_2020.xlsx`

At the code level, the target variable is resolved from the file name:

- `Burned transformers 2019`
- `Burned transformers 2020`

#### Input Variables

Code verification shows that the core feature set includes:

- `LOCATION`
- `POWER`
- `SELF_PROTECTION`
- `AVG_DISCHARGE`
- `MAX_DISCHARGE`
- `BURNING_RATE`
- `CRITICALITY`
- `REMOVABLE_CONNECTORS`
- `NUM_USERS`
- `ENERGY_NOT_SUPPLIED`
- `AIR_NETWORK`
- `CIRCUIT_QUEUE`
- `NETWORK_LENGTH`
- `IS_RESIDENTIAL`
- `IS_POLE`

The branch also defines a wider engineered feature pool, including:

- `ENERGY_PER_USER`
- `LIGHTNING_RISK`
- `NETWORK_PER_POWER`
- `DISCHARGE_RANGE`
- `IS_MACRO`
- `LOW_POWER`
- `POWER_LIGHTNING`
- `NETWORK_RISK`

However, both `main.py` and the branch README indicate that the primary comparative workflow emphasizes a simplified supervised SVM setup for closer methodological comparability with the reference article.

#### Models Used

This branch is centered on an SVM-based modeling strategy:

- RBF-kernel SVM
- `class_weight='balanced'`
- validation-based threshold optimization

Code inspection confirms that the main implementation is built on `sklearn.svm.SVC`.

#### Reported Findings

The latest results reported in the README are:

- For 2019: best model `Reduced SVM`, `F1 = 0.3467`, `Recall = 0.5098`, `Accuracy = 0.7164`
- For 2020: best model `Full SVM`, `F1 = 0.2710`, `Recall = 0.5250`, `Accuracy = 0.7135`

For the 2021 projection, the README reports:

- project estimate: `1275`
- article reference: `852`

#### Academic Interpretation

This branch clearly illustrates that accuracy alone is insufficient for predictive maintenance tasks with strong class imbalance. The explicit use of threshold tuning, recall, PR-AUC, and balanced accuracy reflects a sound methodological perspective. For this reason, the branch contributes not only through modeling, but also through evaluation design.

### 5. `rul_and_failureType_prediction`

Branch link:

- https://github.com/karahasanoglu/transformer-failure-detection/tree/rul_and_failureType_prediction

#### Objective

This branch addresses two related predictive maintenance problems for power transformers within a single framework:

- `FDD`: Fault Detection and Diagnosis
- `RUL`: Remaining Useful Life prediction

The objective is to use DGA-based time series data both for fault diagnosis and for estimation of the remaining useful life of the equipment.

#### Problem Definition

This branch represents a multi-task predictive maintenance setting:

- the FDD component is a multi-class classification problem,
- the RUL component is a regression problem with a continuous target.

#### Dataset Structure

The branch stores the raw data as many CSV time-series files:

- `data/raw/data_train/`
- `data/raw/data_test/`
- `data/raw/data_labels/`

The label files contain two targets:

- `FDD label`
- `RUL label`

Processed datasets are stored as `.npy` arrays:

- `data/processed_merged/train_set/`
- `data/processed_merged/val_set/`
- `data/processed_merged/test_set/`

The README reports the following tensor shapes:

- `train: (1680, 200, 4)`
- `val: (420, 200, 4)`
- `test: (900, 200, 4)`

#### Input Variables

Code inspection shows that the fixed-length time series are constructed from four gas variables:

- `H2`
- `CO`
- `C2H4`
- `C2H2`

Each sample is converted into a sequence of `200` time steps with `4` features.

On the FDD side, statistical summaries and ratio-based features are also extracted. According to the branch README, the main ratios are:

- `R1 = H2 / CO`
- `R2 = C2H2 / C2H4`
- `R3 = H2 / C2H4`
- `R4 = CO / C2H2`

#### Models Used

This branch includes two model families.

For FDD:

- GRNN-like classifier
- Random Forest

For RUL:

- GRU
- LSTM

Code-level verification shows the following core model classes:

- `RandomForestClassifier`
- `tensorflow.keras.layers.GRU`
- `tensorflow.keras.layers.LSTM`

#### Reported Findings

The main findings reported in the README are:

For FDD:

- Random Forest is reported as the strongest FDD model within the branch.

For RUL:

- GRU: `Validation MAE = 9.981`, `Validation RMSE = 15.052`, `Test MAE = 10.438`, `Test RMSE = 15.789`
- LSTM: `Validation MAE = 221.588`, `Validation RMSE = 245.454`, `Test MAE = 218.608`, `Test RMSE = 243.766`

#### Academic Interpretation

This branch is especially valuable because it brings together two major transformer maintenance tasks under a shared data framework. The combination of time-series-based RUL modeling and feature-based FDD classification creates a broad experimental space. The reported results suggest that, under the current setup, GRU is substantially more suitable than LSTM for the RUL task.

## Comparative Summary Across Branches

| Branch | Main Objective | Data Type | Primary Task | Main Model Families |
|---|---|---|---|---|
| `DGA_Analysis` | Fault type classification using DGA | Tabular data | Multi-class classification | Decision Tree, Random Forest, SVM, Dense NN |
| `EFRI` | Electrical fault classification | Current-voltage tabular data | Multi-class classification | Decision Tree, Logistic Regression, Random Forest, SVM, NN |
| `Transformer_Monitoring` | Alarm prediction from IoT monitoring data | Timestamped tabular data | Binary classification | KNN, GaussianNB, Logistic Regression, Random Forest, XGBoost |
| `predictive_maintenance` | Annual failure risk and maintenance prioritization | Annual operational/network data | Binary classification + risk projection | RBF SVM |
| `rul_and_failureType_prediction` | Joint FDD and RUL modeling | DGA time series | Multi-class classification + regression | GRNN, Random Forest, GRU, LSTM |

## General Interpretation

When the repository is viewed as a whole, three broad patterns emerge:

1. For transformer-related tabular datasets, tree-based ensemble models repeatedly deliver the strongest performance.
2. In imbalanced predictive maintenance settings, accuracy alone is not sufficient; recall, F1, PR-AUC, and balanced accuracy become much more meaningful.
3. For DGA-based analysis, both ratio-based feature engineering and time-series modeling provide value, but the most appropriate model family depends on the specific task.

Taken together, the repository should be understood not as a single-model solution, but as an experimental research portfolio showing how different modeling strategies behave under different transformer maintenance and fault analysis scenarios.

## References

This section lists the article titles found in the current local folder structure:

1. `Distribution Transformer Failure Prediction for Predictive Maintenance Using Hybrid One-Class Deep SVDD Classification and Lightning Strike Failures Data`
2. `A cognitive system for fault prognosis in power transformers`
3. `AI-Enabled Predictive Maintenance for Distribution Transformers`
4. `On the use of Machine Learning for predictive maintenance of power transformers`
5. `Predictive Maintenance for Distribution System Operators in Increasing Transformers' Reliability`
6. `Dataset of distribution transformers for predictive maintenance`

## Developer

**Emir Furkan Karahasanoglu**  
**Optiway Solutions**  
**2026**
