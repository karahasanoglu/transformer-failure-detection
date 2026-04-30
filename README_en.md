# Transformer Failure Detection

This repository is a multi-branch research portfolio on transformer fault detection, condition monitoring, predictive maintenance, pseudo-anomaly risk modeling, fault diagnosis, and remaining useful life prediction.

The `main` branch is intended as the academic entry point of the repository. Instead of hosting one standalone experiment, it summarizes and compares the projects developed in the other branches. Each branch addresses a different data source, task definition, and modeling strategy.

## Branch Map

| Branch | Research Focus | Data Type | Main Task | Main Models |
|---|---|---|---|---|
| [`DGA_Analysis`](https://github.com/karahasanoglu/transformer-failure-detection/tree/DGA_Analysis) | Transformer fault type classification with DGA | Gas concentration table | 7-class classification | Decision Tree, Random Forest, SVM, Dense NN |
| [`EFRI`](https://github.com/karahasanoglu/transformer-failure-detection/tree/EFRI) | Electrical fault classification from current and voltage | Three-phase current-voltage data | Multi-class classification | Decision Tree, Logistic Regression, Random Forest, SVM, NN |
| [`Transformer_Monitoring`](https://github.com/karahasanoglu/transformer-failure-detection/tree/Transformer_Monitoring) | IoT-based transformer alarm prediction | Timestamped operational data | Binary classification | KNN, GaussianNB, Logistic Regression, Random Forest, XGBoost |
| [`predictive_maintenance`](https://github.com/karahasanoglu/transformer-failure-detection/tree/predictive_maintenance) | Pseudo-risk labeling and SVM classification for distribution transformers | Annual transformer/network data | Pseudo-anomaly classification | KMeans, Isolation Forest, RBF SVM |
| [`2019-2020-Predictive`](https://github.com/karahasanoglu/transformer-failure-detection/tree/2019-2020-Predictive) | Learning from 2019 and testing pseudo-risk prediction on 2020 | Annual transformer/network data | Forward-year pseudo-risk classification | Random Forest, XGBoost, Autoencoder |
| [`rul_and_failureType_prediction`](https://github.com/karahasanoglu/transformer-failure-detection/tree/rul_and_failureType_prediction) | Joint FDD and RUL prediction from DGA time series | Multivariate DGA time series | Classification + regression | GRNN-like model, Random Forest, GRU, LSTM |

## Repository Scope

The projects are not simple variations of one dataset. They study transformer reliability from different measurement sources and decision levels:

- internal fault diagnosis using DGA gas measurements,
- electrical fault classification using three-phase current and voltage signals,
- alarm prediction from IoT-based transformer monitoring data,
- pseudo-risk modeling from annual transformer and network characteristics,
- fault diagnosis and RUL prediction from DGA time series.

Taken together, the repository shows that predictive maintenance is not only a model selection problem. Data reliability, target definition, class imbalance, temporal generalization, feature engineering, and metric choice are equally important.

## 1. DGA_Analysis

`DGA_Analysis` uses Dissolved Gas Analysis data to classify transformer fault types. Two DGA datasets are merged into a common label space, gas-ratio features are engineered, and several models are compared.

Datasets:

- `data/raw/DGA-dataset.csv`
- `data/raw/DGA_dataset2.csv`
- `data/raw/dga_merged_dataset.csv`

Target classes:

| Class | Meaning |
|---|---|
| `PD` | Partial Discharge |
| `D1` | Low Energy Discharge |
| `D2` | High Energy Discharge |
| `T1` | Thermal Fault < 300C |
| `T2` | Thermal Fault 300C - 700C |
| `T3` | Thermal Fault > 700C |
| `NF` | No Fault / Normal |

Main variables:

- `H2`
- `CH4`
- `C2H6`
- `C2H4`
- `C2H2`

Engineered ratios:

- `R1 = CH4 / H2`
- `R2 = C2H2 / C2H4`
- `R4 = C2H6 / CH4`
- `R5 = C2H4 / C2H6`

Reported results:

| Model | Result |
|---|---:|
| Decision Tree | Test accuracy about `0.8960` |
| Random Forest | Test accuracy about `0.8936` |
| SVM | Test accuracy about `0.7933` |
| Dense Neural Network | Test accuracy about `0.75` |

Academic interpretation: for DGA-based tabular data, classical machine learning remains highly competitive. Tree-based methods are especially strong because they handle nonlinear feature interactions while remaining comparatively interpretable.

## 2. EFRI

`EFRI` classifies electrical fault types from three-phase current and voltage measurements.

Dataset:

- `data/classData.csv`

Input variables:

- `Ia`, `Ib`, `Ic`
- `Va`, `Vb`, `Vc`

Target classes:

- `No Fault`
- `LG Fault`
- `LL Fault`
- `LLG Fault`
- `LLL Fault`
- `GLLL Fault`

Reported results:

| Model | Accuracy |
|---|---:|
| Random Forest | `0.8709` |
| Neural Network | `0.8560` |
| Support Vector Machine | `0.8531` |
| Decision Tree | `0.8442` |
| Logistic Regression | `0.3223` |

Academic interpretation: the poor Logistic Regression result indicates that this task is strongly nonlinear. Random Forest, SVM, and Neural Network models are more suitable for separating the fault classes. The main ambiguity appears between `GLLL Fault` and `LLL Fault`.

## 3. Transformer_Monitoring

`Transformer_Monitoring` predicts the `MOG_A` alarm variable from IoT-based distribution transformer monitoring data collected between June 25, 2019 and April 14, 2020 at 15-minute intervals.

Datasets:

- `data/Overview.csv`
- `data/CurrentVoltage.csv`

The files are merged through `DeviceTimeStamp`.

Main variables:

- voltage and current variables: `VL1`, `VL2`, `VL3`, `IL1`, `IL2`, `IL3`, `VL12`, `VL23`, `VL31`, `INUT`
- monitoring variables: `OTI`, `WTI`, `ATI`, `OLI`, `OTI_A`, `OTI_T`
- target: `MOG_A`

Reported results:

| Model | Accuracy | Interpretation |
|---|---:|---|
| Random Forest | `0.9883` | Strong and balanced |
| XGBoost | `0.9883` | Similar to Random Forest |
| KNN | `0.9538` | Good overall, weaker on alarm recall |
| Logistic Regression | `0.9370` | Interpretable baseline, limited alarm recall |
| GaussianNB | `0.8480` | High alarm recall, more false positives |

Academic interpretation: operational transformer monitoring data contains strong predictive information for alarm classification. Ensemble models provide the best balance, while GaussianNB may be useful when missing alarms is more costly than false positives.

## 4. predictive_maintenance

`predictive_maintenance` studies annual transformer and network records for distribution transformer risk modeling. In the current pipeline, the original `Burned transformers 2019/2020` columns are not used as direct targets. They are removed during preprocessing to reduce target leakage and label reliability issues.

Datasets:

- `data/raw/Dataset_Year_2019.xlsx`
- `data/raw/Dataset_Year_2020.xlsx`

Each file contains `15873` observations.

Engineered variables include:

- `power_per_user`
- `lightning_risk_score`
- `network_density`
- `historical_risk_index`

Pseudo-labeling methods:

- `KMeans(n_clusters=2)`
- `IsolationForest(contamination=0.05)`

Target definition:

```text
target = 1 if KMeans or Isolation Forest marks the observation as risky/anomalous
target = 0 otherwise
```

Pseudo target distribution:

| Dataset | target=0 | target=1 | Risk Rate |
|---|---:|---:|---:|
| 2019 | 12736 | 3137 | 19.76% |
| 2020 | 12778 | 3095 | 19.50% |

SVM results:

| Experiment | Confusion Matrix | Overall Result |
|---|---|---|
| 2019 SVM | `[[2490, 63], [3, 619]]` | Accuracy `0.9792`, macro F1 `0.9682` |
| 2020 SVM | `[[2491, 65], [6, 613]]` | Accuracy `0.9776`, macro F1 `0.9656` |
| 2019-2020 SVM | `[[4995, 95], [9, 1251]]` | Accuracy about `0.98`, macro F1 about `0.97` |

Academic interpretation: these results should not be interpreted as verified field-failure prediction. The SVM learns a pseudo-risk target produced by unsupervised methods. The branch is valuable because it tests whether unsupervised risk definitions are separable by a supervised classifier.

## 5. 2019-2020-Predictive

`2019-2020-Predictive` extends the pseudo-risk pipeline with a more realistic temporal design: 2019 is used for training/validation and 2020 is used as a forward-year test set.

The original `Burned transformers` columns are again removed, and the target is generated through KMeans and Isolation Forest.

Target meaning:

| Label | Meaning |
|---:|---|
| `0` | Normal or low-risk observation |
| `1` | Pseudo-anomalous or high-risk observation |

Models:

- Random Forest
- XGBoost
- Autoencoder anomaly detection

Results:

| Model | Accuracy | target=1 Precision | target=1 Recall | target=1 F1 | Interpretation |
|---|---:|---:|---:|---:|---|
| Random Forest | `0.9911` | `0.98` | `0.97` | `0.98` | Strong and balanced |
| XGBoost | `0.9915` | `0.99` | `0.96` | `0.98` | Fewer false positives, slightly more false negatives |
| Autoencoder | `0.9214` | `0.7276` | `0.9538` | `0.8255` | High risk-class recall, more false positives |

Confusion matrices:

```text
Random Forest:
[[12714, 64],
 [78, 3017]]

XGBoost:
[[12755, 23],
 [112, 2983]]

Autoencoder:
[[11673, 1105],
 [143, 2952]]
```

Academic interpretation: this branch clearly shows the trade-off between precision and recall in maintenance risk screening. Random Forest and XGBoost learn pseudo labels very accurately, while the Autoencoder is more sensitive to risk-like observations but produces more false alarms.

## 6. rul_and_failureType_prediction

`rul_and_failureType_prediction` jointly studies two tasks:

- `FDD`: Fault Detection and Diagnosis
- `RUL`: Remaining Useful Life prediction

The raw data consists of many CSV time-series files:

- `data/raw/data_train/`
- `data/raw/data_test/`
- `data/raw/data_labels/`

Gas variables:

- `H2`
- `CO`
- `C2H4`
- `C2H2`

All samples are converted into fixed-length sequences:

```text
(sequence_length, feature_count) = (200, 4)
```

Processed tensor sizes:

| Split | Shape |
|---|---|
| Train | `(1680, 200, 4)` |
| Validation | `(420, 200, 4)` |
| Test | `(900, 200, 4)` |

FDD features include statistical summaries for each gas variable and ratio features:

- `R1 = H2 / CO`
- `R2 = C2H2 / C2H4`
- `R3 = H2 / C2H4`
- `R4 = CO / C2H2`

FDD results:

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| GRNN-like model | `0.92` | `0.80` | `0.92` |
| Random Forest | `0.96` | `0.91` | `0.96` |

RUL results:

| Model | Validation MAE | Validation RMSE | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|
| GRU | `9.981` | `15.052` | `10.438` | `15.789` |
| LSTM | `221.588` | `245.454` | `218.608` | `243.766` |

Academic interpretation: this is the broadest branch in the repository because it combines feature-based FDD classification with sequence-based RUL regression. Random Forest is the strongest FDD model, and GRU is much stronger than LSTM in the reported RUL setup. The GRU/LSTM comparison should still be interpreted carefully because the models use different windowing and target-scaling strategies.

## Cross-Branch Findings

Several methodological patterns appear across the repository:

1. Tree-based ensemble models are repeatedly strong on structured transformer datasets.
2. DGA ratio features provide useful domain-informed information for fault diagnosis.
3. Linear models are often insufficient for electrical fault classification and alarm prediction.
4. In imbalanced maintenance settings, accuracy alone is not enough; precision, recall, F1, macro F1, confusion matrices, MAE, and RMSE must be interpreted together.
5. Pseudo-labeled branches evaluate learnability of algorithmic risk definitions, not verified field-failure prediction.
6. RUL prediction depends strongly on sequence length, target scaling, and temporal modeling assumptions.

## Academic Limitations

This repository is a research and prototyping environment. Important limitations include:

- Some branches contain absolute file paths, which should be converted to project-relative paths for portability.
- The branches do not yet share a unified experiment tracking pipeline.
- Some results are based on a single train-test split; broader cross-validation or temporal validation would strengthen the conclusions.
- In pseudo-labeled branches, `target=1` does not mean confirmed failure.
- For imbalanced tasks, accuracy should not be used as the only evaluation metric.

## Conclusion

This repository demonstrates multiple data-driven approaches to transformer fault detection and predictive maintenance. Its central message is that no single model is universally best. The appropriate method depends on the data source, target reliability, temporal structure, class imbalance, and the operational cost of false positives and false negatives.

A robust transformer maintenance workflow therefore requires more than a high-accuracy model. It also requires careful preprocessing, domain-informed feature engineering, explicit target definition, appropriate metrics, and honest interpretation of methodological limits.

## Developer

**Emir Furkan Karahasanoglu**  
**Optiway Solutions**  
**2026**
