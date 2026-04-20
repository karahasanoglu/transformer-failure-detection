# CAI DGA Project

This project is a machine learning study developed to classify transformer fault types using Dissolved Gas Analysis (DGA) data. Two different DGA datasets were merged into a shared label space, ratio-based features were generated, the data was cleaned, and multiple models were compared experimentally.

The project initially started with a binary classification setup, but it was later expanded into a 7-class multi-class classification problem.

## Project Goal

The goal of this project is to:

- merge DGA data coming from different sources into a common format,
- generate meaningful ratio-based features from gas concentrations,
- standardize fault classes,
- compare multiple machine learning models,
- identify which model is more suitable for this problem,
- visualize results using confusion matrices and classification reports.

## Problem Definition

DGA is used to estimate internal transformer faults by analyzing the amount of gases dissolved in transformer oil. In this project, the target variable is a 7-class fault structure:

- `PD`: Partial Discharge
- `D1`: Low Energy Discharge
- `D2`: High Energy Discharge
- `T1`: Thermal Fault < 300C
- `T2`: Thermal Fault 300C - 700C
- `T3`: Thermal Fault > 700C
- `NF`: No Fault / Normal

On the code side, these classes were numerically encoded with `LabelEncoder` as follows:

- `D1 -> 0`
- `D2 -> 1`
- `NF -> 2`
- `PD -> 3`
- `T1 -> 4`
- `T2 -> 5`
- `T3 -> 6`

## Project Structure

```text
CAI_DGA_Project/
├── data/
│   ├── raw/
│   │   ├── DGA-dataset.csv
│   │   ├── DGA_dataset2.csv
│   │   └── dga_merged_dataset.csv
│   └── processed/
│       └── dga_merged_processed.csv
├── results/
│   └── *_confusion_matrix.png
├── src/
│   ├── data_preprocessing.py
│   ├── DecisionTree.py
│   ├── RandomForrestModel.py
│   ├── SVM.py
│   ├── neural_network.py
│   └── visulization.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Technologies Used

- Python 3.12
- pandas
- numpy
- scikit-learn
- keras
- tensorflow
- matplotlib
- openpyxl
- `uv`

## Data Sources

Two different raw datasets are used in this project:

1. `data/raw/DGA-dataset.csv`
2. `data/raw/DGA_dataset2.csv`

Labels in the first dataset:

- `Partial discharge`
- `Spark discharge`
- `Arc discharge`
- `Low-temperature overheating`
- `Low/Middle-temperature overheating`
- `Middle-temperature overheating`
- `High-temperature overheating`

Labels in the second dataset:

- `PD`
- `D1`
- `D2`
- `T1`
- `T2`
- `T3`
- `NF`

## Label Mapping Logic

The original labels in the first dataset were mapped to the standardized labels used in the second dataset:

| Dataset 1 label | Unified label |
|---|---|
| `Partial discharge` | `PD` |
| `Spark discharge` | `D1` |
| `Arc discharge` | `D2` |
| `Low-temperature overheating` | `T1` |
| `Low/Middle-temperature overheating` | `T2` |
| `Middle-temperature overheating` | `T2` |
| `High-temperature overheating` | `T3` |

With this mapping, the two separate data sources were turned into one shared multi-class problem.

## Data Preprocessing Pipeline

The preprocessing pipeline is defined in [src/data_preprocessing.py](/home/emirfurkan/Desktop/CAI_DGA_Project/src/data_preprocessing.py:1).

The steps are:

1. Read `DGA-dataset.csv` and `DGA_dataset2.csv`.
2. Map the labels in the first dataset to the shared class names.
3. Clean the `Fail` column in the second dataset and rename it to `Type`.
4. Merge the two datasets.
5. Filter valid shared classes.
6. Generate the `Fault_Type` column using `LabelEncoder`.
7. Create gas-ratio-based features.
8. Remove `inf`, `-inf`, and `NaN` values.
9. Apply IQR-based scaling.
10. Save both the merged raw dataset and the processed dataset.

### Generated Ratio Features

- `R1 = CH4 / H2`
- `R2 = C2H2 / C2H4`
- `R4 = C2H6 / CH4`
- `R5 = C2H4 / C2H6`

### IQR Scaling

The scaling formula used in the project:

```text
(value - median) / (Q3 - Q1)
```

This approach is less sensitive to outliers compared to standard deviation based scaling.

## Dataset Summary

When `data_preprocessing.py` is executed, the following summary is produced:

- `Dataset 1 shape`: `201 x 8`
- `Dataset 2 shape`: `4150 x 8`
- `Merged raw shape`: `4351 x 8`
- `Processed shape`: `4226 x 13`

Class distribution after cleaning:

| Class | Count |
|---|---:|
| `D1` | 629 |
| `D2` | 808 |
| `NF` | 722 |
| `PD` | 335 |
| `T1` | 462 |
| `T2` | 361 |
| `T3` | 909 |

Generated files:

- Merged raw data: [data/raw/dga_merged_dataset.csv](/home/emirfurkan/Desktop/CAI_DGA_Project/data/raw/dga_merged_dataset.csv:1)
- Processed data: [data/processed/dga_merged_processed.csv](/home/emirfurkan/Desktop/CAI_DGA_Project/data/processed/dga_merged_processed.csv:1)

## Modeling Approach

Four different models were tested in the project:

1. Decision Tree
2. Random Forest
3. Support Vector Machine
4. Dense Neural Network (`neural_network.py`, using a fully connected architecture for tabular data)

Common workflow:

1. Read the dataset.
2. Split into `X` and `y`.
3. Apply `train_test_split(..., stratify=y)`.
4. Train the model.
5. Compute accuracy and classification report.
6. Generate a confusion matrix visualization.

## Detailed Model Analysis

### 1. Decision Tree

File: [src/DecisionTree.py](/home/emirfurkan/Desktop/CAI_DGA_Project/src/DecisionTree.py:1)

Techniques used:

- `DecisionTreeClassifier`
- `GridSearchCV`
- hyperparameter search over:
  - `max_depth`
  - `min_samples_split`
  - `criterion`
  - `splitter`

Best parameters:

```text
{'criterion': 'gini', 'max_depth': 15, 'min_samples_split': 2, 'splitter': 'best'}
```

Results:

- Test accuracy: `0.8959810874704491`

Class-level strengths:

- Very strong on `NF`, `D2`, and `T3`
- Relatively weaker on `T2` compared to the other classes

Comment:

Even as a single-tree model, Decision Tree produced very strong results. Once a suitable depth was found, the model showed strong discriminative power on this dataset.

### 2. Random Forest

File: [src/RandomForrestModel.py](/home/emirfurkan/Desktop/CAI_DGA_Project/src/RandomForrestModel.py:1)

Techniques used:

- `RandomForestClassifier`
- `n_estimators=200`
- `max_depth=10`
- `class_weight="balanced"`
- 5-fold cross validation

Results:

- CV scores: `0.8617, 0.9302, 0.9077, 0.8533, 0.7160`
- Mean CV accuracy: `0.8537605438751102`
- Test accuracy: `0.8936170212765957`

Comment:

Random Forest was one of the most balanced and reliable models in the project. It performed especially well on `D1`, `D2`, `NF`, `PD`, and `T3`. The lower performance in the last CV fold suggests some fold-level variability in difficulty or class distribution.

### 3. SVM

File: [src/SVM.py](/home/emirfurkan/Desktop/CAI_DGA_Project/src/SVM.py:1)

Techniques used:

- `SVC(kernel="rbf", C=100, gamma="auto", class_weight="balanced")`
- `RobustScaler`
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`

Important note:

Unlike the other models, the SVM model was trained on `data/raw/dga_merged_dataset.csv` and scaled using `RobustScaler` before training.

Results:

- CV scores: `0.8003, 0.7945, 0.7931, 0.7917, 0.8017`
- Mean CV accuracy: `0.7962643678160919`
- Test accuracy: `0.7933409873708381`

Comment:

SVM performed poorly at first, but improved meaningfully after tuning the kernel and scaling strategy. Even so, it is still weaker than the tree-based models in this project.

### 4. Neural Network (`neural_network.py`)

File: [src/neural_network.py](/home/emirfurkan/Desktop/CAI_DGA_Project/src/neural_network.py:1)

Note:

A multi-layer dense neural network is used for tabular data.

Techniques used:

- `Dense(128) -> Dense(64) -> Output`
- `Dropout`
- `EarlyStopping`
- `Adam`
- `class_weight`

Results:

- Test accuracy: `0.7529550827423168`

Observations:

- Accuracy generally fluctuates within the `0.70 - 0.80` range.
- Results are more unstable compared to classical ML models.

Comment:

The neural network gives acceptable results, but it is not the best model in this project. For tabular DGA data, tree-based methods are more stable and achieve higher accuracy.

## Updated Performance Comparison

| Model | CV Mean Accuracy | Test Accuracy |
|---|---:|---:|
| Decision Tree | GridSearch used | 0.8960 |
| Random Forest | 0.8538 | 0.8936 |
| SVM | 0.7963 | 0.7933 |
| Neural Network | Not reported | 0.7470 |

## Class-Level Observations

Key observations from the outputs:

- `NF` is one of the easiest classes for the strongest models.
- `T2` is relatively difficult for most models.
- `D1` and `T1` are more fragile classes for the neural network.
- `T3` is learned well by SVM, Random Forest, and Decision Tree.
- Neural network results vary more between runs.

## Confusion Matrix and Visualization

The shared confusion matrix utility is located in [src/visulization.py](/home/emirfurkan/Desktop/CAI_DGA_Project/src/visulization.py:1).

The `display_cm(...)` function:

- generates the confusion matrix,
- displays it,
- saves it into the `results/` directory as a `.png` file.

Example generated files:

- `results/decisiontree_confusion_matrix.png`
- `results/randomforest_confusion_matrix.png`
- `results/svm_confusion_matrix.png`
- `results/neural_network_confusion_matrix.png`

## Run Commands

Preprocessing:

```bash
./.venv/bin/python src/data_preprocessing.py
```

Decision Tree:

```bash
./.venv/bin/python src/DecisionTree.py
```

Random Forest:

```bash
./.venv/bin/python src/RandomForrestModel.py
```

SVM:

```bash
./.venv/bin/python src/SVM.py
```

Neural network:

```bash
./.venv/bin/python src/neural_network.py
```

## Overall Conclusion

This project has evolved into a comparative machine learning study that successfully merges two different DGA data sources into a shared 7-class fault classification problem. On the preprocessing side, label standardization, ratio-based feature engineering, cleaning, and IQR scaling were applied. The experiments show that `Decision Tree` and `Random Forest` are the strongest models. `SVM` performs moderately well after tuning, while the neural network gives acceptable but more variable results.

## Strengths

- Two different datasets are merged into a shared label space.
- Ratio-based features are well aligned with the domain.
- The multi-class problem is clearly defined.
- Four different models are compared.
- Confusion matrix outputs are saved automatically.
- Results are reported both with overall accuracy and class-based metrics.

## Limitations

- `SVM.py` uses the merged raw dataset instead of the processed dataset, which makes the comparison only partially consistent.
- Neural network results are not deterministic and vary between runs.
- There are still some naming inconsistencies in the project:
  - `RandomForrestModel.py`
  - `visulization.py`
- A single unified experiment pipeline for all models is not yet implemented.

## Future Improvements

- Move all models to a unified data pipeline
- Save and reuse a common train/test split
- Fix random seeds to stabilize neural network results
- Add normalized confusion matrices
- Add feature importance and explainability analysis such as SHAP
