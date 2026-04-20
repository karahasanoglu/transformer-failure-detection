# Predictive Maintenance Project for Distribution Transformers

This repository is a machine learning prototype developed to estimate fault risk in distribution transformers. The study aims to reproduce a scenario close to the problem formulation presented in the following paper:

`Vita, V.; Fotis, G.; Chobanov, V.; Pavlatos, C.; Mladenov, V. Predictive Maintenance for Distribution System Operators in Increasing Transformers’ Reliability. Electronics 2023, 12, 1356.`

This project preserves the general problem setting of the paper, but it is not a one-to-one copy:

- In this repository, the `burned transformers` labels are directly available and used.
- Therefore, the `k-means -> SVM` hybrid labeling pipeline from the paper was removed from the main workflow.
- The main modeling pipeline is built with `supervised learning`.

This makes the evaluation more transparent and easier to interpret, since the true target labels are already available.

## Project Goal

The goal is to use the 2019 and 2020 transformer datasets to:

- distinguish faulty (`1`) and non-faulty (`0`) transformers,
- improve the detection performance for the faulty class,
- produce a numerical projection for 2021,
- compare the obtained results with the values reported in the paper.

This project should be considered a `proof-of-concept / academic prototype`, not a production-ready system.

## Dataset

Raw files are stored under `data/raw/`:

- `Dataset_Year_2019.xlsx`
- `Dataset_Year_2020.xlsx`

Each file contains the true target column for the corresponding year:

- `Burned transformers 2019` for 2019
- `Burned transformers 2020` for 2020

The features used in the project were kept close to the variable structure described in the paper. The main columns are:

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

Notes:

- The repository also contains additional feature engineering functions.
- However, the main `main.py` workflow uses a simpler feature set to stay more comparable to the supervised SVM experiment discussed in the paper.

## Current Modeling Pipeline

The main workflow runs through `main.py` and follows these steps:

1. Load either the 2019 or 2020 dataset.
2. Automatically resolve the target column from the file name.
3. Scale the features using `StandardScaler`.
4. Split the dataset into `train=14873` and `test=1000` using stratified sampling.
5. Create an additional validation subset from the training data.
6. Train an `RBF SVM` model with `class_weight='balanced'`.
7. Optimize the decision threshold on the validation set.
8. Report the following metrics on the test set:
   - Accuracy
   - Precision
   - Recall
   - F1
   - Specificity
   - Balanced Accuracy
   - ROC-AUC
   - PR-AUC

Then, for each year, two SVM variants are compared:

- `Full SVM`: using all selected features
- `Reduced SVM`: after removing the following features

Removed features:

- `LOCATION`
- `POWER`
- `REMOVABLE_CONNECTORS`
- `ENERGY_NOT_SUPPLIED`
- `AIR_NETWORK`
- `CIRCUIT_QUEUE`

For each year, the model with the higher `F1` score is selected as the preferred one.

## How the 2021 Projection Is Produced

There is no real 2021 feature file in the repository. Therefore, the 2021 result is not a direct transformer-by-transformer prediction, but a `risk projection`.

Current projection logic:

1. Combine the 2019 and 2020 datasets.
2. Train a separate SVM on the merged data.
3. Optimize the threshold for this combined model using a validation subset.
4. Apply the model to the full 2019 and 2020 fleets.
5. Compute the average predicted risk rate across both years.
6. Project that average risk rate onto `15,873` transformers to estimate a 2021 figure.

This number:

- is not an exact fault count,
- represents the number of transformers considered high-risk by the model,
- should therefore be interpreted with caution.

At the time this README was written, the latest comparison was:

- project projection: `1275`
- paper reference: `852`

The gap may be caused by both model differences and the interpretation of the projection process.

## Most Recent Results

The most recent recorded run produced the following highlights:

### 2019

- Best model: `Reduced SVM`
- `F1 = 0.3467`
- `Recall = 0.5098`
- `Balanced Accuracy = 0.7164`

### 2020

- Best model: `Full SVM`
- `F1 = 0.2710`
- `Recall = 0.5250`
- `Balanced Accuracy = 0.7135`

These results show that:

- The model is now able to detect the faulty class in a meaningful way.
- However, `precision` is still low, which means the false alarm rate remains high.
- For that reason, this project is better interpreted as an `early warning / risk prioritization` tool rather than a direct operational decision engine.

## Relation to the Paper

This project is based on the same problem setting and dataset logic as the paper, but it re-implements the task more critically and transparently.

The paper’s overall line of work includes:

- data preprocessing
- normalization
- feature selection
- `k-means`
- `SVM`
- accuracy-oriented evaluation

This repository differs in several ways:

- Since real labels are available, `k-means` was removed from the main pipeline.
- Accuracy is not treated as the only relevant metric.
- Validation-based threshold tuning was added.
- More informative metrics such as `PR-AUC`, `balanced accuracy`, and `specificity` are reported.

## Missing or Unclear Points in the Paper

Although the paper is a useful reference, it contains some methodological ambiguities:

1. The exact scaling technique is not explicitly specified.
- The paper only states that normalization was performed.
- It does not clearly say whether this was `StandardScaler`, `MinMaxScaler`, or another method.

2. Accuracy is overly central in the discussion.
- For an imbalanced dataset, this is risky.
- It is not sufficient to understand real fault-detection performance on the positive class.

3. The transition from `k-means` to `SVM` is not fully transparent.
- It is not clearly explained how cluster labels were mapped to ground truth.
- This can introduce evaluation ambiguity or leakage risk.

4. Fault-class performance is not reported as transparently as it could be.
- Precision, recall, and F1 are not emphasized enough.
- High accuracy may therefore overstate anomaly-detection quality.

5. The 2021 numerical projection is a strong claim despite major data limitations.
- Only two years of data are available.
- Important variables such as temperature, oil level, overload history, and maintenance history are missing.

## Main Limitations of This Repository

This repository inherits similar limitations:

- Only two years of data are available.
- Class imbalance is high.
- No operational maintenance records are available.
- No oil analysis, temperature history, load history, or environmental time-series data are included.

Because of that, the model is:

- academically meaningful,
- experimentally functional,
- but not reliable enough to be treated as a standalone maintenance planning system in real life.

## Directory Structure

```text
transformer_predictive_maintenance/
├── data/
│   └── raw/
│       ├── Dataset_Year_2019.xlsx
│       └── Dataset_Year_2020.xlsx
├── results/
│   ├── 2019_svm_full.png
│   ├── 2019_svm_reduced.png
│   ├── 2020_svm_full.png
│   ├── 2020_svm_reduced.png
│   └── ... other result files
├── src/
│   ├── preprocessing.py
│   ├── svm_model.py
│   ├── metrics_utils.py
│   ├── visualization.py
│   ├── data_analysis.py
├── main.py
└── readme_en.md
```

## File Descriptions

- `main.py`  
  Main entry point. Runs the 2019 and 2020 experiments, selects the best SVM variant, and produces the 2021 projection.

- `src/preprocessing.py`  
  Data loading, column mapping, feature generation, scaling, and train/test splitting.

- `src/svm_model.py`  
  RBF-kernel SVM model, training, threshold optimization, and evaluation logic.

- `src/metrics_utils.py`  
  Shared metric computation and threshold-selection utilities.

- `src/visualization.py`  
  Confusion matrix visualization helpers.

- `src/data_analysis.py`  
  Feature importance and class-separation analysis helpers.

## Installation

The following packages are sufficient, either inside a virtual environment or directly in your Python environment:

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn openpyxl
```

## Running the Project

To run the main pipeline:

```bash
python main.py
```

At the end of execution, the following are produced:

- yearly confusion matrix plots
- model metrics
- the 2021 risk projection

both in the terminal output and under the `results/` directory.

## Conclusion

This project rebuilds the problem discussed in the paper under a more transparent evaluation framework. The results show that the model is not entirely ineffective; however, because the dataset has limited explanatory power, the system is better interpreted as a decision-support or risk-ranking tool rather than a real-world standalone predictive maintenance solution.
