# Distributed Transformer Monitoring

This project aims to classify potential fault or alarm conditions in distribution transformers by using measurement data collected through IoT devices. Although transformers are among the most reliable components of the electrical grid, they are still vulnerable to failures caused by both internal and external factors. In particular, mechanical failures and dielectric failures are among the most critical issues that may lead to severe and catastrophic consequences.

The dataset used in this project was collected through IoT devices between June 25, 2019 and April 14, 2020, with updates recorded every 15 minutes. The main goal is to develop machine learning models that can detect abnormal transformer behavior at an early stage by using operational parameters such as temperature, oil level, current, and voltage.

## Project Purpose

The main purpose of this project is to predict the `MOG_A` variable from transformer monitoring data. `MOG_A` represents the magnetic oil gauge alarm and can be considered an indicator of abnormal or risky operating conditions inside the transformer.

In this context, the project focuses on the following objectives:

- Building a meaningful classification problem from IoT monitoring data
- Comparing the performance of different machine learning models
- Identifying which model provides more reliable alarm/fault detection
- Interpreting model behavior through classification reports and confusion matrices
- Providing a foundation for future predictive maintenance applications

## Dataset Content

The dataset consists of two main files:

- `data/Overview.csv`
- `data/CurrentVoltage.csv`

These two files are merged using the `DeviceTimeStamp` field.

### CurrentVoltage Parameters

- `VL1`: Phase line 1 voltage
- `VL2`: Phase line 2 voltage
- `VL3`: Phase line 3 voltage
- `IL1`: Current line 1
- `IL2`: Current line 2
- `IL3`: Current line 3
- `VL12`: Voltage line 1-2
- `VL23`: Voltage line 2-3
- `VL31`: Voltage line 3-1
- `INUT`: Neutral current

### Overview Parameters

- `OTI`: Oil Temperature Indicator
- `WTI`: Winding Temperature Indicator
- `ATI`: Ambient Temperature Indicator
- `OLI`: Oil Level Indicator
- `OTI_A`: Oil Temperature Indicator Alarm
- `OTI_T`: Oil Temperature Indicator Trip
- `MOG_A`: Magnetic Oil Gauge Indicator Alarm

## Methodology

The preprocessing and modeling pipeline in this project is straightforward:

1. `Overview.csv` and `CurrentVoltage.csv` are loaded.
2. The `DeviceTimeStamp` column is converted to datetime format.
3. The two datasets are merged on the timestamp field.
4. `MOG_A` is selected as the target variable.
5. All columns except `DeviceTimeStamp` and `MOG_A` are used as features.
6. The data is split into training and test sets with an `80% - 20%` ratio.
7. Features are normalized with `MinMaxScaler`.
8. Different classification models are trained and evaluated.
9. A classification report and confusion matrix are generated for each model.

This workflow is implemented in [src/preprocessing.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/preprocessing.py:1) and [src/visualization.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/visualization.py:1).

## Project Structure

```text
Distributed_Transformer_Monitoring/
├── data/
│   ├── CurrentVoltage.csv
│   └── Overview.csv
├── results/
│   ├── GaussianNB_confusion_matrix.png
│   ├── knn_confusion_matrix.png
│   ├── Logistic Regression_confusion_matrix.png
│   ├── RandomForest_confusion_matrix.png
│   └── XGP Classifier_confusion_matrix.png
├── src/
│   ├── preprocessing.py
│   ├── visualization.py
│   └── models/
│       ├── GaussianNB.py
│       ├── knn.py
│       ├── LogisticRegression.py
│       ├── RandomForrest.py
│       └── XGP_Classifier.py
├── pyproject.toml
├── README.md
└── README_en.md
```

## Models Used

Five different classification models are used in this project.

### 1. K-Nearest Neighbors (KNN)

KNN is a distance-based algorithm that determines the class of a new sample by looking at the labels of its nearest neighbors. It can perform well when similar observations cluster naturally in the feature space. Since it depends on distances, feature scaling is especially important for this model.

Code file: [src/models/knn.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/models/knn.py:1)

### 2. Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classifier that assumes feature independence. For continuous numerical data, it models each feature with a Gaussian distribution. It is fast, lightweight, and useful as a strong baseline model.

Code file: [src/models/GaussianNB.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/models/GaussianNB.py:1)

### 3. Logistic Regression

Logistic Regression is a widely used linear model for binary classification problems. It produces probabilistic outputs and is highly interpretable, which makes it valuable as a baseline and comparison model.

Code file: [src/models/LogisticRegression.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/models/LogisticRegression.py:1)

### 4. Random Forest

Random Forest is an ensemble learning method that combines many decision trees. Compared to a single decision tree, it usually provides more stable and generalizable results. It is one of the strongest and most reliable approaches for tabular datasets.

Code file: [src/models/RandomForrest.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/models/RandomForrest.py:1)

### 5. XGBoost Classifier

XGBoost is an advanced ensemble model that implements gradient boosting in a highly efficient and powerful way. It is especially well known for strong performance on structured tabular data and is one of the top-performing models in this project.

Code file: [src/models/XGP_Classifier.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/models/XGP_Classifier.py:1)

## Model Outputs

The following model evaluation results were obtained in this project.

### KNN Results

```text
Acc score : 0.9538236012704617
              precision    recall  f1-score   support

         0.0       0.97      0.98      0.97      3655
         1.0       0.81      0.74      0.77       438

    accuracy                           0.95      4093
   macro avg       0.89      0.86      0.87      4093
weighted avg       0.95      0.95      0.95      4093
```

Evaluation: KNN achieved a strong overall accuracy. However, the recall value of `0.74` for class `1.0` indicates that some actual alarm cases were missed.

### Gaussian Naive Bayes Results

```text
              precision    recall  f1-score   support

         0.0       1.00      0.83      0.91      3655
         1.0       0.41      1.00      0.58       438

    accuracy                           0.85      4093
   macro avg       0.71      0.91      0.75      4093
weighted avg       0.94      0.85      0.87      4093

Accuracy Score: 0.8480332274615197
```

Evaluation: GaussianNB performs very strongly in terms of recall for class `1.0`, meaning it is highly sensitive to alarm conditions. However, its low precision shows that it produces many false positives.

### Logistic Regression Results

```text
              precision    recall  f1-score   support

         0.0       0.94      1.00      0.97      3655
         1.0       0.94      0.44      0.60       438

    accuracy                           0.94      4093
   macro avg       0.94      0.72      0.78      4093
weighted avg       0.94      0.94      0.93      4093

Accuracy Score : 0.9369655509406304
```

Evaluation: Logistic Regression provides high precision, but the low recall for class `1.0` suggests that the model is conservative when predicting alarm cases and misses some critical positives.

### Random Forest Results

```text
              precision    recall  f1-score   support

         0.0       0.99      0.99      0.99      3655
         1.0       0.95      0.94      0.94       438

    accuracy                           0.99      4093
   macro avg       0.97      0.97      0.97      4093
weighted avg       0.99      0.99      0.99      4093

ACC SCORE
0.9882726606401173
```

Evaluation: Random Forest is one of the strongest models in terms of both overall accuracy and class balance. Its high precision and recall for class `1.0` make it especially valuable for real-world alarm detection scenarios.

### XGBoost Results

```text
              precision    recall  f1-score   support

         0.0       0.99      1.00      0.99      3655
         1.0       0.96      0.93      0.94       438

    accuracy                           0.99      4093
   macro avg       0.98      0.96      0.97      4093
weighted avg       0.99      0.99      0.99      4093

Acc score  0.9882726606401173
```

Evaluation: XGBoost also delivers a very strong performance, close to Random Forest. Its ability to preserve class balance while maintaining high accuracy makes it one of the best candidates for this project.

## Overall Model Comparison

When all results are considered together:

- `Random Forest` and `XGBoost` provide the highest accuracy and the most balanced class performance.
- `KNN` performs well overall, but makes more mistakes on the minority class.
- `Logistic Regression` is a useful and interpretable baseline, but is limited in detecting the alarm class.
- `GaussianNB` is notable for its high recall and ability to avoid missing alarm cases, but it produces too many false positives.

If the goal is strong overall performance with balanced predictions, `Random Forest` and `XGBoost` are the most suitable models. If the main priority is to avoid missing risky conditions, a high-recall model such as `GaussianNB` may still be useful in specific operational scenarios.

## Confusion Matrix Outputs

The confusion matrix figures for each model are saved in the `results/` directory:

- `results/knn_confusion_matrix.png`
- `results/GaussianNB_confusion_matrix.png`
- `results/Logistic Regression_confusion_matrix.png`
- `results/RandomForest_confusion_matrix.png`
- `results/XGP Classifier_confusion_matrix.png`

These visualizations are generated by the `plot_cm()` function in [src/visualization.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/visualization.py:1).

## Core Project Files

- [src/preprocessing.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/preprocessing.py:1): Data loading, merging, train-test split, and scaling
- [src/visualization.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/visualization.py:1): Confusion matrix generation and saving
- [src/models/knn.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/models/knn.py:1): KNN model
- [src/models/GaussianNB.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/models/GaussianNB.py:1): Gaussian Naive Bayes model
- [src/models/LogisticRegression.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/models/LogisticRegression.py:1): Logistic Regression model
- [src/models/RandomForrest.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/models/RandomForrest.py:1): Random Forest model
- [src/models/XGP_Classifier.py](/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/src/models/XGP_Classifier.py:1): XGBoost classifier

## Installation

Dependencies are defined in `pyproject.toml`. You can prepare the environment using one of the following methods:

```bash
uv sync
```

or

```bash
pip install -e .
```

## Running the Models

Each model can be executed as a separate Python script:

```bash
python3 src/models/knn.py
python3 src/models/GaussianNB.py
python3 src/models/LogisticRegression.py
python3 src/models/RandomForrest.py
python3 src/models/XGP_Classifier.py
```

## Possible Improvements

This project can be improved further in several directions:

- Add hyperparameter optimization
- Use cross-validation for more robust evaluation
- Handle class imbalance with methods such as `class_weight`, SMOTE, or similar approaches
- Add ROC-AUC, PR-AUC, and F1-focused analysis
- Build a comparison script that summarizes all model outputs in a single table
- Add neural-network-based models such as `MLPClassifier` or `TensorFlow/Keras`
- Perform feature importance analysis to identify the most influential transformer measurements

## Conclusion

This project demonstrates that transformer monitoring data can be used effectively for alarm or fault prediction. In particular, ensemble-based methods such as Random Forest and XGBoost stand out with both high accuracy and balanced class performance. The study provides a useful foundation for predictive maintenance, early warning systems, and improving reliability in energy infrastructure.
