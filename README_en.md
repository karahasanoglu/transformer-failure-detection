# Electrical Fault Detection

This project focuses on classifying electrical faults using machine learning models based on current and voltage measurements.

The dataset contains electrical system signals as input features, and the objective is to predict the type of fault in the system.

## Project Purpose

The main purpose of this project is to detect and classify fault conditions in an electrical power system.

The models in this project are trained to recognize the following fault classes:

- `No Fault`
- `LG Fault`
- `LL Fault`
- `LLG Fault`
- `LLL Fault`
- `GLLL Fault`

## Dataset Information

The dataset file is located at:

- [data/classData.csv](/home/emirfurkan/Desktop/Electrical_Fault_Detection/data/classData.csv)

### Input Features

The input variables used in the dataset are:

- `Ia`
- `Ib`
- `Ic`
- `Va`
- `Vb`
- `Vc`

These variables represent three-phase current and voltage values collected from the electrical system.

### Output Labels

The output classes are derived from four fault indicator columns:

- `G`
- `C`
- `B`
- `A`

In the current project code, these indicator combinations are mapped to fault classes as follows:

- `[0, 0, 0, 0]` -> `No Fault`
- `[1, 0, 0, 1]` -> `LG Fault`
- `[0, 1, 1, 0]` -> `LL Fault`
- `[1, 0, 1, 1]` -> `LLG Fault`
- `[0, 1, 1, 1]` -> `LLL Fault`
- `[1, 1, 1, 1]` -> `GLLL Fault`

### Meaning of Fault Types

- `No Fault`: Normal operating condition
- `LG Fault`: Line-to-ground fault
- `LL Fault`: Line-to-line fault
- `LLG Fault`: Double line-to-ground fault
- `LLL Fault`: Three-phase fault
- `GLLL Fault`: Three-phase-to-ground fault

Note: Although the dataset description includes example fault combinations, the class mapping in this README is based on the implementation currently used in [src/preprocessing.py](/home/emirfurkan/Desktop/Electrical_Fault_Detection/src/preprocessing.py).

## Project Structure

```text
Electrical_Fault_Detection/
├── data/
│   └── classData.csv
├── results/
│   ├── DecisionTree_confusion_matrix.png
│   ├── Logistic Regression_confusion_matrix.png
│   ├── Neural Network_confusion_matrix.png
│   ├── RandomForest_confusion_matrix.png
│   └── Support Vector Machine_confusion_matrix.png
├── src/
│   ├── preprocessing.py
│   ├── visulazition.py
│   └── models/
│       ├── DecisionTree.py
│       ├── LogisticRegression.py
│       ├── NeuralNetwork.py
│       ├── RandomForrest.py
│       └── SVM.py
├── pyproject.toml
├── README.md
└── README_en.md
```

## Data Preprocessing

The preprocessing stage includes the following steps:

- Reading the dataset
- Converting fault indicator combinations into class labels
- Splitting the dataset into training and test sets
- Standardizing the input features
- Encoding the class labels numerically

For the neural network model, the labels are additionally converted into categorical format.

## Models Used

The following machine learning models are used in this project:

- Decision Tree
- Logistic Regression
- Neural Network
- Random Forest
- Support Vector Machine

## Model Results

The following results were obtained from the experiments.

### 1. Decision Tree

- Accuracy: `0.8442`
- Strong performance on `LG Fault`, `LL Fault`, `LLG Fault`, and `No Fault`
- Lower performance on `GLLL Fault` and `LLL Fault`

Classification summary:

```text
              precision    recall  f1-score   support

  GLLL Fault       0.48      0.54      0.51       227
    LG Fault       0.97      1.00      0.98       226
    LL Fault       1.00      0.96      0.98       201
   LLG Fault       1.00      0.98      0.99       227
   LLL Fault       0.47      0.42      0.45       219
    No Fault       0.99      1.00      1.00       473

    accuracy                           0.84      1573
   macro avg       0.82      0.82      0.82      1573
weighted avg       0.84      0.84      0.84      1573
```

Confusion matrix:

- [results/DecisionTree_confusion_matrix.png](/home/emirfurkan/Desktop/Electrical_Fault_Detection/results/DecisionTree_confusion_matrix.png)

### 2. Logistic Regression

- Accuracy: `0.3223`
- This model performed poorly on the dataset
- It mostly predicted `No Fault` and failed to separate most fault classes

Classification summary:

```text
              precision    recall  f1-score   support

  GLLL Fault       0.29      0.15      0.20       227
    LG Fault       0.00      0.00      0.00       226
    LL Fault       0.00      0.00      0.00       201
   LLG Fault       0.00      0.00      0.00       227
   LLL Fault       0.00      0.00      0.00       219
    No Fault       0.33      1.00      0.49       473

    accuracy                           0.32      1573
   macro avg       0.10      0.19      0.12      1573
weighted avg       0.14      0.32      0.18      1573
```

Confusion matrix:

- [results/Logistic Regression_confusion_matrix.png](/home/emirfurkan/Desktop/Electrical_Fault_Detection/results/Logistic%20Regression_confusion_matrix.png)

### 3. Neural Network

- Test Accuracy: `0.856`
- Performed better than Decision Tree and Logistic Regression
- Effective in classifying nonlinear fault patterns

Confusion matrix:

- [results/Neural Network_confusion_matrix.png](/home/emirfurkan/Desktop/Electrical_Fault_Detection/results/Neural%20Network_confusion_matrix.png)

### 4. Random Forest

- Accuracy: `0.8709`
- Achieved the best result among the reported models
- Showed very strong performance across most classes
- Still shows moderate confusion between `GLLL Fault` and `LLL Fault`

Classification summary:

```text
              precision    recall  f1-score   support

  GLLL Fault       0.56      0.55      0.55       227
    LG Fault       1.00      1.00      1.00       226
    LL Fault       1.00      1.00      1.00       201
   LLG Fault       0.99      0.99      0.99       227
   LLL Fault       0.55      0.55      0.55       219
    No Fault       1.00      1.00      1.00       473

    accuracy                           0.87      1573
   macro avg       0.85      0.85      0.85      1573
weighted avg       0.87      0.87      0.87      1573
```

Confusion matrix:

- [results/RandomForest_confusion_matrix.png](/home/emirfurkan/Desktop/Electrical_Fault_Detection/results/RandomForest_confusion_matrix.png)

### 5. Support Vector Machine

- Accuracy: `0.8531`
- Strong overall performance
- Performs especially well on `LG Fault`, `LL Fault`, `LLG Fault`, and `No Fault`
- Shows moderate difficulty in separating `GLLL Fault` and `LLL Fault`

Best hyperparameters:

```text
{'C': 100, 'gamma': 'scale', 'kernel': 'rbf'}
```

Classification summary:

```text
              precision    recall  f1-score   support

  GLLL Fault       0.51      0.34      0.41       227
    LG Fault       0.98      1.00      0.99       226
    LL Fault       1.00      1.00      1.00       201
   LLG Fault       1.00      0.97      0.99       227
   LLL Fault       0.49      0.66      0.56       219
    No Fault       1.00      1.00      1.00       473

    accuracy                           0.85      1573
   macro avg       0.83      0.83      0.82      1573
weighted avg       0.85      0.85      0.85      1573
```

Confusion matrix:

- [results/Support Vector Machine_confusion_matrix.png](/home/emirfurkan/Desktop/Electrical_Fault_Detection/results/Support%20Vector%20Machine_confusion_matrix.png)

## Model Comparison

Based on the reported results:

1. `Random Forest` achieved the best overall performance with `87.09%` accuracy.
2. `Neural Network` and `SVM` also produced strong and competitive results.
3. `Decision Tree` performed well, but remained slightly behind `Random Forest` and `SVM`.
4. `Logistic Regression` does not appear to be suitable for this dataset in its current form.

## How to Run

You can run the model files from the project root directory.

Example commands:

```bash
python3 src/models/DecisionTree.py
python3 src/models/RandomForrest.py
python3 src/models/LogisticRegression.py
python3 src/models/SVM.py
python3 src/models/NeuralNetwork.py
```

If you are using `uv`, you can also run:

```bash
uv run python src/models/DecisionTree.py
```

## Dependencies

The main libraries used in this project are:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `tensorflow`

The dependency list is defined in:

- [pyproject.toml](/home/emirfurkan/Desktop/Electrical_Fault_Detection/pyproject.toml)

## Conclusion

This project shows that electrical faults can be successfully classified using machine learning based on current and voltage measurements.

Among the tested models, `Random Forest` achieved the best result. `Neural Network` and `SVM` also produced strong performance. The most noticeable classification challenge in this dataset is the separation of `GLLL Fault` and `LLL Fault`, since these classes appear to be more similar to each other than the others.
