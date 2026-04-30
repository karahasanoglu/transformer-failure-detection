import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix , ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


df_train = pd.read_csv("/home/emirfurkan/Desktop/2019-2020-predictive-model/data/processed/normalized_dataset_2019.csv")
df_test = pd.read_csv("/home/emirfurkan/Desktop/2019-2020-predictive-model/data/processed/normalized_dataset_2020.csv")

output_dir = "/home/emirfurkan/Desktop/2019-2020-predictive-model/results"


X_train = df_train.drop(columns = ["target"])
y_train = df_train["target"]

X_test = df_test.drop(columns = ["target"])
y_test = df_test["target"]


rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,

    random_state=42,
)

rf_model.fit(X_train, y_train)

y_probs = rf_model.predict_proba(X_test)[:,1]

precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * recall * precision / (recall + precision + 1e-10)

best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

print(f"EN iyi threshold : {best_threshold:.4f}")
print(f"EN iyi f1-score :  {f1_scores[best_index]}")

y_pred = (y_probs >= best_threshold).astype(int)

cm = confusion_matrix(y_test, y_pred)


print(classification_report(y_test, y_pred))
print(cm)

importances = rf_model.feature_importances_
indices = np.argsort(importances)[-10:]

print("En onemli özellikler :")
for i in indices[::-1]:
    print(f"{X_train.columns[i]}: {importances[i]:.4f}")

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)
disp.plot(cmap="Blues")
plt.title("RandomForest-Confusion-Matrix")
plt.savefig(os.path.join(output_dir, "RandomForest-ConfusionMatrix.png"))

