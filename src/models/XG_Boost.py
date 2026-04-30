import pandas as pd
import os
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report , ConfusionMatrixDisplay

df_train = pd.read_csv("/home/emirfurkan/Desktop/2019-2020-predictive-model/data/processed/normalized_dataset_2019.csv")
df_test = pd.read_csv("/home/emirfurkan/Desktop/2019-2020-predictive-model/data/processed/normalized_dataset_2020.csv")

output_dir = "/home/emirfurkan/Desktop/2019-2020-predictive-model/results"

X_train = df_train.drop(columns = ["target"])
y_train = df_train["target"]

X_test = df_test.drop(columns = ["target"])
y_test = df_test["target"]

scale_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

model_xgb = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.01,

    random_state=42
)


model_xgb.fit(X_train, y_train)
y_pred = model_xgb.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)
disp.plot(cmap="Blues")
plt.title("XG-Boost-Confusion-Matrix")
plt.savefig(os.path.join(output_dir, "XG-Boost-ConfusionMatrix.png"))
