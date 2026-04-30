import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

df_2019 = pd.read_csv("/home/emirfurkan/Desktop/transformer_predictive_maintenance/data/processed/normalized_dataset_2019.csv")
df_2020 = pd.read_csv("/home/emirfurkan/Desktop/transformer_predictive_maintenance/data/processed/normalized_dataset_2020.csv")

output_dir = "/home/emirfurkan/Desktop/transformer_predictive_maintenance/results/"

df_2019["yil"] = 2019
df_2020["yil"] = 2020

df = pd.concat([df_2019, df_2020], ignore_index=True)

X = df.drop(columns=["target","yil"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel="rbf" , class_weight="balanced" , C=1.0 , random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot(cmap="Blues")
plt.title("2019-2020 Veri Seti üzerinde SVM Modeli")

plt.savefig(os.path.join(output_dir, "2019-2020-SVM-confusion_matrix.png"))
