import pandas as pd
from sklearn.svm import SVC
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import  confusion_matrix , classification_report , ConfusionMatrixDisplay

df = pd.read_csv("/home/emirfurkan/Desktop/transformer_predictive_maintenance/data/processed/normalized_dataset_2019.csv")
output_dir = "/home/emirfurkan/Desktop/transformer_predictive_maintenance/results/"

X = df.drop(columns=["target"])
y= df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(
    kernel="rbf",
    class_weight="balanced",
    C=1.0,
    gamma="scale",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("2019 Veri Seti üzerinde SVM Modeli")

plt.savefig(os.path.join(output_dir, "2019-SVM-confusion_matrix.png"))