import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.preprocessing import process_data
from src.visualization import plot_cm
X_train , X_test, y_train, y_test = process_data()

model = LogisticRegression()
model.fit(X_train, y_train)

model.predict(X_test)

print(classification_report(y_test, model.predict(X_test)))
print("Accuracy Score : {}".format(accuracy_score(y_test, model.predict(X_test))))

plot_cm(y_test, model.predict(X_test) , model_name='Logistic Regression')
