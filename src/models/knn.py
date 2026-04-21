import sys
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score , classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.preprocessing import process_data
from src.visualization import plot_cm
X_train , X_test, y_train, y_test = process_data()

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)



print("Acc score :",accuracy_score(y_test, knn.predict(X_test)))
print(classification_report(y_test, knn.predict(X_test)))
plot_cm(y_test, knn.predict(X_test) , model_name = 'knn')
