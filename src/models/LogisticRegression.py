import os
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from preprocessing import preprocessing
from visulazition import plot_cm

X_train, y_train, X_test, y_test, scaler , encoder = preprocessing()

param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["lbfgs", "liblinear"],
    "class_weight": [None, "balanced"],
    "max_iter": [1000]
}

grid = GridSearchCV(
    LogisticRegression(),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("Best Params:", grid.best_params_)
print(classification_report(y_test, y_pred ,target_names=encoder.classes_))
print("Acc score:", accuracy_score(y_test, y_pred))

plot_cm(y_test, y_pred , class_names=encoder.classes_ , model_name='Logistic Regression')
