import os
import sys

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report ,accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from preprocessing import preprocessing
from visulazition import plot_cm

X_train, y_train, X_test, y_test, scaler , encoder= preprocessing()

rf = RandomForestClassifier(n_estimators=100 , random_state=42)

rf.fit(X_train, y_train)

rf_prd = rf.predict(X_test)

print(classification_report(y_test,rf_prd , target_names=encoder.classes_))

print("Acc Score : " , accuracy_score(y_test,rf_prd))

plot_cm(y_test,rf_prd , class_names=encoder.classes_, model_name='RandomForest')
