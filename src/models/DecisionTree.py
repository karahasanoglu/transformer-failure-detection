import os
import sys

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from preprocessing import preprocessing
from visulazition import plot_cm

X_train, y_train, X_test, y_test, scaler , encoder = preprocessing()

tree = DecisionTreeClassifier(max_depth=10 , random_state=42)
tree.fit(X_train, y_train)

tree_pred = tree.predict(X_test)

print(classification_report(y_test, tree_pred ,target_names=encoder.classes_))
print("Acc score :" , accuracy_score(y_test, tree_pred))
plot_cm(y_test, tree_pred , class_names=encoder.classes_ , model_name='DecisionTree')
