import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold , GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from visulization import display_cm

df = pd.read_csv("data/processed/dga_merged_processed.csv")

X = df.drop(columns=["Source_ID", "Type" , "Source","Fault_Type"])
y = df["Type"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

param_grid = {
    "max_depth" : [5,10,15,20,None],
    "min_samples_split" : [2,5,10] ,
    "criterion" : ["gini" , "entropy"] ,
    "splitter" : ["best"]
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42) , param_grid , cv=5)

grid_search.fit(X_train , y_train)

print("the best parametres :" , grid_search.best_params_)
model = grid_search.best_estimator_



model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
display_cm(
    y_test,
    y_pred,
    labels=sorted(y.unique()),
    model_name="DecisionTree",
)
