import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score ,classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from visulization import display_cm

df = pd.read_csv("data/processed/dga_merged_processed.csv")

X = df.drop(columns=["Source_ID", "Type" , "Source" ,"Fault_Type"])
y = df["Type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 , stratify=y)

print("Full dataset:")
print(y.value_counts().sort_index())

print("\nTrain:")
print(y_train.value_counts().sort_index())

print("\nTest:")
print(y_test.value_counts().sort_index())

model = RandomForestClassifier( n_estimators=200, max_depth=10, random_state=42 , class_weight="balanced")

scores = cross_val_score(model,X,y,cv=5)

print(f"CV Score: {scores}")
print("Mean CV Accuracy:" , scores.mean())
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)

print(f"Accuracy Score: {acc_score}")

print(classification_report(y_test, y_pred))
display_cm(
    y_test,
    y_pred,
    labels=sorted(y.unique()),
    model_name="RandomForest",
)
