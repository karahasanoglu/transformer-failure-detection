import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder

from visulization import display_cm

df = pd.read_csv("/home/emirfurkan/Desktop/CAI_DGA_Project/data/raw/dga_merged_dataset.csv")


le = LabelEncoder()
df["Type"] = le.fit_transform(df["Type"])


X = df.drop(columns=["Source_ID", "Type", "Source"])
y = df["Type"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# svmde data_preprocessing.py dosyasında tanımlanmış olan IQR scalizasyondan farklı olarak Robust Kullanıyoruz
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = SVC(kernel="rbf", C=100, gamma="auto", class_weight="balanced")


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv)

print("CV Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())


model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)


accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
display_cm(
    y_test,
    y_pred,
    labels=list(range(len(le.classes_))),
    model_name="SVM",
)
