from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path("/home/emirfurkan/Desktop/power-transformer-rul/data/raw")

TRAIN_DIR = BASE_DIR / "data_train"
TEST_DIR = BASE_DIR / "data_test"
LABEL_DIR = BASE_DIR / "data_labels"

SIGMA = 0.5
EPS = 1e-8



def load_folder(folder_path):
    all_data = []

    for file_path in sorted(folder_path.glob("*.csv")):
        df = pd.read_csv(file_path)
        df["file_id"] = file_path.name
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)



def load_labels(path):
    df = pd.read_csv(path)

    # ilk kolon dosya adı, ikinci kolon label varsayımı
    if df.shape[1] == 2:
        df.columns = ["id", "category"]
    else:
        df = df.iloc[:, :2]
        df.columns = ["id", "category"]

    return df



def create_features(df):
    grouped = df.groupby("file_id").agg({
        "H2": ["mean", "std", "max", "min", "last"],
        "CO": ["mean", "std", "max", "min", "last"],
        "C2H4": ["mean", "std", "max", "min", "last"],
        "C2H2": ["mean", "std", "max", "min", "last"],
    })

    grouped.columns = ["_".join(col) for col in grouped.columns]
    grouped = grouped.reset_index()

    # ratio features
    grouped["R1_H2_CO"] = grouped["H2_last"] / (grouped["CO_last"] + EPS)
    grouped["R2_C2H2_C2H4"] = grouped["C2H2_last"] / (grouped["C2H4_last"] + EPS)
    grouped["R3_H2_C2H4"] = grouped["H2_last"] / (grouped["C2H4_last"] + EPS)
    grouped["R4_CO_C2H2"] = grouped["CO_last"] / (grouped["C2H2_last"] + EPS)

    return grouped



class GRNNClassifier:
    def __init__(self, sigma=0.5):
        self.sigma = sigma

    def fit(self, X, y):
        self.X_train = np.array(X, dtype=np.float32)
        self.y_train = np.array(y)
        self.classes_ = np.unique(y)

    def predict(self, X):
        X = np.array(X, dtype=np.float32)
        predictions = []

        for x in X:
            diff = self.X_train - x
            dist2 = np.sum(diff ** 2, axis=1)
            weights = np.exp(-dist2 / (2 * (self.sigma ** 2)))

            scores = []
            for cls in self.classes_:
                score = np.sum(weights[self.y_train == cls])
                scores.append(score)

            pred_class = self.classes_[np.argmax(scores)]
            predictions.append(pred_class)

        return np.array(predictions)



def plot_confusion_matrix(cm, class_names,title):

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()



train_data = load_folder(TRAIN_DIR)
test_data = load_folder(TEST_DIR)

train_labels = load_labels(LABEL_DIR / "labels_fdd_train.csv")
test_labels = load_labels(LABEL_DIR / "labels_fdd_test.csv")

train_features = create_features(train_data)
test_features = create_features(test_data)

train_df = train_features.merge(train_labels, left_on="file_id", right_on="id")
test_df = test_features.merge(test_labels, left_on="file_id", right_on="id")

X_train = train_df.drop(columns=["file_id", "id", "category"])
y_train = train_df["category"]

X_test = test_df.drop(columns=["file_id", "id", "category"])
y_test = test_df["category"]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = GRNNClassifier(sigma=SIGMA)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


def plot_confusion_matrix(cm, class_names, title) :

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()



train_data = load_folder(TRAIN_DIR)
test_data = load_folder(TEST_DIR)

train_labels = load_labels(LABEL_DIR / "labels_fdd_train.csv")
test_labels = load_labels(LABEL_DIR / "labels_fdd_test.csv")

train_features = create_features(train_data)
test_features = create_features(test_data)

train_df = train_features.merge(train_labels, left_on="file_id", right_on="id")
test_df = test_features.merge(test_labels, left_on="file_id", right_on="id")

X_train = train_df.drop(columns=["file_id", "id", "category"])
y_train = train_df["category"]

X_test = test_df.drop(columns=["file_id", "id", "category"])
y_test = test_df["category"]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = GRNNClassifier(sigma=SIGMA)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plot_confusion_matrix(
    cm=cm,
    class_names=np.unique(y_test),
    title="GRNN Fault Classification"
)