from pathlib import Path
import pandas as pd
import numpy as np
from train_grnn_fdd import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path("/home/emirfurkan/Desktop/power-transformer-rul/data/raw")

TRAIN_DIR = BASE_DIR / "data_train"
TEST_DIR = BASE_DIR / "data_test"
LABEL_DIR = BASE_DIR / "data_labels"

TRAIN_FDD_LABELS = LABEL_DIR / "labels_fdd_train.csv"
TEST_FDD_LABELS = LABEL_DIR / "labels_fdd_test.csv"


def load_data_from_folder(folder_path: Path):
    all_files = sorted(folder_path.glob("*.csv"))
    dfs = []

    for file_path in all_files:
        df = pd.read_csv(file_path)
        df["file_id"] = file_path.name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_label_file(path: Path, label_name: str):
    df = pd.read_csv(path).copy()


    if "id" not in df.columns:
        df.columns = ["id", label_name]
    else:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_cols) == 0:
            raise ValueError(f"Numeric label column not found in {path}")
        df = df.rename(columns={numeric_cols[0]: label_name})

    return df[["id", label_name]]


def create_features(df: pd.DataFrame):
    features = df.groupby("file_id").agg({
        "H2": ["mean", "std", "max", "min", "last"],
        "CO": ["mean", "std", "max", "min", "last"],
        "C2H4": ["mean", "std", "max", "min", "last"],
        "C2H2": ["mean", "std", "max", "min", "last"],
    })

    features.columns = ["_".join(col).strip() for col in features.columns.values]
    features = features.reset_index()


    eps = 1e-8
    features["R1_H2_CO"] = features["H2_last"] / (features["CO_last"] + eps)
    features["R2_C2H2_C2H4"] = features["C2H2_last"] / (features["C2H4_last"] + eps)
    features["R3_H2_C2H4"] = features["H2_last"] / (features["C2H4_last"] + eps)
    features["R4_CO_C2H2"] = features["CO_last"] / (features["C2H2_last"] + eps)

    return features


def main():
    data_train = load_data_from_folder(TRAIN_DIR)
    data_test = load_data_from_folder(TEST_DIR)

    labels_fdd_train = load_label_file(TRAIN_FDD_LABELS, "category")
    labels_fdd_test = load_label_file(TEST_FDD_LABELS, "category")

    train_features = create_features(data_train)
    test_features = create_features(data_test)

    train_df = train_features.merge(labels_fdd_train, left_on="file_id", right_on="id")
    test_df = test_features.merge(labels_fdd_test, left_on="file_id", right_on="id")

    X_train = train_df.drop(columns=["file_id", "id", "category"])
    y_train = train_df["category"]

    X_test = test_df.drop(columns=["file_id", "id", "category"])
    y_test = test_df["category"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # feature importance
    importances = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\nTop 15 Important Features:")
    print(importances.head(15))
    plot_confusion_matrix(confusion_matrix(y_test,y_pred) , np.unique(y_test) , title="Random Forest Fault Classification")


if __name__ == "__main__":
    main()