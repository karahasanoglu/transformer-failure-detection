from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

DATASET1_PATH = RAW_DIR / "DGA-dataset.csv"
DATASET2_PATH = RAW_DIR / "DGA_dataset2.csv"
MERGED_RAW_PATH = RAW_DIR / "dga_merged_dataset.csv"
MERGED_PROCESSED_PATH = PROCESSED_DIR / "dga_merged_processed.csv"


DATASET1_MAPPING = {
    "Partial discharge": "PD",
    "Spark discharge": "D1",
    "Arc discharge": "D2",
    "Low-temperature overheating": "T1",
    "Low/Middle-temperature overheating": "T2",
    "Middle-temperature overheating": "T2",
    "High-temperature overheating": "T3",
}


VALID_CLASSES = ["D1", "D2", "NF", "PD", "T1", "T2", "T3"]
FEATURE_COLUMNS = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]


def load_dataset1() -> pd.DataFrame:
    df = pd.read_csv(DATASET1_PATH)
    df["Type"] = df["Type"].astype(str).str.strip()
    df["Type"] = df["Type"].map(DATASET1_MAPPING)
    df = df.rename(columns={"NM": "Source_ID"})
    df["Source"] = "dataset1"
    return df[["Source_ID", *FEATURE_COLUMNS, "Type", "Source"]]


def load_dataset2() -> pd.DataFrame:
    df = pd.read_csv(DATASET2_PATH, sep=";", decimal=",", encoding="utf-8-sig")
    df["Fail"] = df["Fail"].astype(str).str.strip()
    df = df.rename(columns={"Fail": "Type"})
    df["Source_ID"] = np.arange(1, len(df) + 1)
    df["Source"] = "dataset2"
    return df[["Source_ID", *FEATURE_COLUMNS, "Type", "Source"]]


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    df["R1"] = df["CH4"] / df["H2"]
    df["R2"] = df["C2H2"] / df["C2H4"]
    df["R4"] = df["C2H6"] / df["CH4"]
    df["R5"] = df["C2H4"] / df["C2H6"]
    return df


def apply_iqr_scaling(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        q1 = df[col].quantile(0.25)
        q2 = df[col].quantile(0.50)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            df[col] = 0.0
        else:
            df[col] = (df[col] - q2) / iqr

    return df


def main() -> None:
    df1 = load_dataset1()
    df2 = load_dataset2()

    print("Dataset 1 shape:", df1.shape)
    print("Dataset 2 shape:", df2.shape)

    merged_df = pd.concat([df1, df2], ignore_index=True)
    merged_df = merged_df[merged_df["Type"].isin(VALID_CLASSES)].copy()

    print("Merged raw shape:", merged_df.shape)
    print("\nClass distribution before cleaning:")
    print(merged_df["Type"].value_counts().sort_index())

    encoder = LabelEncoder()
    merged_df["Fault_Type"] = encoder.fit_transform(merged_df["Type"])

    print("\nLabel encoding:")
    for label, encoded in zip(encoder.classes_, encoder.transform(encoder.classes_)):
        print(f"{label} -> {encoded}")

    merged_df = add_ratio_features(merged_df)
    merged_df = merged_df.replace([np.inf, -np.inf], np.nan)
    merged_df = merged_df.dropna().reset_index(drop=True)

    scaled_columns = FEATURE_COLUMNS + ["R1", "R2", "R4", "R5"]
    merged_df = apply_iqr_scaling(merged_df, scaled_columns)

    print("\nClass distribution after cleaning:")
    print(merged_df["Type"].value_counts().sort_index())
    print("\nProcessed shape:", merged_df.shape)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pd.concat([df1, df2], ignore_index=True).to_csv(MERGED_RAW_PATH, index=False)
    merged_df.to_csv(MERGED_PROCESSED_PATH, index=False)

    print(f"\nMerged raw dataset saved to: {MERGED_RAW_PATH}")
    print(f"Processed dataset saved to: {MERGED_PROCESSED_PATH}")


if __name__ == "__main__":
    main()
