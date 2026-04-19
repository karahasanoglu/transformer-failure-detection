from pathlib import Path
import pandas as pd

from src.data_preprocessing.load_data import load_dataset, TransformerSample


REQUIRED_COLUMNS = ["H2", "CO", "C2H4", "C2H2"]


def validate_sample(sample: TransformerSample, idx: int):
    if not isinstance(sample, TransformerSample):
        raise TypeError(f"Sample at index {idx} is not a TransformerSample object")

    if not isinstance(sample.file_name, str) or not sample.file_name.endswith(".csv"):
        raise ValueError(f"Invalid file_name at index {idx}: {sample.file_name}")

    if sample.split not in ["train", "test"]:
        raise ValueError(f"Invalid split at index {idx}: {sample.split}")

    if not isinstance(sample.operating_condition, int):
        raise TypeError(f"operating_condition must be int at index {idx}")

    if not isinstance(sample.transformer_id, int):
        raise TypeError(f"transformer_id must be int at index {idx}")

    if not isinstance(sample.data, pd.DataFrame):
        raise TypeError(f"data must be a pandas DataFrame at index {idx}")

    if sample.data.empty:
        raise ValueError(f"Empty dataframe at index {idx} ({sample.file_name})")

    for col in REQUIRED_COLUMNS:
        if col not in sample.data.columns:
            raise ValueError(f"Missing column '{col}' in {sample.file_name}")

    for col in REQUIRED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(sample.data[col]):
            raise TypeError(f"Column '{col}' is not numeric in {sample.file_name}")

    if sample.data[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError(f"Missing values found in gas columns of {sample.file_name}")

    if sample.fdd_label is None:
        raise ValueError(f"fdd_label is missing in {sample.file_name}")

    if sample.rul_label is None:
        raise ValueError(f"rul_label is missing in {sample.file_name}")

    if not isinstance(sample.fdd_label, (int, float)):
        raise TypeError(f"fdd_label must be numeric in {sample.file_name}")

    if not isinstance(sample.rul_label, (int, float)):
        raise TypeError(f"rul_label must be numeric in {sample.file_name}")


def validate_loaded_dataset(train_samples, test_samples):
    if len(train_samples) == 0:
        raise ValueError("train_samples is empty")

    if len(test_samples) == 0:
        raise ValueError("test_samples is empty")

    for i, sample in enumerate(train_samples):
        validate_sample(sample, i)

    for i, sample in enumerate(test_samples):
        validate_sample(sample, i)

    print("Validation completed successfully.")
    print(f"Train sample count: {len(train_samples)}")
    print(f"Test sample count : {len(test_samples)}")

    print("\nRandom train data sample summary:")
    random_data = train_samples[200]
    print(f"file_name           : {random_data.file_name}")
    print(f"split               : {random_data.split}")
    print(f"operating_condition : {random_data.operating_condition}")
    print(f"transformer_id      : {random_data.transformer_id}")
    print(f"shape               : {random_data.data.shape}")
    print(f"columns             : {list(random_data.data.columns)}")
    print(f"fdd_label           : {random_data.fdd_label}")
    print(f"rul_label           : {random_data.rul_label}")


if __name__ == "__main__":
    train_samples, test_samples = load_dataset()
    validate_loaded_dataset(train_samples, test_samples)