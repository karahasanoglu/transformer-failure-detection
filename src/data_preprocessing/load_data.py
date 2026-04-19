from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass
class TransformerSample:
    file_name: str
    split: str
    operating_condition: int
    transformer_id: int
    data: pd.DataFrame
    fdd_label: float | None = None
    rul_label: float | None = None


def _parse_name(file_name: str):
    parts = Path(file_name).stem.split("_")
    return int(parts[0]), int(parts[2])


def _read_label_file(path: str | Path, col_name: str):
    df = pd.read_csv(path)

    if df.shape[1] == 1:
        series = df.iloc[:, 0]
    else:
        numeric_df = df.select_dtypes(include=["number"])
        if not numeric_df.empty:
            series = numeric_df.iloc[:, 0]
        else:
            series = df.iloc[:, -1]

    series = pd.to_numeric(series, errors="coerce")

    if series.isnull().any():
        raise ValueError(f"{path} içinde sayısala çevrilemeyen label var")

    return series.tolist()


def _load_split(data_dir: str | Path, split: str, fdd_labels, rul_labels):
    files = sorted(Path(data_dir).glob("*.csv"))
    samples = []

    if len(files) != len(fdd_labels) or len(files) != len(rul_labels):
        raise ValueError(f"{split}: file count and label count do not match")

    for i, file_path in enumerate(files):
        op_cond, trans_id = _parse_name(file_path.name)

        samples.append(
            TransformerSample(
                file_name=file_path.name,
                split=split,
                operating_condition=op_cond,
                transformer_id=trans_id,
                data=pd.read_csv(file_path),
                fdd_label=fdd_labels[i],
                rul_label=rul_labels[i],
            )
        )

    return samples


def load_dataset():
    base = Path("data/raw")
    labels = base / "data_labels"

    train_fdd = _read_label_file(labels / "/home/emirfurkan/Desktop/power-transformer-rul/data/raw/data_labels/labels_fdd_train.csv", "fdd")
    test_fdd = _read_label_file(labels / "/home/emirfurkan/Desktop/power-transformer-rul/data/raw/data_labels/labels_fdd_test.csv", "fdd")
    train_rul = _read_label_file(labels / "/home/emirfurkan/Desktop/power-transformer-rul/data/raw/data_labels/labels_rul_train.csv", "rul")
    test_rul = _read_label_file(labels / "/home/emirfurkan/Desktop/power-transformer-rul/data/raw/data_labels/labels_rul_test.csv", "rul")

    train_samples = _load_split(base / "/home/emirfurkan/Desktop/power-transformer-rul/data/raw/data_train", "train", train_fdd, train_rul)
    test_samples = _load_split(base / "/home/emirfurkan/Desktop/power-transformer-rul/data/raw/data_test", "test", test_fdd, test_rul)

    return train_samples, test_samples


if __name__ == "__main__":
    train_samples, test_samples = load_dataset()

    print("Train:", len(train_samples))
    print("Test :", len(test_samples))
    print()
    print(train_samples[0].file_name)
    print("FDD:", train_samples[0].fdd_label)
    print("RUL:", train_samples[0].rul_label)
    print(train_samples[0].data.head())