from pathlib import Path
import numpy as np
import pandas as pd


TRAIN_DIR = Path("/home/emirfurkan/Desktop/power-transformer-rul/data/processed_merged/train_set")
VAL_DIR = Path("/home/emirfurkan/Desktop/power-transformer-rul/data/processed_merged/val_set")


def load_labels(split_dir: Path):

    y_fdd = np.load(split_dir / "y_fdd.npy")

    return y_fdd


def print_distribution(name, y):

    unique, counts = np.unique(y, return_counts=True)

    df = pd.DataFrame({
        "class": unique,
        "count": counts,
        "percentage": counts / len(y) * 100
    })

    print(f"\n{name} distribution")
    print(df)
    print("-" * 40)


if __name__ == "__main__":

    y_train = load_labels(TRAIN_DIR)
    y_val = load_labels(VAL_DIR)

    print_distribution("TRAIN", y_train)
    print_distribution("VALIDATION", y_val)