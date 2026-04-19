from pathlib import Path


import numpy as np
import pandas as pd
from src.data_preprocessing.load_data import load_dataset
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = ["H2", "CO", "C2H4", "C2H2"]
SEQUENCE_LENGTH = 200

VAL_RATIO = 0.2

def split_train_validation(X,y_fdd,y_rul):

    X_train , X_val , y_fdd_train , y_fdd_val , y_rul_train , y_rul_val = train_test_split(X , y_fdd , y_rul , test_size = VAL_RATIO
                                                                                           , random_state = 42 , shuffle=True)

    return X_train , X_val , y_fdd_train , y_fdd_val , y_rul_train , y_rul_val

def _last_sequence(sequence: np.ndarray, seq_len: int) -> np.ndarray:

    if len(sequence) >= seq_len:
        return sequence[-seq_len:]

    pad_len = seq_len - len(sequence)

    pad = np.zeros((pad_len, sequence.shape[1]), dtype=sequence.dtype)

    return np.vstack([pad, sequence])


def build_arrays(samples):

    X = []
    y_fdd = []
    y_rul = []

    for sample in samples:

        seq = sample.data[FEATURE_COLUMNS].values.astype(np.float32)

        seq = _last_sequence(seq, SEQUENCE_LENGTH)

        X.append(seq)
        y_fdd.append(sample.fdd_label)
        y_rul.append(sample.rul_label)

    X = np.array(X, dtype=np.float32)
    y_fdd = np.array(y_fdd)
    y_rul = np.array(y_rul, dtype=np.float32)

    return X, y_fdd, y_rul


def save_arrays(output_dir, X, y_fdd, y_rul):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y_fdd.npy", y_fdd)
    np.save(output_dir / "y_rul.npy", y_rul)


if __name__ == "__main__":

    train_samples, test_samples = load_dataset()

    X_train, y_fdd_train, y_rul_train = build_arrays(train_samples)

    X_test, y_fdd_test, y_rul_test = build_arrays(test_samples)

    X_train, X_val, y_fdd_train, y_fdd_val, y_rul_train, y_rul_val = split_train_validation(
        X_train,
        y_fdd_train,
        y_rul_train
    )

    save_arrays("/home/emirfurkan/Desktop/power-transformer-rul/data/processed_merged/train_set", X_train, y_fdd_train, y_rul_train)

    save_arrays("/home/emirfurkan/Desktop/power-transformer-rul/data/processed_merged/val_set" , X_val, y_fdd_val, y_rul_val)
    save_arrays("/home/emirfurkan/Desktop/power-transformer-rul/data/processed_merged/test_set", X_test, y_fdd_test, y_rul_test)

    print("Train shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)

    print("Test for Numpy Files")
    X = np.load("/home/emirfurkan/Desktop/power-transformer-rul/data/processed_merged/train_set/X.npy")

    df = pd.DataFrame(X[0] , columns = FEATURE_COLUMNS)

    print(df.head())



