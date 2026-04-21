import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "classData.csv")

def preprocessing():

    data = pd.read_csv(DATA_PATH)

    fault_mapping = {
        (0, 0, 0, 0): "No Fault",
        (1, 0, 0, 1): "LG Fault",
        (0, 1, 1, 0): "LL Fault",
        (1, 0, 1, 1): "LLG Fault",
        (0, 1, 1, 1): "LLL Fault",
        (1, 1, 1, 1): "GLLL Fault"
    }

    fault_keys = [tuple(row) for row in data.iloc[:, :4].values]
    data["Fault_Type"] = [fault_mapping.get(f, "Unknown Fault") for f in fault_keys]

    X = data.iloc[:, 4:10]
    y = data["Fault_Type"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)   # 1D label

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, scaler, encoder



def preprocessing_nn():

    data = pd.read_csv(DATA_PATH)

    fault_mapping = {
        (0, 0, 0, 0): "No Fault",
        (1, 0, 0, 1): "LG Fault",
        (0, 1, 1, 0): "LL Fault",
        (1, 0, 1, 1): "LLG Fault",
        (0, 1, 1, 1): "LLL Fault",
        (1, 1, 1, 1): "GLLL Fault"
    }

    fault_keys = [tuple(row) for row in data.iloc[:, :4].values]
    data["Fault_Type"] = [fault_mapping.get(f, "Unknown Fault") for f in fault_keys]

    X = data.iloc[:, 4:10]
    y = data["Fault_Type"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, y_train, X_test, y_test, scaler, encoder
