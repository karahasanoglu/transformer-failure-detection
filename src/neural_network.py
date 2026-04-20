import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from visulization import display_cm

DATA_PATH = "data/processed/dga_merged_processed.csv"


df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Source_ID", "Type", "Source", "Fault_Type"])
y = df["Fault_Type"]

# Numeric class id -> original label name mapping
label_map = (
    df[["Fault_Type", "Type"]]
    .drop_duplicates()
    .sort_values("Fault_Type")
)
target_names = label_map["Type"].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Full dataset:")
print(df["Type"].value_counts().sort_index())

print("\nTrain:")
print(df.loc[y_train.index, "Type"].value_counts().sort_index())

print("\nTest:")
print(df.loc[y_test.index, "Type"].value_counts().sort_index())



class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train,
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

num_classes = y.nunique()

model = Sequential(
    [
        Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True,
)

model.fit(
    X_train,
    y_train,
    epochs=250,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    class_weight=class_weight_dict,
    verbose=1,
)

y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
display_cm(
    y_test,
    y_pred,
    labels=sorted(y.unique()),
    model_name="Neural Network",
)
