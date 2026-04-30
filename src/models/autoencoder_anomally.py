import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report , ConfusionMatrixDisplay
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
import matplotlib.pyplot as plt


df_train = pd.read_csv("/home/emirfurkan/Desktop/2019-2020-predictive-model/data/processed/normalized_dataset_2019.csv")
df_test = pd.read_csv("/home/emirfurkan/Desktop/2019-2020-predictive-model/data/processed/normalized_dataset_2020.csv")

output_dir  = "/home/emirfurkan/Desktop/2019-2020-predictive-model/results"

train_df, val_df = train_test_split(
    df_train,
    test_size=0.2,
    stratify=df_train["target"],
    random_state=42
)

X_train_normal = train_df[train_df["target"] == 0].drop("target", axis=1).values

X_val_normal = val_df[val_df["target"] == 0].drop("target", axis=1).values

X_test = df_test.drop("target", axis=1).values
y_test = df_test["target"].values

input_dim = X_train_normal.shape[1]

inputs = tf.keras.Input(shape=(input_dim,))

x = layers.Dense(32, activation="relu")(inputs)
x = layers.Dense(16, activation="relu")(x)
latent = layers.Dense(4, activation="relu", name="latent_space")(x)

x = layers.Dense(16, activation="relu")(latent)
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(input_dim, activation="linear")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
encoder = tf.keras.Model(inputs=inputs, outputs=latent)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

def print_reconstruction_metrics(epoch, logs):
    val_reconstructed = model.predict(X_val_normal, verbose=0)
    val_errors = np.mean((X_val_normal - val_reconstructed) ** 2, axis=1)
    threshold = np.percentile(val_errors, 95)

    test_reconstructed = model.predict(X_test, verbose=0)
    test_errors = np.mean((X_test - test_reconstructed) ** 2, axis=1)

    print(
        f" - loss: {logs['loss']:.5f}"
        f" - val_error_mean: {val_errors.mean():.5f}"
        f" - threshold_p95: {threshold:.5f}"
        f" - test_error_mean: {test_errors.mean():.5f}"
    )

model.compile(
    optimizer="adam",
    loss="mse"
)

history = model.fit(
    X_train_normal,
    X_train_normal,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    shuffle=True,
    callbacks=[early_stop, LambdaCallback(on_epoch_end=print_reconstruction_metrics)],
    verbose=1
)

reconstructed_val = model.predict(X_val_normal)
val_errors = np.mean((X_val_normal - reconstructed_val) ** 2, axis=1)

threshold = np.percentile(val_errors, 95)

reconstructed_test = model.predict(X_test)
test_errors = np.mean((X_test - reconstructed_test) ** 2, axis=1)

y_pred = np.where(test_errors > threshold, 1, 0)

acc = np.mean(y_pred == y_test)

cm = confusion_matrix(y_test, y_pred)

print("Model accuracy:", acc)
print("Threshold:", threshold)
print("Confusion Matrix:")
print(cm)
print(classification_report(y_test, y_pred, digits=4, zero_division=0))

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("AutoEncoder Model Confusion Matrix")
plt.savefig(os.path.join(output_dir, "AutoEncoder-ConfusionMatrix.png"))

