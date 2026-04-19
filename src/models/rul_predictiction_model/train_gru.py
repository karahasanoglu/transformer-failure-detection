from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping



TRAIN_DIR = Path("/home/emirfurkan/Desktop/power-transformer-rul/data/processed_merged/train_set")
VAL_DIR = Path("/home/emirfurkan/Desktop/power-transformer-rul/data/processed_merged/val_set")
TEST_DIR = Path("/home/emirfurkan/Desktop/power-transformer-rul/data/processed_merged/test_set")

TIMESTEPS = 50
N_FEATURES = 4
BATCH_SIZE = 32
EPOCHS = 50
RUL_CLIP = 500.0



X_train = np.load(TRAIN_DIR / "X.npy").astype(np.float32)
y_train = np.load(TRAIN_DIR / "y_rul.npy").astype(np.float32)

X_val = np.load(VAL_DIR / "X.npy").astype(np.float32)
y_val = np.load(VAL_DIR / "y_rul.npy").astype(np.float32)

X_test = np.load(TEST_DIR / "X.npy").astype(np.float32)
y_test = np.load(TEST_DIR / "y_rul.npy").astype(np.float32)

# sadece son 50 timestep
X_train = X_train[:, -TIMESTEPS:, :]
X_val = X_val[:, -TIMESTEPS:, :]
X_test = X_test[:, -TIMESTEPS:, :]

print("Train:", X_train.shape, y_train.shape)
print("Val  :", X_val.shape, y_val.shape)
print("Test :", X_test.shape, y_test.shape)



x_mean = X_train.mean(axis=(0, 1), keepdims=True)
x_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

X_train = (X_train - x_mean) / x_std
X_val = (X_val - x_mean) / x_std
X_test = (X_test - x_mean) / x_std



y_train = np.clip(y_train, 0, RUL_CLIP)
y_val = np.clip(y_val, 0, RUL_CLIP)
y_test = np.clip(y_test, 0, RUL_CLIP)

y_train = y_train / RUL_CLIP
y_val = y_val / RUL_CLIP
y_test = y_test / RUL_CLIP



model = Sequential([
    Input(shape=(TIMESTEPS, N_FEATURES)),
    GRU(64),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mae",
    metrics=["mae"]
)

model.summary()



early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)



def evaluate(name, X, y_true):
    y_pred = model.predict(X, verbose=0).flatten()

    # tekrar gerçek ölçeğe çevir
    y_pred = y_pred * RUL_CLIP
    y_true = y_true * RUL_CLIP

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)
    print("MAE :", round(mae, 3))
    print("RMSE:", round(rmse, 3))

    print("\nFirst 10 predictions:")
    for i in range(10):
        print(f"true={y_true[i]:.2f}   pred={y_pred[i]:.2f}")


evaluate("VALIDATION", X_val, y_val)
evaluate("TEST", X_test, y_test)

