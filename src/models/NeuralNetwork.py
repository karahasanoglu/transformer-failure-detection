import os
import sys

from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping

from tensorflow.keras.models import Sequential

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from preprocessing import preprocessing_nn
from visulazition import plot_cm

X_train, y_train, X_test, y_test, scaler , encoder = preprocessing_nn()

num_classes = y_train.shape[1]

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(num_classes, activation="softmax")
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5 , restore_best_weights=True)

history = model.fit(
    X_train,
    y_train,
    epochs=40,
    validation_split=0.2,
    batch_size=32,
    callbacks= [es]
)

loss, acc = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
print("Test accuracy:", acc)
plot_cm(y_test , y_pred , class_names=encoder.classes_ , model_name="Neural Network")
