import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DEFAULT_RESULTS_DIR = os.path.join(BASE_DIR, "results")


def plot_cm(
    y_true,
    y_pred,
    class_names=None,
    model_name="model",
    save_dir=DEFAULT_RESULTS_DIR,
    show=False,
    cmap="Blues"
):


    # numpy çevir
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # one-hot ise class index'e çevir
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)

    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)


    cm = confusion_matrix(y_true, y_pred)

    # klasör oluştur
    os.makedirs(save_dir, exist_ok=True)


    fig, ax = plt.subplots(figsize=(8, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(ax=ax, cmap=cmap, values_format="d", colorbar=False)

    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()

    # kaydet
    file_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(file_path, dpi=120)

    if show:
        plt.show()

    plt.close()

    print(f"CM saved -> {file_path}")