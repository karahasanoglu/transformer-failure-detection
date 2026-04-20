from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def display_cm(y_true, y_pred, labels, model_name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)

    ax.set_title(f"{model_name} Confusion Matrix")
    fig.tight_layout()

    output_path = RESULTS_DIR / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Confusion matrix saved to: {output_path}")
