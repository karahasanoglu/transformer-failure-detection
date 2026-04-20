"""Supervised SVM pipeline for transformer fault prediction."""
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from src.metrics_utils import evaluate_predictions, find_best_threshold, print_evaluation

class TransformerSVM:
    def __init__(
        self,
        kernel: str = 'rbf',
        C: float = 1.0,
        gamma: str | float = 'scale',
        use_smote: bool = False,
        class_weight: str | dict | None = 'balanced',
        threshold: float = 0.5,
    ):
        self.use_smote = use_smote
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,
            random_state=42,
            cache_size=2000,
            class_weight=class_weight,
        )
        self._smote = SMOTE(
            random_state=42,
            k_neighbors=5,
            sampling_strategy='auto',
        )
        self._threshold: float = threshold

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Gercek etiketlerle egitir."""
        X_fit, y_fit = X_train, y_train
        if self.use_smote:
            X_fit, y_fit = self._smote.fit_resample(X_train, y_train)
            counts = np.bincount(y_fit)
            print(f"[SVM] SMOTE Sonrası Dağılım : 0={counts[0]}  1={counts[1]}")
        else:
            counts = np.bincount(np.asarray(y_train, dtype=int))
            print(f"[SVM] Eğitim Dağılımı      : 0={counts[0]}  1={counts[1]} (SMOTE kapalı)")

        self.model.fit(X_fit, y_fit)
        print("[SVM] Eğitim Tamamlandı.")
        print(f"[SVM] Karar Eşiği         : {self._threshold:.2f}")

    def optimize_threshold(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        min_recall: float = 0.20,
    ) -> dict:
        y_prob = self.model.predict_proba(X_val)[:, 1]
        self._threshold, metrics = find_best_threshold(
            np.asarray(y_val),
            y_prob,
            min_recall=min_recall,
        )
        print(f"[SVM] Validation Optimum Eşik : {self._threshold:.3f}")
        print(
            "[SVM] Validation Metrikleri    : "
            f"Recall={metrics.get('recall', 0):.4f}  "
            f"Specificity={metrics.get('specificity', 0):.4f}  "
            f"BalancedAcc={metrics.get('balanced_accuracy', 0):.4f}"
        )
        return metrics

    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        param_grid = {
            'C': [0.5, 1, 5, 10, 20],
            'gamma': ['scale', 0.1, 0.01, 0.001],
            'kernel': ['rbf'],
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        search = GridSearchCV(
            estimator=SVC(
                probability=True,
                random_state=42,
                cache_size=2000,
                class_weight=self.model.class_weight,
            ),
            param_grid=param_grid,
            scoring='balanced_accuracy',
            n_jobs=-1,
            cv=cv,
            refit=True,
        )
        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        print(f"[SVM] Grid Search En İyi Parametreler : {search.best_params_}")
        print(f"[SVM] Grid Search En İyi Skor         : {search.best_score_:.4f}")

    def evaluate(self, X_test: np.ndarray, y_actual: np.ndarray) -> dict:
        y_prob = self.model.predict_proba(X_test)[:, 1]
        results = evaluate_predictions(np.asarray(y_actual), y_prob, self._threshold)
        print_evaluation("TEST SETİ PERFORMANS SONUÇLARI", results)
        return results

    def predict_with_threshold(self, X: np.ndarray) -> dict:
        y_prob = self.model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= self._threshold).astype(int)
        return {"y_pred": y_pred, "y_prob": y_prob, "threshold": self._threshold}
