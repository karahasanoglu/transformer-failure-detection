import warnings
import os
from sklearn.model_selection import train_test_split
import numpy as np

warnings.filterwarnings('ignore')

from src.preprocessing import preprocess_transformer_data, load_full_feature_matrix
from src.visualization import save_confusion_matrix
from src.svm_model import TransformerSVM

DATA_PATH_2019 = "data/raw/Dataset_Year_2019.xlsx"
DATA_PATH_2020 = "data/raw/Dataset_Year_2020.xlsx"

REDUCED_FEATURES_TO_DROP = [
    'LOCATION',
    'POWER',
    'REMOVABLE_CONNECTORS',
    'ENERGY_NOT_SUPPLIED',
    'AIR_NETWORK',
    'CIRCUIT_QUEUE',
]


def print_learning_summary(title: str, y_true, y_pred) -> None:
    y_true_arr = y_true.to_numpy() if hasattr(y_true, "to_numpy") else y_true
    class0_ok = int(((y_true_arr == 0) & (y_pred == 0)).sum())
    class0_bad = int(((y_true_arr == 0) & (y_pred == 1)).sum())
    class1_ok = int(((y_true_arr == 1) & (y_pred == 1)).sum())
    class1_bad = int(((y_true_arr == 1) & (y_pred == 0)).sum())

    print(f"\n{'-' * 55}")
    print(f"  {title}")
    print(f"{'-' * 55}")
    print(f"  Class -1 doğru öğrenilen : {class0_ok}")
    print(f"  Class -1 hatalı öğrenilen: {class0_bad}")
    print(f"  Class +1 doğru öğrenilen : {class1_ok}")
    print(f"  Class +1 hatalı öğrenilen: {class1_bad}")


def print_projection_summary(projected_failures: int, paper_reference: int = 852) -> None:
    difference = projected_failures - paper_reference
    sign = "+" if difference >= 0 else ""
    print(f"\n{'=' * 55}")
    print("  2021 ARIZA SAYISI PROJEKSİYONU")
    print(f"{'=' * 55}")
    print(f"  Bu projedeki tahmin      : {projected_failures}")
    print(f"  Makaledeki referans      : {paper_reference}")
    print(f"  Fark                     : {sign}{difference}")


def train_combined_projection_model(feature_names: list[str]) -> tuple[TransformerSVM, object]:
    X_2019, y_2019, _ = load_full_feature_matrix(
        DATA_PATH_2019,
        feature_names,
        scaler=None,
    )
    X_2020, y_2020, _ = load_full_feature_matrix(
        DATA_PATH_2020,
        feature_names,
        scaler=None,
    )

    X_all = np.vstack([X_2019, X_2020])
    y_all = np.concatenate([y_2019.to_numpy(), y_2020.to_numpy()])

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_all_scaled,
        y_all,
        test_size=0.20,
        random_state=42,
        stratify=y_all,
    )

    print("\n" + "=" * 55)
    print("  2021 PROJEKSİYONU İÇİN BİRLEŞİK MODEL")
    print("=" * 55)
    print(f"[Projection] Birleşik eğitim boyutu : {X_all_scaled.shape}")
    print(f"[Projection] Validation boyutu      : {X_val.shape}")

    combined_model = TransformerSVM(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        use_smote=False,
        class_weight='balanced',
        threshold=0.5,
    )
    combined_model.train(X_fit, y_fit)
    combined_model.optimize_threshold(X_val, y_val, min_recall=0.30)
    return combined_model, scaler


def run_project_svm(title: str, path: str, prefix: str) -> dict:
    print("\n" + "=" * 55 + f"\n  {title}\n" + "=" * 55)
    X_train, X_test, y_train, y_test, scaler, feature_names, _, _ = preprocess_transformer_data(
        path,
        test_size=1000,
        random_state=42,
        article_mode=True,
    )

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train.to_numpy(),
        test_size=0.20,
        random_state=42,
        stratify=y_train.to_numpy(),
    )
    print(f"[Validation] Train alt küme : {X_fit.shape} | Validation: {X_val.shape}")

    print("\n" + "-" * 55)
    print("  AŞAMA 1: SVM (Tüm Seçili Değişkenler)")
    print("-" * 55)
    svm_full = TransformerSVM(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        use_smote=False,
        class_weight='balanced',
        threshold=0.5,
    )
    svm_full.train(X_fit, y_fit)
    svm_full.optimize_threshold(X_val, y_val, min_recall=0.30)
    full_results = svm_full.evaluate(X_test, y_test)
    save_confusion_matrix(
        y_test,
        full_results['y_pred'],
        f"{title} SVM Full",
        f"{prefix}_svm_full",
    )
    print_learning_summary("Öğrenme Özeti (Tüm Değişkenler)", y_test, full_results['y_pred'])

    print("\n" + "-" * 55)
    print("  AŞAMA 2: Feature Azaltılmış SVM")
    print("-" * 55)
    kept_features = [feat for feat in feature_names if feat not in REDUCED_FEATURES_TO_DROP]
    kept_indices = [feature_names.index(feat) for feat in kept_features]
    X_fit_reduced = X_fit[:, kept_indices]
    X_val_reduced = X_val[:, kept_indices]
    X_test_reduced = X_test[:, kept_indices]
    print(f"[Model] Çıkarılan feature'lar  : {', '.join(REDUCED_FEATURES_TO_DROP)}")
    print(f"[Model] Kalan feature sayısı   : {len(kept_features)}")

    svm_reduced = TransformerSVM(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        use_smote=False,
        class_weight='balanced',
        threshold=0.5,
    )
    svm_reduced.train(X_fit_reduced, y_fit)
    svm_reduced.optimize_threshold(X_val_reduced, y_val, min_recall=0.30)
    reduced_results = svm_reduced.evaluate(X_test_reduced, y_test)
    save_confusion_matrix(
        y_test,
        reduced_results['y_pred'],
        f"{title} SVM Reduced",
        f"{prefix}_svm_reduced",
    )
    print_learning_summary("Öğrenme Özeti (Feature Azaltılmış)", y_test, reduced_results['y_pred'])

    selected_model = svm_reduced if reduced_results['f1'] >= full_results['f1'] else svm_full
    selected_features = kept_features if reduced_results['f1'] >= full_results['f1'] else feature_names
    selected_name = "Reduced SVM" if reduced_results['f1'] >= full_results['f1'] else "Full SVM"
    selected_results = reduced_results if reduced_results['f1'] >= full_results['f1'] else full_results
    print(f"[Seçim] {prefix} için tercih edilen model: {selected_name} (F1={selected_results['f1']:.4f})")

    return {
        'model': selected_model,
        'feature_names': selected_features,
        'scaler': scaler,
        'results': {
            'full': full_results,
            'reduced': reduced_results,
        },
    }


def main():
    if not os.path.exists('results'):
        os.makedirs('results')

    result_2019 = run_project_svm("2019 VERİ SETİ ANALİZİ", DATA_PATH_2019, "2019")
    result_2020 = run_project_svm("2020 VERİ SETİ ANALİZİ", DATA_PATH_2020, "2020")

    full_feature_count_2019 = len(result_2019['feature_names'])
    full_feature_count_2020 = len(result_2020['feature_names'])
    projection_features = result_2020['feature_names'] if full_feature_count_2020 >= full_feature_count_2019 else result_2019['feature_names']

    combined_model, combined_scaler = train_combined_projection_model(projection_features)

    X_full_2019, _, _ = load_full_feature_matrix(
        DATA_PATH_2019,
        projection_features,
        combined_scaler,
    )
    X_full_2020, _, _ = load_full_feature_matrix(
        DATA_PATH_2020,
        projection_features,
        combined_scaler,
    )
    pred_2019 = combined_model.predict_with_threshold(X_full_2019)['y_pred']
    pred_2020 = combined_model.predict_with_threshold(X_full_2020)['y_pred']
    avg_risk_rate = (pred_2019.mean() + pred_2020.mean()) / 2.0
    projected_failures_2021 = int(round(15873 * avg_risk_rate))
    print_projection_summary(projected_failures_2021, paper_reference=852)

    print("\n  [Başarılı] Proje için supervised SVM akışı hazırlandı.")
    print("  [Not] k-means akıştan çıkarıldı; 2021 projeksiyonu 2019+2020 birleşik modelinden üretildi.\n")


if __name__ == "__main__":
    main()
