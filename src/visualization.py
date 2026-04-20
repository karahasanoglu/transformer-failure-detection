
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA


def save_confusion_matrix(y_true, y_pred, title, filename):

    os.makedirs('results', exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))


    axis_labels = ['0 (Sağlam)', '1 (Arızalı)']


    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=axis_labels,
                yticklabels=axis_labels)

    plt.title(f'Karmaşıklık Matrisi: {title}')
    plt.ylabel('Gerçek Durum')
    plt.xlabel('Modelin Tahmini')


    plt.tight_layout()

    plt.savefig(f'results/{filename}.png')
    plt.close()
    print(f"[Görselleştirme] Matris kaydedildi: results/{filename}.png")


def save_cluster_plot(X, clusters, title, filename):

    os.makedirs('results', exist_ok=True)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Küme ID (0: Sağlam, 1: Riskli)')

    plt.title(f'k-Means Kümeleme: {title} (PCA İzdüşümü)')
    plt.xlabel('Temel Bileşen 1')
    plt.ylabel('Temel Bileşen 2')

    plt.tight_layout()
    plt.savefig(f'results/{filename}.png')
    plt.close()
    print(f"[Görselleştirme] Küme grafiği kaydedildi: results/{filename}.png")