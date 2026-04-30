import os

import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

df_2019 = pd.read_csv("/home/emirfurkan/Desktop/2019-2020-predictive-model/data/processed/normalized_dataset_2019.csv")
df_2020 = pd.read_csv("/home/emirfurkan/Desktop/2019-2020-predictive-model/data/processed/normalized_dataset_2020.csv")

output_dir = "/home/emirfurkan/Desktop/2019-2020-predictive-model/results"

df_sample = df_2019.sample(5000)
df_sample2 = df_2020.sample(5000)

tsne = TSNE(n_components=2 , random_state=42)

X_embedded = tsne.fit_transform(df_sample.drop(columns=['target']))

plt.scatter(X_embedded[:,0], X_embedded[:,1], c=df_sample2['target'], cmap='coolwarm', alpha=0.6)
plt.title(" 2019 Veri Kümesinin  Uzaydaki Dağılımı")
plt.savefig(os.path.join(output_dir, "tsne_2019.png"))



tsnne = TSNE(n_components=2 , random_state=42)

X_embedded = tsnne.fit_transform(df_sample2.drop(columns=['target']))

plt.scatter(X_embedded[:,0], X_embedded[:,1], c=df_sample['target'], cmap='coolwarm', alpha=0.6)
plt.title(" 2020 Veri Kümesinin  Uzaydaki Dağılımı")
plt.savefig(os.path.join(output_dir, "tsne_2020.png"))
