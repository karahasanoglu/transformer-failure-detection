import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

df_2019 = pd.read_csv("/home/emirfurkan/Desktop/transformer_predictive_maintenance/data/processed/processed_dataset_2019.csv")
df_2020 = pd.read_csv("/home/emirfurkan/Desktop/transformer_predictive_maintenance/data/processed/processed_dataset_2020.csv")

for df in (df_2019, df_2020):
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

df_2019 = df_2019.astype({col: int for col in df_2019.select_dtypes('bool').columns})
df_2020 = df_2020.astype({col: int for col in df_2020.select_dtypes('bool').columns})

X_2019 = df_2019.copy()
X_2020 = df_2020.copy()

scaler = StandardScaler()
X_2019_scaled = scaler.fit_transform(X_2019)
X_2020_scaled = scaler.transform(X_2020)


kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
c2019 = kmeans.fit_predict(X_2019_scaled)
c2020 = kmeans.predict(X_2020_scaled)

dist = kmeans.transform(X_2019_scaled).min(axis=1)
anom_cluster = pd.DataFrame({"c": c2019, "d": dist}).groupby("c")["d"].mean().idxmax()

p_km_2019 = (c2019 == anom_cluster).astype(int)
p_km_2020 = (c2020 == anom_cluster).astype(int)

iso = IsolationForest(contamination=0.05, random_state=42)
p_iso_2019 = (iso.fit_predict(X_2019_scaled) == -1).astype(int)
p_iso_2020 = (iso.predict(X_2020_scaled) == -1).astype(int)

df_2019 = X_2019.copy()
df_2020 = X_2020.copy()

df_2019["target"] = ((p_km_2019 + p_iso_2019) > 0).astype(int)
df_2020["target"] = ((p_km_2020 + p_iso_2020) > 0).astype(int)


print("\npseudo etiketli 2019 dataset dagilimi\n")
print(df_2019["target"].value_counts(normalize=True))

print("\npseudo etiketli 2020 dataset dagilimi\n")
print(df_2020["target"].value_counts(normalize=True))

df_2019.to_csv("/home/emirfurkan/Desktop/transformer_predictive_maintenance/data/processed/2019_pseudo.csv", index=False)
df_2020.to_csv("/home/emirfurkan/Desktop/transformer_predictive_maintenance/data/processed/2020_pseudo.csv", index=False)
