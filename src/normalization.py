import pandas as pd
from sklearn.preprocessing import StandardScaler

df_2019 = pd.read_csv("/home/emirfurkan/Desktop/transformer_predictive_maintenance/data/processed/2019_pseudo.csv")
df_2020 = pd.read_csv("/home/emirfurkan/Desktop/transformer_predictive_maintenance/data/processed/2020_pseudo.csv")

numeric_features = ["power" , "average_earth_discharge_density_ddt_rayskm2ao" ,
                    "burning_rate__failuresyear" , "number_of_users" , "km_of_network_lt" ,
                    "power_per_user" , "lightning_risk_score" , "network_density" ,
                    "historical_risk_index"]

scaler = StandardScaler()

df_2019[numeric_features] = scaler.fit_transform(df_2019[numeric_features])
df_2020[numeric_features] = scaler.transform(df_2020[numeric_features])

if "Unnamed: 0" in df_2019.columns :
    df_2019 = df_2019.drop(columns = ["Unnamed: 0"])

if "Unnamed: 0" in df_2020.columns :
    df_2020 = df_2020.drop(columns = ["Unnamed: 0"])

df_2019 = df_2019.astype({col: int for col in df_2019.select_dtypes('bool').columns})
df_2020 = df_2020.astype({col: int for col in df_2020.select_dtypes('bool').columns})



target_dir = "/home/emirfurkan/Desktop/transformer_predictive_maintenance/data/processed/"

df_2019.to_csv(target_dir + "normalized_dataset_2019.csv" , index=False)
df_2020.to_csv(target_dir + "normalized_dataset_2020.csv" , index=False)
print("Pseudo dataset normalizasyonu tamamlandı")
