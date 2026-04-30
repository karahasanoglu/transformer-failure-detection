import  pandas as pd
import re
import seaborn as sns
import os
import matplotlib.pyplot as plt

def clean_columns(cols):
    clean_cols = []
    for col in cols:
        col = col.lower()
        col = re.sub(r'[^a-z0-9\s]', '', col)

        col = col.strip().replace(' ', '_')
        col = re.sub(r'\s+', '_', col)

        clean_cols.append(col)

    return clean_cols

def apply_feature_engineering(df) :
    # Kullanıcı başına düşen ortalama güç
    df['power_per_user'] = df['power'] / (df['number_of_users'] + 1)

    # Yıldırım Risk Faktörü (Koruma 0 ise risk artar, 1 ise azalır)

    df['lightning_risk_score'] = df['average_earth_discharge_density_ddt_rayskm2ao'] * (2 - df['selfprotection'])

    #Şebeke Yoğunluğu
    df['network_density'] = df['km_of_network_lt'] / (df['power'] + 1)

    # Kritiklik ve Yanma Oranı Etkileşimi
    df['historical_risk_index'] = df['burning_rate__failuresyear'] * df[
        'criticality_according_to_previous_study_for_ceramics_level']

    return df

def plot_correlation_matrix(df, title:str):
    save_path = "/home/emirfurkan/Desktop/transformer_predictive_maintenance/results"
    corr = df.select_dtypes(include='number').corr()

    plt.figure(figsize=(12,12))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title(title)


    png_file = os.path.join(save_path, f'{title}.png')
    plt.savefig(png_file , bbox_inches='tight' , dpi=300)
    plt.close()

def target_correlation(df, target_col) :

    save_path = "/home/emirfurkan/Desktop/transformer_predictive_maintenance/results"
    corr = df.select_dtypes(include='number').corr()[target_col]

    corr = corr.drop(target_col).sort_values(ascending=False) # hedef özelliğin kendisiyle olan korelasyonunun çıkardık ve diğer ilişkileri sıraladım

    plt.figure(figsize=(8, 10))
    sns.heatmap(corr.to_frame(), annot=True, cmap='coolwarm', cbar=False)
    plt.title(f'Target ({target_col}) ile Değişkenlerin İlişkisi')

    png_file = os.path.join(save_path, f'{target_col}_correlation.png')
    plt.savefig(png_file , bbox_inches='tight' , dpi=300)
    plt.close()

def drop_misleading_target(df, target_col):
    if target_col in df.columns:
        return df.drop(columns=[target_col])
    return df



df_2019 = pd.read_excel("/home/emirfurkan/Desktop/transformer_predictive_maintenance/data/raw/Dataset_Year_2019.xlsx")
df_2020 = pd.read_excel("/home/emirfurkan/Desktop/transformer_predictive_maintenance/data/raw/Dataset_Year_2020.xlsx")

df_2019.columns = clean_columns(df_2019.columns)

#print("2019 veriseti sütün isimleri:\n",df_2019.columns)

df_2020.columns = clean_columns(df_2020.columns)

#print("\n\n2020 veriseti sütün isimleri:\n",df_2020.columns)

df_2019 = drop_misleading_target(df_2019, "burned_transformers_2019")
df_2020 = drop_misleading_target(df_2020, "burned_transformers_2020")

df_2019 = apply_feature_engineering(df_2019)
df_2020 = apply_feature_engineering(df_2020)

categorical_cols = ["type_of_clients" , "type_of_installation"]

df_2019 = pd.get_dummies(df_2019, columns=categorical_cols)
df_2020 = pd.get_dummies(df_2020, columns=categorical_cols)


plot_correlation_matrix(df_2019, title="2019 Correlation Matrix")


plot_correlation_matrix(df_2020 , title="2020 Correlation Matrix")

cols_to_drop = [
    "location" , "maximum_ground_discharge_density_ddt_rayskm2ao" ,
    "electric_power_not_supplied_eens_kwh" , "circuit_queue" ,
    "air_network"
]

df_2019 = df_2019.drop(columns=cols_to_drop)
df_2020 = df_2020.drop(columns=cols_to_drop)

df_2020 = df_2020.rename(columns={'type_of_installation_POLE WITH ANTI-FRAU NET': 'type_of_installation_POLE WITH ANTI-FRAUD NET'})

target_path = "/home/emirfurkan/Desktop/transformer_predictive_maintenance/data/processed/"


df_2019.to_csv(target_path + "processed_dataset_2019.csv", index=False)
df_2020.to_csv(target_path + "processed_dataset_2020.csv", index=False)
