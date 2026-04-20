
import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def _normalize_col_name(col: str) -> str:
    """Normalize Excel/CSV column names so mapping is consistent."""
    return re.sub(r'\s+', ' ', str(col)).strip()


RAW_COLUMN_MAP = {
    'LOCATION': 'LOCATION',
    'POWER': 'POWER',
    'SELF-PROTECTION': 'SELF_PROTECTION',
    # 'ao' veya 'año' karmaşasını önlemek için Excel'deki tam karşılığıyla güncelliyoruz:
    'Average earth discharge density DDT [Rays/km^2-año]': 'AVG_DISCHARGE',
    'Maximum ground discharge density DDT [Rays/km^2-año]': 'MAX_DISCHARGE',
    'Average earth discharge density DDT [Rays/km^2-ao]': 'AVG_DISCHARGE', # Alternatif encoding
    'Maximum ground discharge density DDT [Rays/km^2-ao]': 'MAX_DISCHARGE',  # Alternatif encoding
    'Burning rate  [Failures/year]': 'BURNING_RATE',
    'Criticality according to previous study for ceramics level': 'CRITICALITY',
    'Removable connectors': 'REMOVABLE_CONNECTORS',
    'Type of clients': 'CLIENT_TYPE',
    'Number of users': 'NUM_USERS',
    'Electric power not supplied EENS [kWh] ': 'ENERGY_NOT_SUPPLIED',
    'Type of installation': 'INSTALL_TYPE',
    'Air network': 'AIR_NETWORK',
    'Circuit Queue': 'CIRCUIT_QUEUE',
    'km of network LT:': 'NETWORK_LENGTH',
    'Burned transformers 2019': 'TARGET_2019',
    "Burned transformers 2020": "TARGET_2020",
}

COLUMN_MAP = {_normalize_col_name(k): v for k, v in RAW_COLUMN_MAP.items()}

BASE_FEATURES = [
    'LOCATION', 'POWER', 'SELF_PROTECTION', 'AVG_DISCHARGE', 'MAX_DISCHARGE',
    'BURNING_RATE', 'CRITICALITY', 'REMOVABLE_CONNECTORS', 'NUM_USERS',
    'ENERGY_NOT_SUPPLIED', 'AIR_NETWORK', 'CIRCUIT_QUEUE', 'NETWORK_LENGTH',
]

ENGINEERED_FEATURES = [
    'ENERGY_PER_USER', 'LIGHTNING_RISK', 'NETWORK_PER_POWER',
    'DISCHARGE_RANGE', 'IS_RESIDENTIAL', 'IS_POLE', 'IS_MACRO', 'LOW_POWER',
    'POWER_LIGHTNING', 'NETWORK_RISK',
]

PROJECT_FEATURES = [
    'LOCATION',
    'POWER',
    'SELF_PROTECTION',
    'AVG_DISCHARGE',
    'MAX_DISCHARGE',
    'BURNING_RATE',
    'CRITICALITY',
    'REMOVABLE_CONNECTORS',
    'NUM_USERS',
    'ENERGY_NOT_SUPPLIED',
    'AIR_NETWORK',
    'CIRCUIT_QUEUE',
    'NETWORK_LENGTH',
    'IS_RESIDENTIAL',
    'IS_POLE',
]


def _validate_required_columns(df: pd.DataFrame) -> None:
    required_columns = {
        'POWER', 'SELF_PROTECTION', 'AVG_DISCHARGE', 'MAX_DISCHARGE',
        'BURNING_RATE', 'CRITICALITY', 'CLIENT_TYPE', 'NUM_USERS',
        'ENERGY_NOT_SUPPLIED', 'INSTALL_TYPE', 'NETWORK_LENGTH',
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _resolve_target_column(path: str, df: pd.DataFrame) -> str:
    filename = os.path.basename(path).lower()
    if '2019' in filename and 'TARGET_2019' in df.columns:
        return 'TARGET_2019'
    if '2020' in filename and 'TARGET_2020' in df.columns:
        return 'TARGET_2020'

    available_targets = [col for col in ('TARGET_2019', 'TARGET_2020') if col in df.columns]
    if len(available_targets) == 1:
        return available_targets[0]

    raise KeyError("Dataset target column could not be resolved from file name or columns.")


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ENERGY_PER_USER'] = df['ENERGY_NOT_SUPPLIED'] / (df['NUM_USERS'] + 1)
    df['LIGHTNING_RISK'] = df['AVG_DISCHARGE'] * df['CRITICALITY']
    df['NETWORK_PER_POWER'] = df['NETWORK_LENGTH'] / (df['POWER'] + 1)
    df['PROTECTION_RISK'] = (1 - df['SELF_PROTECTION']) * df['BURNING_RATE']
    df['DISCHARGE_RANGE'] = df['MAX_DISCHARGE'] - df['AVG_DISCHARGE']
    df['IS_RESIDENTIAL'] = df['CLIENT_TYPE'].apply(lambda x: 1 if 'STRATUM' in str(x).upper() else 0)
    df['IS_POLE'] = (df['INSTALL_TYPE'] == 'POLE').astype(int)
    df['IS_MACRO'] = df['INSTALL_TYPE'].astype(str).str.contains('MACRO', na=False).astype(int)
    df['LOW_POWER'] = (df['POWER'] <= 15).astype(int)
    df['POWER_LIGHTNING'] = df['POWER'] * df['AVG_DISCHARGE']
    df['NETWORK_RISK'] = df['NETWORK_LENGTH'] * df['BURNING_RATE']
    return df


def preprocess_transformer_data(
        path: str,
        test_size: float | int = 0.2,
        random_state: int = 42,
        article_mode: bool = False,
):
    if path.endswith('.xlsx'):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding='ISO-8859-1')

    df.columns = [_normalize_col_name(c) for c in df.columns]
    df.rename(columns=COLUMN_MAP, inplace=True)
    _validate_required_columns(df)
    target_column = _resolve_target_column(path, df)
    df = _add_engineered_features(df)
    df['TARGET'] = df[target_column].astype(int)

    all_features = PROJECT_FEATURES if article_mode else (BASE_FEATURES + ENGINEERED_FEATURES)
    leak_columns = [col for col in df.columns if col.startswith('TARGET_')]
    df = df.drop(columns=leak_columns, errors='ignore')
    X = df[all_features]
    y = df['TARGET']
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_train_raw = X_train.reset_index(drop=True).copy()
    X_test_raw = X_test.reset_index(drop=True).copy()
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"[Preprocessing] Train : {X_train_scaled.shape}")
    print(f"[Preprocessing] Test  : {X_test_scaled.shape}")
    print(
        f"[Preprocessing] Class distribution -> "
        f"Train: {np.bincount(np.asarray(y_train, dtype=int))} | "
        f"Test: {np.bincount(np.asarray(y_test, dtype=int))}"
    )
    print(f"[Preprocessing] Kullanılan hedef sütunu: {target_column}")
    print(f"[Preprocessing] Özellik sütun sayısı: {len(all_features)} (burned transformers hariç)")
    if article_mode:
        print("[Preprocessing] Bölme yöntemi: Proje split'i (train=14873, test=1000, stratified)")
    else:
        print("[Preprocessing] Bölme yöntemi: Stratified Train/Test Split")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, all_features, X_train_raw, X_test_raw


def load_full_feature_matrix(
    path: str,
    feature_names: list[str],
    scaler,
):
    if path.endswith('.xlsx'):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding='ISO-8859-1')

    df.columns = [_normalize_col_name(c) for c in df.columns]
    df.rename(columns=COLUMN_MAP, inplace=True)
    _validate_required_columns(df)
    target_column = _resolve_target_column(path, df)
    df = _add_engineered_features(df)
    df['TARGET'] = df[target_column].astype(int)

    X_full_raw = df[feature_names].reset_index(drop=True)
    y_full = df['TARGET'].reset_index(drop=True)
    X_full_scaled = X_full_raw.to_numpy() if scaler is None else scaler.transform(X_full_raw)
    return X_full_scaled, y_full, X_full_raw
