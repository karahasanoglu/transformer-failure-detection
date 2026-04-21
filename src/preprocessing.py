import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
def process_data():
    tf = pd.read_csv("/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/data/Overview.csv")

    tf['DeviceTimeStamp'] = pd.to_datetime(tf['DeviceTimeStamp'])

    cv = pd.read_csv("/home/emirfurkan/Desktop/Distributed_Transformer_Monitoring/data/CurrentVoltage.csv")

    cv['DeviceTimeStamp'] = pd.to_datetime(cv['DeviceTimeStamp'])


    df = pd.merge(tf, cv, on='DeviceTimeStamp')

    X = df.drop(["DeviceTimeStamp" , "MOG_A"], axis=1)
    y = df["MOG_A"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test