import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():

    df = pd.read_csv("stroke.csv").dropna()

    y = df["stroke"]
    X = df.drop("stroke", axis=1)

    X = pd.get_dummies(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    n = len(X_train)

    clients = [
        (X_train[:n//3], y_train[:n//3]),
        (X_train[n//3:2*n//3], y_train[n//3:2*n//3]),
        (X_train[2*n//3:], y_train[2*n//3:])
    ]

    return clients, X_test, y_test