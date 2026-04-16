from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from utils import load_data
import numpy as np

def train_centralized():

    clients, X_test, y_test = load_data()

    X_train = np.vstack([c[0] for c in clients])
    y_train = np.hstack([c[1] for c in clients])

    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    return accuracy_score(y_test, pred)