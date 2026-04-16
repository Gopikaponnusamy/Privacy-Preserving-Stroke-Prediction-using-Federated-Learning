import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from utils import load_and_split


def federated_training(rounds=5):

    clients, X_test, y_test = load_and_split()

    acc_history = []

    for r in range(rounds):

        local_models = []

        for Xc, yc in clients:

            model = LogisticRegression(max_iter=300, class_weight="balanced")
            model.fit(Xc, yc)
            local_models.append(model)

        weights = [len(Xc) for Xc, _ in clients]
        total = sum(weights)

        global_model = LogisticRegression(max_iter=300)

        global_model.coef_ = np.sum([m.coef_ * w for m, w in zip(local_models, weights)], axis=0) / total
        global_model.intercept_ = np.sum([m.intercept_ * w for m, w in zip(local_models, weights)], axis=0) / total
        global_model.classes_ = np.unique(y_test)

        pred = global_model.predict(X_test)

        acc_history.append(accuracy_score(y_test, pred))

    final_pred = global_model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, final_pred),
        "precision": precision_score(y_test, final_pred),
        "recall": recall_score(y_test, final_pred),
        "f1": f1_score(y_test, final_pred),
        "conf_matrix": confusion_matrix(y_test, final_pred),
        "roc_auc": roc_auc_score(y_test, final_pred)
    }

    return acc_history, metrics