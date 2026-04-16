import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix")
    return fig


def plot_learning_curve(fl, central):
    fig, ax = plt.subplots()

    ax.plot(fl, marker="o", label="Federated")
    ax.plot(central, marker="o", label="Centralized")

    ax.set_title("FL vs Centralized Learning Curve")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Accuracy")
    ax.legend()

    return fig