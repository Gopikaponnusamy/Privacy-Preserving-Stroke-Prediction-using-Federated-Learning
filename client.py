import flwr as fl
from sklearn.linear_model import LogisticRegression
from utils import load_data

# load dataset (each client uses full data for now simulation)
X_train, X_test, y_train, y_test = load_data()


class StrokeClient(fl.client.NumPyClient):

    def __init__(self):
        self.model = LogisticRegression(max_iter=200)

    def get_parameters(self, config):
        return [self.model.coef_, self.model.intercept_]

    def set_parameters(self, parameters):
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(X_train, y_train)
        return self.get_parameters(config), len(X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc = self.model.score(X_test, y_test)
        print("Client Accuracy:", acc)
        return 0.0, len(X_test), {"accuracy": acc}


# IMPORTANT: this line is missing in your code
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=StrokeClient()
)