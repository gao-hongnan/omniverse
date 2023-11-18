"""Shows a violation of the strategy pattern."""

from enum import Enum, auto


class ModelType(Enum):
    LINEAR_REGRESSION = auto()
    DECISION_TREE = auto()
    K_NEAREST_NEIGHBORS = auto()
    GRADIENT_BOOSTING = auto()
    NEURAL_NETWORK = auto()


class LinearRegression:
    def fit(self, X, y):
        print("Fitting linear regression model")


class DecisionTreeRegressor:
    def fit(self, X, y):
        print("Fitting decision tree model")


class KNeighborsRegressor:
    def fit(self, X, y):
        print("Fitting KNN model")


class GradientBoostingRegressor:
    def fit(self, X, y):
        print("Fitting gradient boosting model")


class NeuralNetworkRegressor:
    def fit(self, X, y):
        print("Fitting neural network model")


class Trainer:
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        self.model = None

    def fit(self, X, y):
        if self.model_type == ModelType.LINEAR_REGRESSION:
            self.model = LinearRegression()
        elif self.model_type == ModelType.DECISION_TREE:
            self.model = DecisionTreeRegressor()
        elif self.model_type == ModelType.K_NEAREST_NEIGHBORS:
            self.model = KNeighborsRegressor()
        elif self.model_type == ModelType.GRADIENT_BOOSTING:
            self.model = GradientBoostingRegressor()
        elif self.model_type == ModelType.NEURAL_NETWORK:
            self.model = NeuralNetworkRegressor()
        else:
            raise ValueError("Invalid model type")
        self.model.fit(X, y)


if __name__ == "__main__":
    trainer = Trainer(model_type=ModelType.LINEAR_REGRESSION)
    trainer.fit(X=None, y=None)
