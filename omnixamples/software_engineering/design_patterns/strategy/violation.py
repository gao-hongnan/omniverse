"""Shows a violation of the strategy pattern."""
from enum import Enum, auto
from typing import Any, List, Union


class ModelType(Enum):
    LINEAR_REGRESSION = auto()
    DECISION_TREE = auto()
    K_NEAREST_NEIGHBORS = auto()
    GRADIENT_BOOSTING = auto()
    NEURAL_NETWORK = auto()


class LinearRegression:
    def train(self, X: List[List[float]], y: List[float]) -> None:
        print("Training linear regression model")
        print(f"X: {X}, y: {y}")


class DecisionTree:
    def train(self, X: List[List[float]], y: List[float]) -> None:
        print("Training decision tree model")
        print(f"X: {X}, y: {y}")


class KNN:
    def train(self, X: List[List[float]], y: List[float]) -> None:
        print("Training KNN model")
        print(f"X: {X}, y: {y}")


class GradientBoosting:
    def train(self, X: List[List[float]], y: List[float]) -> None:
        print("Training gradient boosting model")
        print(f"X: {X}, y: {y}")


class NeuralNetwork:
    def train(self, X: List[List[float]], y: List[float]) -> None:
        print("Training neural network model")
        print(f"X: {X}, y: {y}")


class Trainer:
    def __init__(self, model_type: ModelType) -> None:
        self.model_type = model_type
        self.model: Union[
            LinearRegression,
            DecisionTree,
            KNN,
            GradientBoosting,
            NeuralNetwork,
        ]

    def fit(self, X: Any, y: Any) -> None:
        if self.model_type == ModelType.LINEAR_REGRESSION:
            self.model = LinearRegression()
        elif self.model_type == ModelType.DECISION_TREE:
            self.model = DecisionTree()
        elif self.model_type == ModelType.K_NEAREST_NEIGHBORS:
            self.model = KNN()
        elif self.model_type == ModelType.GRADIENT_BOOSTING:
            self.model = GradientBoosting()
        elif self.model_type == ModelType.NEURAL_NETWORK:
            self.model = NeuralNetwork()
        else:
            raise ValueError("Invalid model type")

        self.model.train(X, y)


if __name__ == "__main__":
    X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    y = [0.0, 1.0, 0.0]

    # Initialize with Linear Regression
    trainer = Trainer(model_type=ModelType.LINEAR_REGRESSION)
    trainer.fit(X=X, y=y)

    # Swap to Decision Tree???
    trainer.model_type = ModelType.DECISION_TREE
    trainer.fit(X=X, y=y)

    try:

        class RandomForest:
            def train(self, X: List[List[float]], y: List[float]) -> None:
                print("Training Random Forest model")
                print(f"X: {X}, y: {y}")

        # This will fail because RandomForest is not in the ModelType enum
        trainer.model_type = "RANDOM_FOREST"  # type: ignore[assignment]
        trainer.fit(X=X, y=y)
    except Exception as exc:
        print(f"Failed to add new model type without modifying Trainer class: {exc}")
