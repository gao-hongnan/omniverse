"""Demonstrates the strategy pattern."""
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List


class ModelType(Enum):
    LINEAR_REGRESSION = auto()
    DECISION_TREE = auto()
    K_NEAREST_NEIGHBORS = auto()
    GRADIENT_BOOSTING = auto()
    NEURAL_NETWORK = auto()


class TrainingStrategy(ABC):
    @abstractmethod
    def train(self, X: List[List[float]], y: List[float]) -> None:
        ...


class LinearRegressionStrategy(TrainingStrategy):
    def train(self, X: List[List[float]], y: List[float]) -> None:
        print("Training linear regression model")
        print(f"X: {X}, y: {y}")


class DecisionTreeStrategy(TrainingStrategy):
    def train(self, X: List[List[float]], y: List[float]) -> None:
        print("Training decision tree model")
        print(f"X: {X}, y: {y}")


class KNNStrategy(TrainingStrategy):
    def train(self, X: List[List[float]], y: List[float]) -> None:
        print("Training KNN model")
        print(f"X: {X}, y: {y}")


class GradientBoostingStrategy(TrainingStrategy):
    def train(self, X: List[List[float]], y: List[float]) -> None:
        print("Training gradient boosting model")
        print(f"X: {X}, y: {y}")


class NeuralNetworkStrategy(TrainingStrategy):
    def train(self, X: List[List[float]], y: List[float]) -> None:
        print("Training neural network model")
        print(f"X: {X}, y: {y}")


class Trainer:
    def __init__(self, strategy: TrainingStrategy):
        self.strategy = strategy

    def train(self, X: List[List[float]], y: List[float]) -> None:
        self.strategy.train(X, y)


def get_training_strategy(model_type: ModelType) -> TrainingStrategy:
    if model_type == ModelType.LINEAR_REGRESSION:
        return LinearRegressionStrategy()
    elif model_type == ModelType.DECISION_TREE:
        return DecisionTreeStrategy()
    elif model_type == ModelType.K_NEAREST_NEIGHBORS:
        return KNNStrategy()
    elif model_type == ModelType.GRADIENT_BOOSTING:
        return GradientBoostingStrategy()
    elif model_type == ModelType.NEURAL_NETWORK:
        return NeuralNetworkStrategy()
    else:
        raise ValueError("Invalid model type")


if __name__ == "__main__":
    X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    y = [0.0, 1.0, 0.0]

    model_type = ModelType.LINEAR_REGRESSION
    strategy = get_training_strategy(model_type)
    trainer = Trainer(strategy=strategy)
    trainer.train(X=X, y=y)
