from __future__ import annotations

from enum import Enum, auto
from typing import Callable, Dict, List, Protocol, Type


class TrainingStrategy(Protocol):
    def train(self, X: List[List[float]], y: List[float]) -> None: ...


class ModelType(Enum):
    LINEAR_REGRESSION = auto()
    DECISION_TREE = auto()
    K_NEAREST_NEIGHBORS = auto()
    GRADIENT_BOOSTING = auto()
    NEURAL_NETWORK = auto()
    RANDOM_FOREST = auto()


class StrategyRegistry:
    _strategies: Dict[ModelType, Type[TrainingStrategy]] = {}

    @classmethod
    def register(
        cls: Type[StrategyRegistry], model_type: ModelType
    ) -> Callable[[Type[TrainingStrategy]], Type[TrainingStrategy]]:
        def decorator(strategy_class: Type[TrainingStrategy]) -> Type[TrainingStrategy]:
            cls._strategies[model_type] = strategy_class
            return strategy_class

        return decorator

    @classmethod
    def get_strategy(cls: Type[StrategyRegistry], model_type: ModelType) -> TrainingStrategy:
        strategy_class = cls._strategies.get(model_type)
        if not strategy_class:
            raise ValueError(f"No strategy registered for {model_type}")
        return strategy_class()


@StrategyRegistry.register(ModelType.LINEAR_REGRESSION)
class LinearRegressionStrategy:
    def train(self, X: List[List[float]], y: List[float]) -> None:
        print("Training linear regression model")
        print(f"X: {X}, y: {y}")


@StrategyRegistry.register(ModelType.DECISION_TREE)
class DecisionTreeStrategy:
    def train(self, X: List[List[float]], y: List[float]) -> None:
        print("Training decision tree model")
        print(f"X: {X}, y: {y}")


@StrategyRegistry.register(ModelType.K_NEAREST_NEIGHBORS)
class KNNStrategy(TrainingStrategy):
    def train(self, X: List[List[float]], y: List[float]) -> None:
        print("Training KNN model")
        print(f"X: {X}, y: {y}")


@StrategyRegistry.register(ModelType.GRADIENT_BOOSTING)
class GradientBoostingStrategy(TrainingStrategy):
    def train(self, X: List[List[float]], y: List[float]) -> None:
        print("Training gradient boosting model")
        print(f"X: {X}, y: {y}")


@StrategyRegistry.register(ModelType.NEURAL_NETWORK)
class NeuralNetworkStrategy(TrainingStrategy):
    def train(self, X: List[List[float]], y: List[float]) -> None:
        print("Training neural network model")
        print(f"X: {X}, y: {y}")


class Trainer:
    def __init__(self, strategy: TrainingStrategy) -> None:
        self._strategy = strategy

    @property
    def strategy(self) -> TrainingStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: TrainingStrategy) -> None:
        self._strategy = strategy

    def train(self, X: List[List[float]], y: List[float]) -> None:
        self._strategy.train(X, y)


if __name__ == "__main__":
    X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    y = [0.0, 1.0, 0.0]

    # Initialize with Linear Regression
    trainer = Trainer(strategy=StrategyRegistry.get_strategy(ModelType.LINEAR_REGRESSION))
    print("Training with Linear Regression:")
    trainer.train(X=X, y=y)

    # Swap to Decision Tree without changing the trainer object
    trainer.strategy = StrategyRegistry.get_strategy(ModelType.DECISION_TREE)
    print("\nSwapped to Decision Tree:")
    trainer.train(X=X, y=y)

    # Swap to KNN without changing the trainer object
    trainer.strategy = StrategyRegistry.get_strategy(ModelType.K_NEAREST_NEIGHBORS)
    print("\nSwapped to KNN:")
    trainer.train(X=X, y=y)

    # Swap to Gradient Boosting without changing the trainer object
    trainer.strategy = StrategyRegistry.get_strategy(ModelType.GRADIENT_BOOSTING)
    print("\nSwapped to Gradient Boosting:")
    trainer.train(X=X, y=y)

    # Dynamically add a new strategy
    @StrategyRegistry.register(ModelType.RANDOM_FOREST)
    class RandomForestStrategy:
        def train(self, X: List[List[float]], y: List[float]) -> None:
            print("Training Random Forest model")
            print(f"X: {X}, y: {y}")

    trainer.strategy = StrategyRegistry.get_strategy(ModelType.RANDOM_FOREST)
    print("\nSwapped to Random Forest:")
    trainer.train(X=X, y=y)
