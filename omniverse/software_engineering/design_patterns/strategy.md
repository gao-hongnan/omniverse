---
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Strategy

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Organized_Chaos-orange)
[![Code](https://img.shields.io/badge/View-Code-blue?style=flat-square&logo=github)](https://github.com/gao-hongnan/omniverse/tree/5123256f3872d0fed1d2c69bf78b13f27314dad5/omnixamples/software_engineering/design_patterns/strategy)

```{contents}
```

We quote verbatim from the
[Design Patterns book](https://refactoring.guru/design-patterns/strategy):

```{epigraph}
The
Strategy pattern suggests that you take a class that does something specific in
a lot of different ways and extract all of these algorithms into separate
classes called strategies.

The original class, called context, must have a field for storing a reference to
one of the strategies. The context delegates the work to a linked strategy
object instead of executing it on its own.

The context isn’t responsible for selecting an appropriate algorithm for the
job. Instead, the client passes the desired strategy to the context. In fact,
the context doesn’t know much about strategies. It works with all strategies
through the same generic interface, which only exposes a single method for
triggering the algorithm encapsulated within the selected strategy.

This way the context becomes independent of concrete strategies, so you can add
new algorithms or modify existing ones without changing the code of the context
or other strategies.

-- [Design Patterns book](https://refactoring.guru/design-patterns/strategy)
```

## The Compliant And The Violation

````{tab} Compliant
```python

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

    # Initialize with Linear Regression
    trainer = Trainer(strategy=get_training_strategy(ModelType.LINEAR_REGRESSION))
    print("Training with Linear Regression:")
    trainer.train(X=X, y=y)

    # Swap to Decision Tree without changing the trainer object
    trainer.strategy = get_training_strategy(ModelType.DECISION_TREE)
    print("\nSwapped to Decision Tree:")
    trainer.train(X=X, y=y)

    # Swap to KNN without changing the trainer object
    trainer.strategy = get_training_strategy(ModelType.K_NEAREST_NEIGHBORS)
    print("\nSwapped to KNN:")
    trainer.train(X=X, y=y)

    # Swap to Gradient Boosting without changing the trainer object
    trainer.strategy = get_training_strategy(ModelType.GRADIENT_BOOSTING)
    print("\nSwapped to Gradient Boosting:")
    trainer.train(X=X, y=y)

    class RandomForest(TrainingStrategy):
        def train(self, X: List[List[float]], y: List[float]) -> None:
            print("Training Random Forest model")
            print(f"X: {X}, y: {y}")

    trainer.strategy = RandomForest()
    print("\nSwapped to Random Forest:")
    trainer.train(X=X, y=y)

    # NOTE: now you also can add this to the factory function easily, just
    # change 1 place.
```
````

````{tab} Violation
```python
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
        trainer.model_type = "RANDOM_FOREST"
        trainer.fit(X=X, y=y)
    except Exception as exc:
        print(f"Failed to add new model type without modifying Trainer class: {exc}")
```
````

````{tab} Complex Strategy With Factory And Registry Pattern
```python
from __future__ import annotations

from enum import Enum, auto
from typing import Dict, List, Protocol, Type


class TrainingStrategy(Protocol):
    def train(self, X: List[List[float]], y: List[float]) -> None:
        ...


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
    def register(cls: Type[StrategyRegistry], model_type: ModelType) -> callable:
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
```
````

## The Context, Strategy, Concrete Strategy, Business Logic, Client, And Factory

1. **Context** Object in our example is the `Trainer` class. It maintains a
   reference to a strategy object and uses the strategy object to perform the
   training.
2. **Strategy** Object in our example is the `TrainingStrategy` class. It
   implements the `train` method and serves as the abstract interface for all
   concrete strategies.
3. **Concrete Strategy** Object in our example is the
   `LinearRegressionStrategy`, `DecisionTreeStrategy`, `KNNStrategy`,
   `GradientBoostingStrategy`, and `NeuralNetworkStrategy` classes. They
   implement the `train` method.
4. **Business Logic** is the `train` method in the `Trainer` class.
5. **Client** is the `__main__` block. It creates the context object and the
   strategy objects, and then uses the context object to perform the training.
6. **Factory** is the `get_training_strategy` function. It creates the strategy
   objects and serves a very simple factory design for the client.

## Pros and Cons

Again, we can find the cons easily, such as the client needs to know all the
concrete strategies. Also, this strategy pattern may really be over-complicated
for simple tasks. You can take a look at the `violation.py` code in my
[code repository](https://github.com/gao-hongnan/omniverse/tree/5123256f3872d0fed1d2c69bf78b13f27314dad5/omnixamples/software_engineering/design_patterns/strategy)
for a violation of this pattern and realize that the violation is not "that
bad".

For the pros, we quote again:

```{epigraph}
-   You can swap algorithms used inside an object at runtime.
-   You can isolate the implementation details of an algorithm from the code that uses it.
-   You can replace inheritance with composition.
-   Open/Closed Principle. You can introduce new strategies without having to change the context.

-- [Design Patterns book](https://refactoring.guru/design-patterns/strategy)
```

It is fairly easy to see.

1. You can swap algorithms used inside an object at runtime. This can be seen
   from our `setter`, we can change the strategy at runtime as and when we wish.
   Though in python it is easily doable so I do not see a super strong point
   here.
2. Isolate the implementation details of an algorithm from the code that uses
   it. This is clear from abstracting the strategies from the base interface. In
   general, it is a clean design to do so.
3. You can replace inheritance with composition. This is clear from the
   `Trainer` class, we are not inheriting from the strategies, but rather we are
   composing the strategies in the `Trainer` class.

    ```python
    class Trainer:
        def __init__(self, strategy: TrainingStrategy) -> None:
            self._strategy = strategy
    ```

4. Open/Closed Principle. You can introduce new strategies without having to
   change the context. This is clear from the `get_training_strategy` function,
   we can easily add a new strategy by adding a new case to the `ModelType` enum
   and a new strategy class.

## Complex Strategy With Factory And Registry Pattern

You can see from the above 3rd tab we can easily combine the factory pattern and
the registry pattern to create a more complex strategy. The registry allows
_automatic_ and _dynamic_ strategy creation where user does not need to add
"if-else" logic to create a strategy.

## References And Further Readings

-   [Strategy Design Pattern - Refactoring.Guru](https://refactoring.guru/design-patterns/strategy)
-   [Strategy Design Pattern - ArjanCodes](https://github.com/ArjanCodes/betterpython/tree/main/3%20-%20strategy%20pattern)
