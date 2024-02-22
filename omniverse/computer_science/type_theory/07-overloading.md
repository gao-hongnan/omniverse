## Overload

### Example BaseEstimator

Use

```
class Unsupervised:
    def __repr__(self):
        return "Unsupervised()"

UNSUPERVISED = Unsupervised()

class BaseEstimator(ABC):
    @overload
    def fit(self, X: T, y: T) -> BaseEstimator:
        """Overload for supervised learning."""

    @overload
    def fit(self, X: T, y: Unsupervised = UNSUPERVISED) -> BaseEstimator:
        """Overload for unsupervised learning."""

    @abstractmethod
    def fit(self, X: T, y: Union[T, Unsupervised] = UNSUPERVISED) -> BaseEstimator:
        """
        Fit the model according to the given training data.

        For supervised learning, y should be the target data.
        For unsupervised learning, y should be Unsupervised.
        """
        pass

# Example subclass
class MyEstimator(BaseEstimator):
    def fit(self, X: T, y: Union[T, Unsupervised] = UNSUPERVISED) -> BaseEstimator:
        if y is UNSUPERVISED:
            # Unsupervised learning logic
            ...
        else:
            # Supervised learning logic
            ...
        return self
```

### Overload and Coercion Relationship

```
a: int   = 2
b: float = 3.0
c = a + b
```

## Add

implement `__add__` as overload variant.
