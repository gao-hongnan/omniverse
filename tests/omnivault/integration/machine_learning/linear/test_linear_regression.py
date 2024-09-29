"""
Notice that the intercept and coefficient values are not exactly the same when
comparing sklearn's method and mine. This is because we are using slightly
different ways to solve the question.

My Normal Equation: solving the normal equations by directly inverting the
X.T @ X matrix.

Normal Equation SKLEARN: On the other hand, scikit-learn uses scipy.linalg.lstsq
under the hood, which uses for example an SVD-based approach. That is, the
mechanism there does not invert the matrix and is therefore different than
ours. Note that there are many ways to solve the linear least squares
problem.
"""
import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.model_selection import train_test_split
from omnivault.machine_learning.linear.linear_regression import LinearRegression as CustomLinearRegression


@pytest.mark.parametrize("n_features", [3, 5, 10])
@pytest.mark.parametrize("solver", ["Closed Form Solution", "Batch Gradient Descent"])
def test_linear_regression_vs_sklearn(n_features: int, solver: str) -> None:
    # Generate synthetic data
    X, y = make_regression(
        n_samples=1000,
        n_features=n_features,
        noise=0.1,
        random_state=1992,
    )
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1930)

    # Fit custom model
    lr_custom = CustomLinearRegression(
        solver=solver, has_intercept=True, num_epochs=10000 if solver == "Batch Gradient Descent" else None
    )
    lr_custom.fit(X_train, y_train)

    # Fit sklearn model
    lr_sklearn = SklearnLinearRegression(fit_intercept=True)
    lr_sklearn.fit(X_train, y_train)

    # Check if the parameters are close enough
    np.testing.assert_allclose(lr_custom.coef_, lr_sklearn.coef_, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(lr_custom.intercept_, lr_sklearn.intercept_, rtol=1e-2, atol=1e-2)

    # Check if predictions are close enough
    custom_pred = lr_custom.predict(X_val)
    sklearn_pred = lr_sklearn.predict(X_val)
    np.testing.assert_allclose(custom_pred, sklearn_pred, rtol=1e-2, atol=1e-2)

    # Check R-squared score
    custom_r2 = lr_custom.score(X_val, y_val)
    sklearn_r2 = lr_sklearn.score(X_val, y_val)
    np.testing.assert_allclose(custom_r2, sklearn_r2, rtol=1e-2, atol=1e-2)
