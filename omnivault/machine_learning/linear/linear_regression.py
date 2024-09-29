r"""Linear Regression Class.

High-level module implementing linear regression with support for multiple
solvers and regularization.

Features:
    - Closed Form Solution
    - Batch Gradient Descent
    - Stochastic Gradient Descent

Supports L1 and L2 regularization.

NOTE:
    1. Regularization parameter `C` is only applicable when `regularization` is set.
    2. Ensure that input data does not contain NaN or infinite values.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, List, Optional, TypeVar

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import NotFittedError
from typing_extensions import Concatenate, ParamSpec

from omnivault.machine_learning.estimator import BaseEstimator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


P = ParamSpec("P")
T = TypeVar("T")


def not_fitted(func: Callable[Concatenate[LinearRegression, P], T]) -> Callable[Concatenate[LinearRegression, P], T]:
    @functools.wraps(func)
    def wrapper(self: LinearRegression, *args: P.args, **kwargs: P.kwargs) -> T:
        if not self._fitted:
            raise NotFittedError
        else:
            return func(self, *args, **kwargs)

    return wrapper


class LinearRegression(BaseEstimator):
    """
    Linear Regression model supporting multiple solvers and regularization.

    Attributes
    ----------
    coef_ : NDArray[np.floating[Any]]
        Coefficient vector.
    intercept_ : float
        Intercept term.
    has_intercept : bool
        Whether to include an intercept in the model.
    solver : str
        Solver type: {"Closed Form Solution", "Batch Gradient Descent",
        "Stochastic Gradient Descent"}.
    learning_rate : float
        Learning rate for gradient descent solvers.
    loss_function : LossFunction
        Loss function to be minimized.
    regularization : Optional[str]
        Type of regularization: {"l1", "l2"} or None.
    C : Optional[float]
        Regularization strength. Must be a positive float.
    num_epochs : int
        Number of epochs for gradient descent solvers.
    _fitted : bool
        Flag indicating whether the model has been fitted.
    loss_history : List[float]
        History of loss values during training.
    optimal_betas : NDArray[np.floating[Any]]
        Optimal beta coefficients after fitting.
    """

    def __init__(
        self,
        has_intercept: bool = True,
        solver: str = "Closed Form Solution",
        learning_rate: float = 0.1,
        loss_function: Optional[Any] = None,
        regularization: Optional[str] = None,
        C: Optional[float] = None,
        num_epochs: Optional[int] = 1000,
    ) -> None:
        """
        Initialize the Linear Regression model with specified parameters.

        Parameters
        ----------
        has_intercept : bool, default=True
            Whether to include an intercept in the model.
        solver : str, default="Closed Form Solution"
            Solver type: {"Closed Form Solution", "Batch Gradient Descent", "Stochastic Gradient Descent"}.
        learning_rate : float, default=0.01
            Learning rate for gradient descent solvers.
        loss_function : LossFunction, default=LossFunction.l2_loss()
            Loss function to be minimized.
        regularization : Optional[str], default=None
            Type of regularization: {"l1", "l2"} or None.
        C : Optional[float], default=None
            Regularization strength. Must be a positive float.
        num_epochs : int, default=1000
            Number of epochs for gradient descent solvers.
        """
        self.solver = solver
        self.has_intercept = has_intercept
        self.learning_rate = learning_rate
        self.loss_function = loss_function or self.l2_loss
        self.regularization = regularization
        self.C = C
        self.num_epochs = num_epochs

        self.coef_: NDArray[np.floating[Any]]
        self.intercept_: float | None = None
        self._fitted: bool = False
        self.optimal_betas: NDArray[np.floating[Any]]

        self.loss_history: List[float] = []

        # Validate regularization parameters
        if self.regularization is not None and self.C is None:
            raise ValueError("Regularization strength 'C' must be provided when using regularization.")

        if self.regularization not in {None, "l1", "l2"}:
            raise ValueError("Regularization must be one of {None, 'l1', 'l2'}.")

        # Validate solver
        valid_solvers = {"Closed Form Solution", "Batch Gradient Descent", "Stochastic Gradient Descent"}
        if self.solver not in valid_solvers:
            raise ValueError(f"Solver must be one of {valid_solvers}.")

    @staticmethod
    def l2_loss(y_true: NDArray[np.floating[Any]], y_pred: NDArray[np.floating[Any]]) -> float:
        return np.square(y_true - y_pred).mean()  # type: ignore[no-any-return]

    def _add_regularization(self, loss: float, w: NDArray[np.floating[Any]]) -> float:
        """
        Apply regularization to the loss.

        Parameters
        ----------
        loss : float
            Current loss value.
        w : NDArray[np.floating[Any]]
            Weight vector.

        Returns
        -------
        float
            Updated loss with regularization.
        """
        if self.regularization == "l1":
            loss += self.C * np.abs(w[1:]).sum()
        elif self.regularization == "l2":
            assert self.C is not None
            loss += (0.5 * self.C) * np.square(w[1:]).sum()
        return loss

    def _initialize_weights(self, n_features: int) -> NDArray[np.floating[Any]]:
        """
        Initialize weights for gradient descent solvers.

        Parameters
        ----------
        n_features : int
            Number of features.

        Returns
        -------
        NDArray[np.floating[Any]]
            Initialized weight vector.
        """
        return np.zeros((n_features, 1))

    def _check_input_shapes(
        self, X: NDArray[np.floating[Any]], y_true: NDArray[np.floating[Any]]
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """
        Validate and reshape input data.

        Parameters
        ----------
        X : NDArray[np.floating[Any]]
            Input feature matrix.
        y_true : NDArray[np.floating[Any]]
            True target values.

        Returns
        -------
        tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
            Reshaped feature matrix and target vector.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            logging.info("Reshaped X to 2D array with shape %s.", X.shape)

        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
            logging.info("Reshaped y_true to 2D array with shape %s.", y_true.shape)

        if X.shape[0] != y_true.shape[0]:
            raise ValueError("Number of samples in X and y_true must be equal.")

        return X, y_true

    def fit(self, X: NDArray[np.floating[Any]], y_true: NDArray[np.floating[Any]]) -> LinearRegression:
        """
        Fit the Linear Regression model to the data.

        Parameters
        ----------
        X : NDArray[np.floating[Any]]
            Input feature matrix of shape (n_samples, n_features).
        y_true : NDArray[np.floating[Any]]
            True target values of shape (n_samples,).

        Returns
        -------
        LinearRegression
            The fitted model.
        """
        X, y_true = self._check_input_shapes(X, y_true)

        # Add intercept term if necessary
        if self.has_intercept:
            X = np.insert(X, 0, 1, axis=1)
            logging.info("Added intercept term to X.")

        n_samples, n_features = X.shape
        logging.info("Fitting model with %d samples and %d features.", n_samples, n_features)

        if self.solver == "Closed Form Solution":
            XtX = X.T @ X
            det = np.linalg.det(XtX)
            if det == 0:
                logging.warning("Singular matrix encountered. Using pseudo-inverse instead.")
                XtX_inv = np.linalg.pinv(XtX)
            else:
                XtX_inv = np.linalg.inv(XtX)
            Xty = X.T @ y_true
            self.optimal_betas = XtX_inv @ Xty
            logging.info("Computed optimal betas using Closed Form Solution.")

        elif self.solver in {"Batch Gradient Descent", "Stochastic Gradient Descent"}:
            assert self.num_epochs is not None
            self.optimal_betas = self._initialize_weights(n_features)
            logging.info("Initialized weights for Gradient Descent.")

            for epoch in range(1, self.num_epochs + 1):
                if self.solver == "Batch Gradient Descent":
                    y_pred = X @ self.optimal_betas
                elif self.solver == "Stochastic Gradient Descent":
                    indices = np.random.permutation(n_samples)  # noqa: NPY002
                    for i in indices:
                        xi = X[i].reshape(1, -1)
                        yi = y_true[i]
                        y_pred = xi @ self.optimal_betas
                        error = y_pred - yi
                        gradient = xi.T @ error
                        if self.regularization == "l2":
                            assert self.C is not None
                            gradient[1:] += self.C * self.optimal_betas[1:]
                        elif self.regularization == "l1":
                            gradient[1:] += self.C * np.sign(self.optimal_betas[1:])
                        self.optimal_betas -= self.learning_rate * gradient
                    continue  # Skip the rest of the loop for SGD

                error = y_pred - y_true
                loss = self.loss_function(y_true=y_true, y_pred=y_pred)
                loss = self._add_regularization(loss, self.optimal_betas)
                self.loss_history.append(loss)

                gradient = (2 / n_samples) * (X.T @ error)
                if self.regularization == "l2":
                    assert self.C is not None
                    gradient[1:] += self.C * self.optimal_betas[1:]
                elif self.regularization == "l1":
                    gradient[1:] += self.C * np.sign(self.optimal_betas[1:])

                self.optimal_betas -= self.learning_rate * gradient

                if epoch % 100 == 0 or epoch == 1:
                    logging.info("Epoch %d | Loss: %.4f", epoch, loss)

        # Set coefficients and intercept
        if self.has_intercept:
            self.intercept_ = float(self.optimal_betas[0])
            self.coef_ = self.optimal_betas[1:].flatten()
            logging.info("Set intercept and coefficients.")
        else:
            self.coef_ = self.optimal_betas.flatten()
            self.intercept_ = 0.0
            logging.info("Set coefficients without intercept.")

        self._fitted = True
        logging.info("Model fitting complete.")

        return self

    @not_fitted
    def predict(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """
        Predict target values using the fitted model.

        Parameters
        ----------
        X : NDArray[np.floating[Any]]
            Input feature matrix of shape (n_samples, n_features).

        Returns
        -------
        NDArray[np.floating[Any]]
            Predicted target values of shape (n_samples,).
        """
        if self.has_intercept:
            X = np.insert(X, 0, 1, axis=1)

        assert self.optimal_betas is not None
        y_pred = X @ self.optimal_betas
        return y_pred.flatten()

    def residuals(self, X: NDArray[np.floating[Any]], y_true: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """
        Calculate residuals between true and predicted values.

        Parameters
        ----------
        X : NDArray[np.floating[Any]]
            Input feature matrix.
        y_true : NDArray[np.floating[Any]]
            True target values.

        Returns
        -------
        NDArray[np.floating[Any]]
            Residuals of shape (n_samples,).
        """
        y_pred = self.predict(X)
        residuals = y_true.flatten() - y_pred
        return residuals

    @not_fitted
    def score(self, X: NDArray[np.floating[Any]], y_true: NDArray[np.floating[Any]]) -> float:
        """
        Calculate the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : NDArray[np.floating[Any]]
            Input feature matrix.
        y_true : NDArray[np.floating[Any]]
            True target values.

        Returns
        -------
        float
            R^2 score.
        """
        y_pred = self.predict(X)
        ss_total = np.sum((y_true.flatten() - np.mean(y_true)) ** 2)
        ss_res = np.sum((y_true.flatten() - y_pred) ** 2)
        r2_score = 1 - (ss_res / ss_total)
        return float(r2_score)
