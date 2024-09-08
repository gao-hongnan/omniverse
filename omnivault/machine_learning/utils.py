import logging
from typing import Any, List, Optional, Tuple, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from numpy.typing import NDArray
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from omnivault.machine_learning.estimator import BaseEstimator
from omnivault.utils.probability_theory.plot import plot_contour, plot_scatter  # type: ignore[attr-defined]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TEstimator = TypeVar("TEstimator", bound=BaseEstimator)


def run_classifier(
    estimator: TEstimator,  # this type hint is the same name as scikit-learn
    X: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    test_size: float = 0.2,
    random_state: int = 1992,
    class_names: Optional[List[str]] = None,
) -> TEstimator:
    """Run a generic classifier on a dataset and returns the fitted classifier."""

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    estimator.fit(X_train, y_train)
    y_preds_train = estimator.predict(X_train)
    y_preds_test = estimator.predict(X_test)

    train_report = classification_report(
        y_train,
        y_preds_train,
        labels=np.unique(y_train),
        target_names=class_names if class_names else np.unique(y_train),
        output_dict=False,
    )

    test_report = classification_report(
        y_test,
        y_preds_test,
        labels=np.unique(y_test),
        target_names=class_names if class_names else np.unique(y_test),
        output_dict=False,
    )

    logger.info(f"Train Classification report: \n{train_report}")
    logger.info("")
    print_mislabeled_points(y_train, y_preds_train)
    logger.info("")
    logger.info(f"Test Classification report: \n{test_report}")
    print_mislabeled_points(y_test, y_preds_test)

    return estimator


def plot_classifier_decision_boundary(
    estimator: TEstimator, X: NDArray[np.floating[Any]], y: NDArray[np.floating[Any]]
) -> None:
    """Plot the decision boundary of a classifier."""
    assert X.shape[1] == 2, "Can only plot decision boundary for 2 features."

    estimator.fit(X, y)
    y_preds = estimator.predict(X)
    logger.info(f"Train Classification report: \n{classification_report(y, y_preds)}")
    print_mislabeled_points(y, y_preds)

    # setup marker generator and color map
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])
    plot_decision_regions(
        X,
        y,
        classifier=estimator,
        markers=markers,
        colors=colors,
        contourf=True,
        alpha=0.3,
        cmap=cmap,
    )


def print_mislabeled_points(y_trues: NDArray[np.floating[Any]], y_preds: NDArray[np.floating[Any]]) -> None:
    """Print the mislabeled points."""
    mislabeled_points = (y_trues != y_preds).sum()
    logger.info("Mislabeled points: %s", mislabeled_points / y_trues.shape[0])


def make_meshgrid(
    x1: NDArray[np.floating[Any]], x2: NDArray[np.floating[Any]], step: float = 0.02
) -> Tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Create a mesh of points to plot in

    Parameters
    ----------
    x1 : NDArray[np.floating[Any]]
        data to base x1-axis meshgrid on
    x2 : NDArray[np.floating[Any]]
        data to base x2-axis meshgrid on
    step : float, optional (default=0.02)
        stepsize for meshgrid, by default 0.02

    Returns
    -------
    xx1, xx2 : NDArray[np.floating[Any]]
        generated meshgrid
    """
    x1_min, x1_max = x1.min() - 1, x1.max() + 1
    x2_min, x2_max = x2.min() - 1, x2.max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, step), np.arange(x2_min, x2_max, step))
    return xx1, xx2


def plot_decision_regions(
    X: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    classifier: BaseEstimator,
    # test_idx: Optional[int] = None,
    markers: Optional[Tuple[str, ...]] = None,
    colors: Optional[Tuple[str, ...]] = None,
    cmap: Optional[ListedColormap] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs: Any,
) -> None:
    """Plot decision regions.

    Code adapted from https://github.com/rasbt/mlxtend/blob/master/mlxtend/plotting/decision_regions.py

    The purpose of the re-implementation is to understand what's going on under the hood.
    """
    ax = ax or plt.gca()

    # setup marker generator and color map
    markers = ("s", "x", "o", "^", "v") if markers is None else markers
    colors = ("red", "blue", "lightgreen", "gray", "cyan") if colors is None else colors
    cmap = ListedColormap(colors[: len(np.unique(y))]) if cmap is None else cmap

    xx1, xx2 = make_meshgrid(X[:, 0], X[:, 1])

    X_input_space = np.array([xx1.ravel(), xx2.ravel()]).T  # N x 2 matrix

    Z = classifier.predict(X_input_space)
    Z = Z.reshape(xx1.shape)  # reshape to match xx1 and xx2 to plot contour

    contour = plot_contour(ax, xx1, xx2, Z, cmap=cmap, **kwargs)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    ax.clabel(contour, inline=True, fontsize=8)

    for idx, cl in enumerate(np.unique(y)):
        plot_scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            ax=ax,
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=cl,
            edgecolor="black",
        )
    plt.show()
