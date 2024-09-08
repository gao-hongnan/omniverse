import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB

from omnivault.machine_learning.generative.naive_bayes import NaiveBayesGaussianLogLikelihood


@pytest.mark.parametrize(argnames="num_classes", argvalues=[2, 3, 4])
def test_naive_bayes_vs_sklearn(num_classes: int) -> None:
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=num_classes,
        random_state=1992,
    )

    nb_custom = NaiveBayesGaussianLogLikelihood(num_classes=num_classes, random_state=1992)
    nb_custom.fit(X, y)

    nb_sklearn = GaussianNB()
    nb_sklearn.fit(X, y)

    # Check if the parameters are close enough
    np.testing.assert_allclose(nb_custom.pi, nb_sklearn.class_prior_, rtol=1e-3)
    np.testing.assert_allclose(nb_custom.theta[:, :, 0], nb_sklearn.theta_, rtol=1e-3)
    np.testing.assert_allclose(nb_custom.theta[:, :, 1], nb_sklearn.var_, rtol=1e-3)

    # Check if predictions are close enough
    custom_probs = nb_custom.predict_proba(X)
    sklearn_probs = nb_sklearn.predict_proba(X)
    np.testing.assert_allclose(custom_probs, sklearn_probs, rtol=1e-2, atol=1e-2)

    if hasattr(nb_custom, "predict_log_proba"):
        custom_log_probs = nb_custom.predict_log_proba(X)
        sklearn_log_probs = nb_sklearn.predict_log_proba(X)
        np.testing.assert_allclose(custom_log_probs, sklearn_log_probs, rtol=1e-2, atol=1e-2)
