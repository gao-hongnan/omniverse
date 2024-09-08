"""Naive Bayes Classifier.

High level module with business logic.

Classification using Bayes Rule P(Y|X) = P(X|Y)*P(Y)/P(X),
    or Posterior = Likelihood * Prior / Scaling Factor
    P(Y|X) - The posterior is the probability that sample x is of class y given the
            feature values of x being distributed according to distribution of y and the prior.
    P(X|Y) - Likelihood of data X given class distribution Y.
            Gaussian distribution (given by _calculate_likelihood)
    P(Y)   - Prior (given by _calculate_prior)
    P(X)   - Scales the posterior to make it a proper probability distribution.
            This term is ignored in this implementation since it doesn't affect
            which class distribution the sample is most likely to belong to.
    Classifies the sample as the class that results in the largest P(Y|X) (posterior)

Reference from https://github.com/scikit-learn/scikit-learn/blob/98cf537f5/sklearn/naive_bayes.py.

NOTE:
    1. This does not implement the log-likelihood version. Note that the log-likehood
    version will avoid underflow when the likelihood is very small.

    2. The choice of not defining theta to be of shape K \times D \times 2
    is because we want theta to follow my notes in concept.md. This may
    not be a good design here.

    3. Rename parameters to theta.

    4. Try to implement predict_proba and predict_log_proba following sklearn.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
from scipy.special import logsumexp

from src.base.estimator import BaseEstimator
from src.base.types import T

"""Concrete Implementation of Naive Bayes Hyperparams."""
from __future__ import annotations

from src.base.hyperparams import Hyperparams

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class NaivesBayesHyperparams(Hyperparams):
    random_state: int
    num_classes: int


# pylint: disable=too-many-instance-attributes
class NaiveBayesGaussian(BaseEstimator):
    num_samples: int
    num_features: int

    theta: List[List[Dict[str, float]]]
    pi: T

    prior: T
    likelihood: T
    posterior: T

    def __init__(self, random_state: int = 1992, num_classes: int = 3) -> None:
        self.random_state = random_state
        self.num_classes = num_classes

    def _set_num_samples_and_features(self, X: T) -> None:
        # num_samples unused since we vectorized when estimating parameters
        self.num_samples, self.num_features = X.shape

    def fit(self, X: T, y: Optional[T] = None) -> NaiveBayesGaussian:
        """Fit Naive Bayes classifier according to X, y.

        Note:
            Fitting Naive Bayes involves us finding the theta and pi vector.

        Args:
            X (T): N x D matrix
            y (T): N x 1 vector
        """
        self._set_num_samples_and_features(X)

        # Calculate the mean and variance of each feature for each class
        self.theta = self._estimate_likelihood_parameters(X, y)  # this is theta_{X|Y}
        self.pi = self._estimate_prior_parameters(y)  # this is \boldsymbol{\pi}
        return self

    def _estimate_prior_parameters(self, y: T) -> T:
        """Calculate the prior probability of each class.

        Returns a vector of prior probabilities for each class.
        prior = [P(Y = 0), P(Y = 1), ..., P(Y = k)]
        """
        pi = np.zeros(self.num_classes)
        # use for loop for readability or np.bincount(y) / len(y)
        for k in range(self.num_classes):
            pi[k] = np.sum(y == k) / len(y)
        return pi

    def _estimate_likelihood_parameters(self, X: T, y: T) -> List[List[Dict[str, float]]]:
        """Estimate the mean and variance of each feature for each class.

        The final theta should have shape K \times D.
        """
        # corresponds to theta_{X|Y} matrix but the last two dimensions
        # is the mean and variance of the feature d given class k.
        parameters = np.zeros((self.num_classes, self.num_features, 2))

        for k in range(self.num_classes):
            # Only select the rows where the label equals the given class
            X_where_k = X[np.where(y == k)]  # shape = (num_samples, num_features)
            for d in range(self.num_features):
                mean = X_where_k[:, d].mean()
                var = X_where_k[:, d].var()
                # encode mean as first element and var as second
                parameters[k, d, :] = [mean, var]
        return parameters

    @staticmethod
    def _calculate_conditional_gaussian_pdf(x: T, mean: float, var: float, eps: float = 1e-4) -> float:
        """Univariate Gaussian likelihood of the data x given mean and var.

        $\mathbb{P}(X_d = x_d | Y = k)$

        Args:
            eps (float): Added in denominator to prevent division by zero.
        """
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var + eps)))
        return coeff * exponent

    def _calculate_prior(self) -> T:
        """Calculate the prior probability of each class.

        Returns a vector of prior probabilities for each class.
        prior = [P(Y = 0), P(Y = 1), ..., P(Y = K)].T
        This is our matrix M1 in the notes, and M1 = pi
        due to the construction of the Catagorical distribution.
        """
        prior = self.pi
        return prior

    def _calculate_joint_likelihood(self, x: T) -> T:
        """Calculate the joint likelihood of the data x given the parameters.

        $P(X|Y) = \prod_{d=1}^{D} P(X_d|Y)$

        This is our matrix M2 (M3) in the notes.

        Args:
            x (T): A vector of shape (num_features,).

        Returns:
            T: A vector of shape (num_classes,).
        """
        likelihood = np.ones(self.num_classes)  # M2 matrix in notes
        M3 = np.ones((self.num_classes, self.num_features))  # M3 matrix in notes
        for k in range(self.num_classes):
            for d in range(self.num_features):
                mean = self.theta[k, d, 0]
                var = self.theta[k, d, 1]
                M3[k, d] = self._calculate_conditional_gaussian_pdf(x[d], mean, var)

        likelihood = np.prod(M3, axis=1)
        return likelihood

    def _calculate_posterior(self, x: T) -> T:
        """Calculates posterior for one single data point x."""
        # x: (num_features,) 1 sample
        self.prior = self._calculate_prior()
        self.likelihood = self._calculate_joint_likelihood(x)
        # NOTE: technically this is not posterior as it is not normalized with marginal!
        # M3 * M1
        self.posterior = self.likelihood * self.prior
        return self.posterior

    def predict_one_sample(self, x: T) -> T:
        """Predict the posterior of one sample x."""
        return self._calculate_posterior(x)

    def predict(self, X: T) -> T:
        """Predict the class labels of all the samples in X. Note
        that X can be any data (i.e. unseen data)."""
        num_samples = X.shape[0]
        y_preds = np.ones(num_samples)
        for sample_index, x in enumerate(X):
            # argmax returns the index of the maximum value
            y_preds[sample_index] = np.argmax(self.predict_one_sample(x), axis=0)
        return y_preds

    def predict_pdf(self, X: T) -> T:
        """Predict the class PDF of all the samples in X."""
        num_samples = X.shape[0]
        # note the shape is num_samples x num_classes because we are not argmax it
        y_probs = np.ones((num_samples, self.num_classes))
        for sample_index, x in enumerate(X):
            y_probs[sample_index] = self.predict_one_sample(x)
        return y_probs

    def predict_proba(self, X: T) -> T:
        """Predict the class probabilities of all the samples in X.
        Normalize it to get the probabilities."""
        y_probs = self.predict_pdf(X)
        # normalize the pdf to get the probs
        y_probs = y_probs / np.sum(y_probs, axis=1, keepdims=True)
        return y_probs


class NaiveBayesGaussianLogLikelihood(NaiveBayesGaussian):
    """In order not to pollute the original class, we create a new class
    and implement the log-likelihood version of the Gaussian Naive Bayes."""

    def _calculate_joint_log_likelihood(self, x: T) -> T:
        """Calculate the joint log-likelihood of the data x given the parameters.

        log(P(X|Y)) = sum_{d=1}^{D} log(P(X_d|Y))

        Args:
            x (T): A vector of shape (num_features,).

        Returns:
            log_likelihood (T): A vector of shape (num_classes,).
        """
        log_likelihood = np.zeros(self.num_classes)
        for k in range(self.num_classes):
            for d in range(self.num_features):
                mean = self.theta[k, d, 0]
                var = self.theta[k, d, 1]
                log_likelihood[k] += np.log(self._calculate_conditional_gaussian_pdf(x[d], mean, var))
        return log_likelihood

    def predict_log_proba(self, X: T) -> T:
        """Predict the log-probabilities of all the samples in X.

        Args:
            X (T): N x D matrix.

        Returns:
            T: N x K matrix with log-probabilities for each sample.
        """
        num_samples = X.shape[0]
        log_probs = np.zeros((num_samples, self.num_classes))
        for sample_index, x in enumerate(X):
            log_joint_likelihood = self._calculate_joint_log_likelihood(x)
            log_prior = np.log(self.pi)
            log_posterior = log_joint_likelihood + log_prior
            # log_probs[sample_index] = log_posterior - logsumexp(log_posterior)
            log_probs[sample_index] = log_posterior - np.log(np.sum(np.exp(log_posterior)))
        return log_probs

    def predict_proba(self, X: T) -> T:
        """Predict the probabilities of all the samples in X.

        np.exp is used to convert log-probabilities to probabilities.

        Args:
            X (T): N x D matrix.

        Returns:
            T: N x K matrix with probabilities for each sample.
        """
        log_probs = self.predict_log_proba(X)
        return np.exp(log_probs)
