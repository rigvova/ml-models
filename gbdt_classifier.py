from __future__ import annotations

from typing import Tuple, Callable

import numpy as np
from scipy.special import expit

from tree_regressor import DecisionTreeRegressor


class GradientBoostingClassifier:
    """Gradient Boosting on regression trees for binary classification."""
    def __init__(
            self,
            n_estimators: int = 100,
            learning_rate: float = 0.1,
            max_depth: int = 3,
            min_samples_split: int = 2,
            loss: str or Callable = "logloss",
            subsample_size: float = 0.5,
            replace: bool = False,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample_size = subsample_size
        self.replace = replace

        if loss == "logloss":
            self.loss = self._logloss
        else:
            self.loss = loss

        self.base_pred_ = None
        self.trees_ = None

    @staticmethod
    def log_odds(y: np.ndarray) -> float:
        """Computes log of the odds for the given target."""
        pos = y.sum()
        return float(np.log(pos / (len(y) - pos)))

    @staticmethod
    def _logloss(y_true: np.ndarray, y_pred: np.ndarray or float) -> Tuple[float, np.ndarray]:
        """
        Compute the logloss and gradient for a given set of target labels
        and predicted positive class probabilities.
        """
        a = 0.001
        loss = -float(np.mean(y_true * np.log(y_pred + a) + (1 - y_true) * np.log(1 - y_pred + a)))
        grad = y_pred - y_true
        return loss, grad

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        """
        Applies the sigmoid function to the input logits.

        The sigmoid function is defined as sigmoid(x) = 1 / (1 + exp(-x)),
        and it maps any input to a value in the range (0, 1).

        Args:
            logits (numpy.ndarray): The input logits.

        Returns:
            numpy.ndarray: The output after applying the sigmoid function.
        """
        return expit(logits)

    def _subsample(self, X, y, probas):
        """Performs sampling for SGD."""
        n_samples = len(y)

        idx = np.random.choice(
            n_samples,
            size=int(n_samples * self.subsample_size),
            replace=self.replace
        )

        return X[idx], y[idx], probas[idx]

    def fit(self, X, y) -> GradientBoostingClassifier:
        """
        Fit the model to the data.

        Args:
            X: array-like of shape (n_samples, n_features)
            y: array-like of shape (n_samples,)

        Returns:
            GradientBoostingClassifier: The fitted model.
        """
        # use log of the odds as the base proba prediction
        self.base_pred_ = np.ones_like(y) * self.log_odds(y)
        self.trees_ = []
        probas = self._sigmoid(self.base_pred_)

        # Iteratively train trees on anti gradients
        # of previous predictions
        n_trees = 0
        _, grad = self.loss(y, probas)
        while n_trees < self.n_estimators:

            # Fit a new tree on anti grad of the previous prediction error
            new_tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            X_sub, antigrad_sub, probas_sub = self._subsample(X, -grad, probas)
            new_tree.fit(X_sub, antigrad_sub, probas=probas_sub)

            # Add the tree to the list
            self.trees_.append(new_tree)
            n_trees += 1

            # Update the gradient
            logits = self.predict(X)
            probas = self._sigmoid(logits)

            _, grad = self.loss(y, probas)

        return self

    def predict(self, X):
        """
        Predict the target labels of new data.

        Args:
            X: array-like of shape (n_samples, n_features)

        Returns:
            np.ndarray: The model prediction, array  of shape (n_samples,),
        """
        probas = self.predict_proba(X)
        return (probas >= 0.5).astype(np.uint8)

    def predict_proba(self, X):
        """
        Predict the target probabilities of new data.

        Args:
            X: array-like of shape (n_samples, n_features)

        Returns:
            np.ndarray: The model prediction, array  of shape (n_samples,),
        """
        logits = np.ones(shape=X.shape[0]) * self.base_pred_[0]
        for tree in self.trees_:
            logits += tree.predict(X) * self.learning_rate

        return self._sigmoid(logits)
