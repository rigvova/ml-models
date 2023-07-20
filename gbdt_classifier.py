from __future__ import annotations

from typing import Tuple, Callable

import numpy as np

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
    def _logloss(y_true: np.ndarray, y_pred: np.ndarray or float) -> Tuple[float, np.ndarray]:
        """
        Compute the logloss and gradient for a given set of target labels
        and predicted positive class probabilities.
        """
        a = 0.001
        loss = -float(np.mean(y_true * np.log(y_pred + a) + (1 - y_true) * np.log(1 - y_pred + a)))
        grad = y_pred - y_true
        return loss, grad

    def _subsample(self, X, y):
        """Performs sampling for SGD."""
        n_samples = len(y)

        idx = np.random.choice(
            n_samples,
            size=int(n_samples * self.subsample_size),
            replace=self.replace
        )

        sub_X, sub_y = X[idx], y[idx]

        return sub_X, sub_y

    def fit(self, X, y) -> GradientBoostingClassifier:
        """
        Fit the model to the data.

        Args:
            X: array-like of shape (n_samples, n_features)
            y: array-like of shape (n_samples,)

        Returns:
            GradientBoostingClassifier: The fitted model.
        """
        self.base_pred_ = np.mean(y)  # positive class proba
        self.trees_ = []

        # Iteratively train trees on anti gradients
        # of previous predictions
        n_trees = 0
        _, grad = self.loss(y, self.base_pred_)
        while n_trees < self.n_estimators:

            # Fit a new tree on anti grad of the previous prediction error
            new_tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            new_tree.fit(*self._subsample(X, -grad))

            # Add the tree to the list
            self.trees_.append(new_tree)
            n_trees += 1

            # Update the gradient
            pred = self.predict(X)
            _, grad = self.loss(y, pred)

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
        pred = np.ones((X.shape[0])) * self.base_pred_
        for tree in self.trees_:
            pred += tree.predict(X) * self.learning_rate

        return pred
