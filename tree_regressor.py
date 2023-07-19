from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Node:
    """Decision tree node."""
    n_samples: int = None  # how many data objects are in node
    feature: int = None  # node split feature
    threshold: float = None  # node feature split threshold
    score: float = None  # node (leaf) prediction (target mean)
    mse: float = None  # mse when `score` is predicted
    left: Node = None
    right: Node = None


class DecisionTreeRegressor:
    """Implements a decision tree regressor."""

    def __init__(self, max_depth: int, min_samples_split: int = 2):
        """
        Initializes a new instance of DecisionTreeRegressor.

        Args:
            max_depth (int): The maximum depth of the tree. This parameter defines the maximum
                number of nodes along the longest path from the root node down to the farthest leaf node.
                The value must be greater than 0.
            min_samples_split (int, optional): The minimum number of samples required to split
                an internal node. Defaults to 2.
        """
        self.n_features_: int or None = None
        self.tree_: Node or None = None  # tree root

        self.max_depth: int = max_depth
        self.min_samples_split: int = min_samples_split

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeRegressor:
        """
        Build a decision tree regressor from the training set (X, y).

        Args:
            X (np.ndarray): The input samples with shape (n_samples, n_features).
            y (np.ndarray): The target values with shape (n_samples,).

        Returns:
            DecisionTreeRegressor: The fitted decision tree regressor object.
        """
        self.n_features_ = X.shape[1]
        self.tree_ = self._split_node(X, y)
        return self

    @staticmethod
    def _mse(y: np.ndarray) -> float:
        """Compute the mse criterion for a given set of target values."""
        return float(np.mean((y - np.mean(y)) ** 2))

    def _weighted_mse(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """
        Computes the weighted mean squared error (MSE) criterion for two given sets of target values.
        The weighted MSE is computed as the sum of the MSE of the left set multiplied by its length and
        the MSE of the right set multiplied by its length.

        Args:
            y_left (np.ndarray): The target values of the left set, array of shape (n_samples_left,).
            y_right (np.ndarray): The target values of the right set, array of shape (n_samples_right,).

        Returns:
            float: The weighted MSE of the two sets. Returns NaN if the right set is empty.
        """
        left_arr_n = len(y_left)
        right_arr_n = len(y_right)
        if right_arr_n == 0:
            return np.nan
        return left_arr_n * self._mse(y_left) + right_arr_n * self._mse(y_right)
        # Could use normalized version instead:
        # return ((left_arr_n * self._mse(y_left) + right_arr_n * self._mse(y_right))
        #         / (left_arr_n + right_arr_n))

    def _best_feature_split(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Find the best split for a node (one feature)"""
        unq = sorted(np.unique(x))
        best_threshold = None
        best_wmse = float("inf")

        for thresh in unq:
            y_left = y[x <= thresh]
            y_right = y[x > thresh]
            wmse = self._weighted_mse(y_left, y_right)
            if wmse < best_wmse:
                best_wmse = wmse
                best_threshold = thresh

        return best_wmse, best_threshold

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        """Find the best split for a node."""
        best_feature = None
        best_threshold = None
        best_wmse = float('inf')

        for i in range(X.shape[1]):
            wmse, threshold = self._best_feature_split(X[:, i], y)
            if wmse < best_wmse:
                best_wmse = wmse
                best_threshold = threshold
                best_feature = i

        return best_feature, best_threshold

    def _split_node(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Split a node and return the resulting left and right child nodes."""
        node = Node()
        node.n_samples = X.shape[0]
        node.score = float(np.mean(y))
        node.mse = self._mse(y)

        if (depth == self.max_depth) or (node.n_samples < self.min_samples_split):
            node.left = None
            node.right = None
        else:
            node.feature, node.threshold = self._best_split(X, y)
            idx_left = (X[:, node.feature] <= node.threshold)
            idx_right = (X[:, node.feature] > node.threshold)
            node.left = self._split_node(X[idx_left], y[idx_left], depth=depth+1)
            node.right = self._split_node(X[idx_right], y[idx_right], depth=depth+1)

        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression target for X.

        Args:
            X (np.ndarray): The input samples, array of shape (n_samples, n_features).

        Returns:
            np.ndarray: The predicted values, array of shape (n_samples,).
        """
        return np.array([self._predict_one_sample(x) for x in X])

    def _predict_one_sample(self, features: np.ndarray) -> float:
        """
        Predict the target value of a single sample.

        Args:
            features (np.ndarray): The feature values of the sample to be predicted,
                array of shape (n_features,).

        Returns:
            float: The predicted target value for the input sample.
        """
        return self._tree_descent(self.tree_, features)

    def _tree_descent(self, node: Node, features: np.ndarray) -> float:
        """
        Navigate the decision tree to predict the target value of a single sample.

        Args:
            node (Node): The current node in the decision tree.
            features (np.ndarray): The feature values of the sample to be predicted,
                array of shape (n_features,).

        Returns:
            float: The predicted target value for the current node. If the node is a leaf,
                the node's score is returned. If the node is not a leaf, the function
                is called recursively on either the left or right child of the node.
        """
        if node.left is None:
            # Node is leaf
            return node.score
        else:
            if features[node.feature] <= node.threshold:
                return self._tree_descent(node.left, features)
            else:
                return self._tree_descent(node.right, features)
