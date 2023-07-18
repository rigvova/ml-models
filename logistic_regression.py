from __future__ import annotations

from typing import Tuple

import numpy as np


class LogisticRegression:
    """
    Implements Linear Regression with Mean Squared Error (MSE) loss and regularization options.

    Attributes:
        w (numpy.ndarray): The weights of the model.
        w0 (float): The bias term of the model.
        n_samples (int): The number of training samples.
        n_features (int): The number of features.
        lr (float): The learning rate for gradient descent.
        tol (float): Tolerance for the optimization algorithm, used to determine
            convergence has been reached.
        alpha (float): The coefficient for L2 regularization (Ridge).
        beta (float): The coefficient for L1 regularization (Lasso).
    """

    def __init__(self, lr=0.01, tol=0.001, alpha=0.0, beta=0.0):
        """
        Initializes the LinearRegression model with the given learning rate, L2 regularization coefficient,
        and L1 regularization coefficient.

        Args:
            lr (float, optional): Learning rate for gradient descent. Defaults to 0.01.
            tol (float, optional): Tolerance for the optimization algorithm, used to determine
                when convergence has been reached. Defaults to 0.001.
            alpha (float, optional): The coefficient for L2 regularization. Defaults to 0.0.
            beta (float, optional): The coefficient for L1 regularization. Defaults to 0.0.
        """
        self.w = None
        self.w0 = None
        self.n_samples = None
        self.n_features = None

        self.lr = lr
        self.tol = tol
        self.alpha = alpha
        self.beta = beta

    def _logloss(self, y: np.ndarray, preds: np.ndarray) -> float:
        """
        Computes the log loss (also called logistic loss or cross-entropy loss) between the true and predicted values.

        Args:
            y (numpy.ndarray): The true target labels.
            preds (numpy.ndarray): The predicted positive class probabilities.

        Returns:
            float: The log loss.
        """
        return -np.sum(y * np.log(preds) + (1 - y) * np.log(1 - preds)) / self.n_samples

    def _logloss_anti_grad_step(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Performs one anti-gradient step for the log loss function and checks if the log loss decreases by a value
        less than the tolerance.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target values.

        Returns:
            bool: Flag indicating whether to stop the algorithm (if the decrease in log loss
                is less than the tolerance).
        """
        stop_flag = False
        y_probas = self._sigmoid(X @ self.w + self.w0)
        dw, dw0 = self._logloss_gradient(X, y, y_probas)
        next_w = self.w - self.lr * dw
        next_w0 = self.w0 - self.lr * dw0
        next_probas = self._sigmoid(X @ next_w + next_w0)
        if self._logloss(y, y_probas) - self._logloss(y, next_probas) < self.tol:
            stop_flag = True
        else:
            self.w = next_w
            self.w0 = next_w0

        return stop_flag

    def _logloss_gradient(self, X: np.ndarray, y: np.ndarray, preds: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Computes the gradient of the log loss function with respect to the weights and bias.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The true target labels.
            preds (numpy.ndarray): The predicted positive class probabilities.

        Returns:
            tuple: A tuple containing the gradients with respect to the weights and bias.
        """
        dw = ((X.T @ (preds - y)) / self.n_samples
              + self.alpha * 2 * self.w
              + self.beta * np.sign(self.w))
        dw0 = np.sum(preds - y) / self.n_samples
        return dw, dw0

    def _set_initial_params(self, X: np.ndarray):
        """
        Sets the initial parameters of the model based on the input data.

        Args:
            X (numpy.ndarray): The input data.
        """
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.w = np.random.rand(self.n_features)
        self.w0 = 0

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
        return 1 / (1 + np.exp(-logits))

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        """
        Trains the model using the provided data `X` and target labels `y`.

        Args:
            X (numpy.ndarray): The input data, a 2D array with the shape (N, D).
            y (numpy.ndarray): The target values, an array with the shape (N,).
        """
        self._set_initial_params(X)

        y_pred = self._sigmoid(X @ self.w + self.w0)
        print("Initial Logloss: {}".format(self._logloss(y, y_pred)))

        step = 1
        while True:
            stop_flag = self._logloss_anti_grad_step(X, y)
            if stop_flag:
                break
            y_pred = self._sigmoid(X @ self.w + self.w0)
            print("Step: {}, Logloss: {}".format(step, self._logloss(y, y_pred)))
            step += 1

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target probabilities using the model for the given data.

        Args:
            X (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The predicted target values.
        """
        return self._sigmoid(X @ self.w + self.w0)
