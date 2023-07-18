from __future__ import annotations

from typing import Tuple

import numpy as np


class LinearRegression:
    """
    Implements Linear Regression with Mean Squared Error (MSE) loss and regularization options.

    Attributes:
        w (numpy.ndarray): The weights of the model.
        w0 (float): The bias term of the model.
        n_samples (int): The number of training samples.
        n_features (int): The number of features.
        lr (float): The learning rate for gradient descent.
        alpha (float): The coefficient for L2 regularization (Ridge).
        beta (float): The coefficient for L1 regularization (Lasso).
    """

    def __init__(self, lr=0.01, alpha=0.0, beta=0.0):
        """
        Initializes the LinearRegression model with the given learning rate, L2 regularization coefficient,
        and L1 regularization coefficient.

        Args:
            lr (float, optional): Learning rate for gradient descent. Defaults to 0.01.
            alpha (float, optional): The coefficient for L2 regularization. Defaults to 0.0.
            beta (float, optional): The coefficient for L1 regularization. Defaults to 0.0.
        """
        self.w = None
        self.w0 = None
        self.n_samples = None
        self.n_features = None

        self.lr = lr
        self.alpha = alpha
        self.beta = beta

    def _mse_anti_grad_step(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Performs one anti-gradient step for the MSE loss function and checks if the MSE loss decreases.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target values.

        Returns:
            bool: Flag indicating whether to stop the algorithm (if MSE loss doesn't decrease).
        """
        stop_flag = False
        y_pred = X @ self.w + self.w0
        dw, dw0 = self._mse_gradient(X, y, y_pred)
        next_w = self.w - self.lr * dw
        next_w0 = self.w0 - self.lr * dw0
        next_pred = X @ next_w + next_w0
        if self._mse_loss(y, y_pred) > self._mse_loss(y, next_pred):
            self.w = next_w
            self.w0 = next_w0
        else:
            stop_flag = True

        return stop_flag

    def _mse_gradient(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Computes the gradient of the MSE loss function with respect to the weights and bias.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target values.
            y_pred (numpy.ndarray): The predicted values.

        Returns:
            tuple: A tuple containing the gradients with respect to the weights and bias.
        """
        dw = ((X.T @ (y_pred - y)) / self.n_samples
              + self.alpha * 2 * self.w
              + self.beta * np.sign(self.w))
        dw0 = np.sum(y_pred - y) / self.n_samples
        return dw, dw0

    def _mse_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the MSE loss between the true and predicted values.

        Args:
            y_true (numpy.ndarray): The true target values.
            y_pred (numpy.ndarray): The predicted target values.

        Returns:
            float: The MSE loss.
        """
        return np.sum((y_true - y_pred) ** 2) / self.n_samples

    def _rmse_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the Root Mean Square Error (RMSE) between the true and predicted values.

        Args:
            y_true (numpy.ndarray): The true target values.
            y_pred (numpy.ndarray): The predicted target values.

        Returns:
            float: The RMSE loss.
        """
        return np.sqrt(self._mse_loss(y_true, y_pred))

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

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearRegression:
        """
        Trains the model using the provided data `X` and target labels `y`.

        Args:
            X (numpy.ndarray): The input data, a 2D array with the shape (N, D).
            y (numpy.ndarray): The target values, an array with the shape (N,).
        """
        self._set_initial_params(X)

        y_pred = X @ self.w + self.w0
        print("Initial RMSE: {}".format(self._rmse_loss(y, y_pred)))

        step = 1
        while True:
            stop_flag = self._mse_anti_grad_step(X, y)
            if stop_flag:
                break
            y_pred = X @ self.w + self.w0
            print("Step: {}, RMSE: {}".format(step, self._rmse_loss(y, y_pred)))
            step += 1

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target values using the model for the given data.

        Args:
            X (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The predicted target values.
        """
        return X @ self.w + self.w0
