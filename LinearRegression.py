import numpy as np

class LinearRegression():
    def __init__(self, lr=0.01, alpha=0.0, beta=0.0):
        self.w = None
        self.w0 = None
        self.n_samples = 0
        self.n_features = 0

        self.lr = lr
        self.alpha = alpha
        self.beta = beta


    def __mse_anti_grad_step(self, X, y):
        stop_flag = False
        y_pred    = X @ self.w + self.w0
        dw, dw0   = self.__mse_gradient(X, y, y_pred)
        next_w    = self.w  - self.lr * dw
        next_w0   = self.w0 - self.lr * dw0
        if self.__mse_loss(y, y_pred) > self.__mse_loss(y, X @ next_w + next_w0):
            self.w  = next_w
            self.w0 = next_w0
        else:
            stop_flag=True

        return stop_flag


    def __mse_gradient(self, X, y, y_pred):
        dw  = -X.T @ (y - y_pred) / (self.n_samples) + self.alpha * 2 * self.w  + self.beta * np.sign(self.w)
        dw0 = -np.sum(y - y_pred) / (self.n_samples) + self.alpha * 2 * self.w0 + self.beta * np.sign(self.w0)
        return dw, dw0


    def __mse_loss(self, y_true, y_pred):
        return np.sum((y_true - y_pred)**2) / self.n_samples

    
    def __rmse_loss(self, y_true, y_pred):
        return np.sqrt(self.__mse_loss(y_true, y_pred))


    def __set_initial_params(self, X):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.w = np.random.rand(self.n_features)
        self.w0 = 0


    def fit(self, X, y):
        self. __set_initial_params(X)

        y_pred = X @ self.w + self.w0
        print("Initial RMSE: {}".format(self.__rmse_loss(y, y_pred)))

        stop_flag = False
        step = 1
        while True:
            stop_flag = self.__mse_anti_grad_step(X, y)
            if stop_flag:
                break
            else:
                y_pred = X @ self.w + self.w0
                print("Step: {}, RMSE: {}".format(step, self.__rmse_loss(y, y_pred)))
                step += 1
            
    
    def predict(self, X):
        return X @ self.w + self.w0
