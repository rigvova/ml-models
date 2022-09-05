import numpy as np

class LogisticRegression():
    def __init__(self, lr=0.01, tol=0.001, alpha=0.0, beta=0.0):
        self.w = None
        self.w0 = None
        self.n_samples = 0
        self.n_features = 0
        
        self.lr = lr
        self.tol = tol
        self.alpha = alpha
        self.beta = beta


    def __logloss(self, y_true, y_pred):
        return -np.sum(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred)) / self.n_samples


    def __logloss_anti_grad_step(self, X, y):
        stop_flag = False
        y_pred    = self.__sigmoid(X @ self.w + self.w0)
        dw, dw0   = self.__logloss_gradient(X, y, y_pred)
        next_dw   = self.w - self.lr * dw
        next_dw0  = self.w0 - self.lr * dw0

        if self.__logloss(y, y_pred) - self.__logloss(y, self.__sigmoid(X @ next_dw + next_dw0)) < self.tol:
            stop_flag=True
        else:
            self.w  = next_dw
            self.w0 = next_dw0
        
        return stop_flag


    def __logloss_gradient(self, X, y_true, y_pred):
        dw  = -X.T @ (y_true-y_pred) / self.n_samples + self.alpha * 2 * self.w  + self.beta * np.sign(self.w)
        dw0 = -np.sum(y_true-y_pred) / self.n_samples
        return dw, dw0


    def __set_initial_params(self, X):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.w = np.random.rand(self.n_features)
        self.w0 = 0


    def __sigmoid(self, logits):
        return 1 / (1 + np.exp(-logits))


    def fit(self, X, y):
        self.__set_initial_params(X)

        y_pred = self.__sigmoid(X @ self.w + self.w0)
        print("Initial Logloss: {}".format(self.__logloss(y, y_pred)))

        stop_flag = False
        step = 1
        while True:
            stop_flag = self.__logloss_anti_grad_step(X, y)
            if stop_flag:
                break
            else:
                y_pred = self.__sigmoid(X @ self.w + self.w0)
                print("Step: {}, Logloss: {}".format(step, self.__logloss(y, y_pred)))
                step += 1


    def predict_proba(self, X):
        return self.__sigmoid(X @ self.w + self.w0)
