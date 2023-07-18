import numpy as np

class RegressionTree:
    def __init__(
        self,
        max_depth=2,
        min_leaf_size=3
        ):

        self.X = None
        self.y = None

        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size

        self.tree = None


    def fit(self, X, y):
        self.X = X
        self.y = y
        self.tree = RegressionNode(X, y, 0,
            max_depth=self.max_depth,
            min_leaf_size=self.min_leaf_size)
        self.tree.split()


    def predict(self, X):
        return np.apply_along_axis(
            self.tree.get_prediction,
            axis=1,
            arr=X)


class RegressionNode:
    def __init__(
        self,
        X,
        y,
        cur_depth=0,
        max_depth=2,
        min_leaf_size=3
        ):
        
        # Init data
        self.X = X
        self.y = y
        self.n_samples  = X.shape[0]
        self.n_features = X.shape[1]

        # Init params
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size

        # Depth level of this node
        self.cur_depth = cur_depth

        # Node children & split
        self.left  = None
        self.right = None
        self.best_feature = None
        self.best_split   = None

        self.ymean = np.mean(y) # target mean (node prediction)

        self.mse = self._mse(self.y, self.ymean, self.n_samples) # prediction error

        self.is_leaf = False # True if cur node is last node

    
    def _calc_best_split(self):
        best_feature = None
        best_split = None
        best_mse = self.mse

        # Pick best feature for split
        for feature in range(self.n_features):
            sorted_values = self._make_splits_array(self.X[:, feature].ravel())
            # Find best split value
            for split in sorted_values:
                mask = (self.X[:,feature].ravel() < split)
                yleft  = self.y[mask]
                yright = self.y[~mask]

                yleft_mean  = np.mean(yleft)
                yright_mean = np.mean(yright)

                split_mse = (np.sum((yleft-yleft_mean)**2) + np.sum((yright-yright_mean)**2)) \
                             / self.n_samples

                if split_mse < best_mse:
                    best_feature = feature
                    best_split   = split
                    best_mse     = split_mse

        return (best_feature, best_split)


    @staticmethod
    def _make_splits_array(arr: np.array) -> np.array:
        """
        Makes array of all possible splits for a numerical 1D-array.
        Uses moving average window of size 2.
        [1,2,3] -> [1.5, 2.5]
        """
        return np.convolve(sorted(set(arr)), np.ones(2)/2, mode='valid')


    def _mse(self, y_true, y_pred, n_samples):
        return np.sum((y_true - y_pred)**2) / n_samples


    def get_prediction(self, sample):
        if self.is_leaf:
            return self.ymean
        else:
            if sample[self.best_feature] <= self.best_split:
                return self.left.get_prediction(sample)
            else:
                return self.right.get_prediction(sample)


    def split(self):
        if (self.cur_depth < self.max_depth) and (self.n_samples >= self.min_leaf_size):
            self.best_feature, self.best_split = self._calc_best_split()

            # Node is a leaf if no splits were found
            if self.best_split is None:
                self.is_leaf = True
                return 

            mask = (self.X[:, self.best_feature] <= self.best_split)

            Xleft  = self.X[mask,:]
            yleft  = self.y[mask]
            Xright = self.X[~mask,:]
            yright = self.y[~mask]

            self.left = RegressionNode(
                Xleft,
                yleft,
                self.cur_depth+1,
                self.max_depth,
                self.min_leaf_size)
            self.left.split()

            self.right = RegressionNode(
                Xright,
                yright,
                self.cur_depth+1,
                self.max_depth,
                self.min_leaf_size)
            self.right.split()
        
        else:
            self.is_leaf = True
