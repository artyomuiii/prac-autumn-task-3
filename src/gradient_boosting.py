import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


class GradientBoostingMSE:
    
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use alpha * learning_rate instead of alpha

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        if feature_subsample_size is None:
            feature_subsample_size = 1/3
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        
        # trees_parameters
        self.random_state = 42
        self.continuity = False
        
        for param_name, value in trees_parameters.items():
            if param_name == "random_state":
                self.random_state = value
            if param_name == "continuity":
                self.continuity = value
        
        self.b_list = None
        self.alpha_list = None

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """
        self.b_list = []
        self.alpha_list = []
        f = np.zeros_like(y)
        np.random.seed(self.random_state)
        
        for t in range(self.n_estimators):
            y_grad = 2 * (f - y)
            
            b = DecisionTreeRegressor(max_depth=self.max_depth, max_features=self.feature_subsample_size,
                                      random_state=self.random_state)
            b.fit(X, -y_grad)
            y_pred = b.predict(X)
            
            def fun(x):
                return mean_squared_error(f + x * y_pred, y)
            alpha = minimize_scalar(fun).x
            
            if alpha >= 0:
                self.b_list.append(b)
                self.alpha_list.append(alpha)
            
            f += alpha * self.learning_rate * y_pred

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        predict = np.zeros(X.shape[0])
        for b, alpha in zip(self.b_list, self.alpha_list):
            predict += self.learning_rate * alpha * b.predict(X)
        
        return predict
    
    def predict_y(self, X, y):
        rmse_list = []
        predict = np.zeros(X.shape[0])
        for b, alpha in zip(self.b_list, self.alpha_list):
            predict += self.learning_rate * alpha * b.predict(X)
            rmse_list.append(mean_squared_error(y, predict, squared=False))
        
        return rmse_list