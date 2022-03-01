import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


class RandomForestMSE:
    
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        if feature_subsample_size is None:
            feature_subsample_size = 1/3
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        
        # trees_parameters
        self.random_state = 42
        self.eps_val = None
        self.continuity = False
        
        for param_name, value in trees_parameters.items():
            if param_name == "random_state":
                self.random_state = value
            if param_name == "eps_val":
                self.eps_val = value
            if param_name == "continuity":
                self.continuity = value
                
        self.b_list = None
        self.continuity_list = []

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects

        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects
        """
        b_list = []
        np.random.seed(self.random_state)
        for t in range(self.n_estimators):
            # с возвращением
            u_mask = np.random.randint(X.shape[0], size=X.shape[0])
            U = X[u_mask]
            y_u = y[u_mask]
            
            b = DecisionTreeRegressor(max_depth=self.max_depth, max_features=self.feature_subsample_size, 
                                      random_state=self.random_state)
            b.fit(U, y_u)
            b_list.append(b)
            
            if (X_val != None) and (y_val != None) and (self.eps_val != None):
                pred_val = b.predict(X_val)
                if mean_squared_error(y_val, pred_val) > self.eps_val:
                    b_list.pop()
            
                
        self.b_list = b_list

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        T_continuity = 0
        T = len(self.b_list)
        if T == 0:
            raise TypeError("RandomForestMSE predict: T == 0")
        
        predict = np.zeros(X.shape[0])
        for b in self.b_list:
            predict += b.predict(X)
            if self.continuity:
                T_continuity += 1
                self.continuity_list.append(predict / T_continuity)
        predict = predict / T
        
        return predict
    
    def predict_y(self, X, y):
        rmse_list = []
        T_continuity = 0
        predict = np.zeros(X.shape[0])
        for b in self.b_list:
            predict += b.predict(X)
            T_continuity += 1
            rmse_list.append(mean_squared_error(y, predict / T_continuity, squared=False))
        return rmse_list