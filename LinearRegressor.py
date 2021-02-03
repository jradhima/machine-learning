# Implementation of a simple linear regressor object
# Uses numpy

import numpy as np
from scipy.stats import t

class LinearRegression():

    def __init__(self):
        # Initialize weights and yhat to None
        # Indicate that model has not been fitted yet
        self.betas = None
        self.fitted = False

    def fit(self, X, y, verbose=False):
        # Find data dimension, introduce intercept to design matrix
        n_observations, n_features = X.shape
        X_appended = np.hstack((np.ones((n_observations, 1)), X))

        # Compute pseudo-inverse, if possible
        if (n_features > n_observations):
            print("""
            Design matrix is unidentifiable, less observations than features\n
            OR data not given in [n_observations, n_features] format.\n
            Try again with correct data or data format. Returning "None".
            """)
            return None
        try:
            # Compute Moore-Penrose Pseudo-inverse
            mini_inverse = np.linalg.inv(X_appended.T @ X_appended)
            pseudo_inverse = mini_inverse @ X_appended.T
        except np.linalg.LinAlgError:
            # Not invertible. Skip this one.
            print('Design matrix is not invertible.\nReturning None.')
            return None

        # Compute coefficients and change model status
        self.betas = (pseudo_inverse @ y[:, None])
        self.fitted = True

        # Print model MSE on training dataset
        y_hat = (X_appended @ self.betas)
        #print(y_hat.T.shape, y.shape)
        sum_squares = np.sum((y_hat.T - y) ** 2)
        deg_free = n_observations - n_features - 1
        print(f"Model trained.\nUnbiased MSE is {sum_squares / deg_free}.")
        print(f"Biased MSE is {sum_squares / n_observations}.")

        # If asked, produce statistical report
        if verbose == True:
            # Compute sample variance
            var_hat = sum_squares / deg_free
            # Compute betas' variance and t-scores
            betas_cov = mini_inverse * var_hat
            t_scores = self.betas.T / np.sqrt(np.diag(betas_cov))
            p_values = lambda x, d: t.sf(np.abs(x), d) * 2
            p_vals = p_values(t_scores, deg_free)
            
            # Print results
            print(f"The regression coefficients are {self.betas.T[0]}.")
            print(f"Their standard deviation is {np.sqrt(np.diag(betas_cov))}.")
            print(f"The corresponding p-values are {p_vals[0]}.")




    def predict(self, X, y=None):
        """
        Input is a numpy 2-D array, returns predictions, and optionally MSE
        """ 
        # Check to see if model has been trained
        if self.fitted == False:
            print('Model not fitted yet.\nReturning None.')
            return None
        
        # Append matrix, create design matrix
        X_appended = np.hstack((np.ones((X.shape[0], 1)), X))
        # Compute predictions
        predictions = X_appended @ self.betas

        # If y is given, calculate and return MSE
        if not y.any() == None:
            sum_squares = np.sum((predictions.T - y) ** 2)
            print(f"Prediction MSE is {sum_squares / X_appended.shape[0]}.")
        return predictions




        
    