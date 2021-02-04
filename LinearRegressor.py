# Implementation of a simple linear regressor object
# Uses numpy

import numpy as np
from scipy.stats import t

class LinearRegression():

    def __init__(self):
        # Initialize weights and Results
        # Indicate that model has not been fitted yet
        self.__betas = None
        self.__fitted = False
        self.__results = {}

    def __repr__(self):
        return "LinearRegression()"

    def __str__(self):
        return "Linear regressor object with methods 'fit' and 'predict'."

    def fit(self, X, y, statistical_report=False, verbose=False):

        """
        Fit linear regression model to dataset.

        X is a 2-D numpy array with dimensions [n_observations, n_features].

        y is a 1-D numpy array with length n_observations.

        statistical_report is boolean, True for p_values, coefficient standard deviation and more.
        """

        # Find data dimension, introduce intercept to design matrix
        n_observations, n_features = X.shape
        X_appended = np.hstack((np.ones((n_observations, 1)), X))

        # Compute pseudo-inverse, if possible
        if (n_features > n_observations):
            print("""
            Design matrix is unidentifiable, less observations than features
            OR data not given in [n_observations, n_features] format.
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
        self.__betas = (pseudo_inverse @ y[:, None])
        self.__fitted = True

        # Compute fitted values, sum of squared errors and degrees of freedom
        y_hat = (X_appended @ self.__betas)
        sum_squares = np.sum((y_hat.T - y) ** 2)
        deg_free = n_observations - n_features - 1

        # Print results
        if verbose == True:
            print(f"Model trained.\nUnbiased MSE is {sum_squares / deg_free}.")
            print(f"Biased MSE is {sum_squares / n_observations}.")

        # Update results
        self.__results['fitted_values'] = y_hat.T[0]
        self.__results['biased_mse'] = sum_squares / n_observations
        self.__results['unbiased_mse'] = sum_squares / deg_free

        # Optionaly, produce statistical report
        if statistical_report == True:
            # Compute sample variance
            var_hat = sum_squares / deg_free

            # Compute betas' variance
            betas_cov = mini_inverse * var_hat
            betas_std = np.sqrt(np.diag(betas_cov))

            # Compute betas t-scores and p-values
            t_scores = self.__betas.T / betas_std
            p_value = lambda x, d: t.sf(np.abs(x), d) * 2
            p_vals = p_value(t_scores, deg_free)
            
            # Print results
            if verbose == True:
                print(f"The regression coefficients are {self.__betas.T[0]}.")
                print(f"Their standard deviation is {betas_std}.")
                print(f"The corresponding p-values are {p_vals[0]}.")

            # Update results
            self.__results['coefficients_std'] = betas_std
            self.__results['coefficients_pvalues'] = p_vals[0]
            self.__results['degrees_freedom'] = deg_free

    @property
    def coefficients(self):
        return self.__betas.T[0][:]

    @property
    def report(self):
        return self.__results

    @property
    def items(self):
        return self.__results.keys()
            
    def predict(self, X, y=None):
        """
        Input is a numpy 2-D array, returns predictions and MSE

        predictions (, mse) = model.predict(X (,y))
        """ 
        # Check to see if model has been trained
        if self.__fitted == False:
            print('Model not fitted yet.\nReturning None.')
            return None
        
        # Append matrix, create design matrix
        X_appended = np.hstack((np.ones((X.shape[0], 1)), X))

        # Compute predictions
        predictions = X_appended @ self.__betas

        # If y is given, calculate and return MSE
        if not y.any() == None:
            mse = np.sum((predictions.T - y) ** 2) / X_appended.shape[0]
            print(f"Prediction MSE is {mse}.")
            return predictions.T, mse
        return predictions.T