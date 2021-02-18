import numpy as np
from scipy.stats import t


class LinReg:

    def __init__(self):
        # Initialize weights and Results
        # Indicate that model has not been fitted yet
        self.__betas = None
        self.__fitted = False
        self.__results = {}

    def __repr__(self):
        return "LinReg()"

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


class LogReg:

    def __init__(self):
        pass
    
    def fit(self, X, y):

        # Get train data shape and preprocess
        n_observations, n_features = X.shape
        X_appended = np.hstack((np.ones((n_observations, 1)), X))

        # Initialize weights
        self.__betas = np.zeros((n_features + 1, 1))

        # Train model
        self.trainMle(X_appended, y)


    # Calculate new beta
    def trainMle(self, X, y):

        betaDifference = np.inf
        
        while np.abs(betaDifference) > 1e-3:
            
            # Update probability estimates and W matrix
            probabilities = self.calculate_probabilities(X)
            dubMatrix = (probabilities * (1 - probabilities)) * np.identity(y.size)

            # Calculate new beta
            betaStep = np.linalg.inv(X.T @ dubMatrix @ X) @ X.T @ (y[:, None] - probabilities)
            self.__betas += betaStep

            # Calculate step magnitude
            betaDifference = np.sum(betaStep)/np.sum(self.__betas)
            print(f"Step magnitude relative to coefficient vector: {betaDifference}")

    def calculate_probabilities(self, X):
        return 1 / (1 + np.exp(-X @ self.__betas))

    def score(self, X, y=None):
        
        # Calculate probabilities
        X_appended = np.hstack((np.ones((X.shape[0], 1)), X))
        probabilities = self.calculate_probabilities(X_appended)

        if y.all() != None:
            
            # Assign class to values above/below threshold
            mask = (probabilities >= 0.5)
            predictions = np.zeros_like(probabilities)
            predictions[mask] = 1
            return {'predictions': predictions,
                    'probabilities': probabilities,
                    'accuracy': np.sum(predictions == y[:, None]) / y.size}
        
        return probabilities

    def coefficients(self):
        return self.__betas.flatten()


class KNN:

    def __init__(self):
        # Initialize classifier, set status to untrained
        self.__trained = False

    def fit(self, X, y):
        """
        Fit K-nearest-neighbor classifier to data.
        X is a 2D numpy array of numerical values, preferably normalized
        y is a 1D numpy array of target values
        Training is simply memorizing data
        """
        self.__X = X
        self.__y = y
        self.__trained = True

    def predict(self, X, k):
        """
        Prediction using the K-nearest-classifier
        X is a 2D numpy array of numerical values
        k is the number of neighbors voting for the prediction
        returns an array of predictions
        """
        # Sanity check
        if self.__trained == False:
            return 'Model not trained yet'
        
        # Reshape data to 2D array if possible
        if len(X.shape) == 1:
            X = X[None, :]
                
        # Compute the distances between all given points and all training points
        distances = np.sum((X[:, None, :] - self.__X[None, :, :]) ** 2, axis = -1)
        
        # Find the K nearest neighbors
        k_nearest_neighbors = np.argsort(distances, axis=-1)[:,:k]

        # Run over all columns and find the most common neighbor in each, then return his label
        n_observations = X.shape[0]
        idx = [np.argmax(np.bincount(k_nearest_neighbors[i])) for i in range(n_observations)]
        return self.__y[idx]