# Implementation of a K-nearest-neighbors classifier

import numpy as np

class knn:

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