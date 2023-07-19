from ml_utils import *

class KNNClassifier:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        """ Assign training data of X and y as members """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        """ Predict based on passed data X """
        if len(X) > 2:
            y_pred = []
            for x in X:
                _y_pred = self._clf(x)
                y_pred.append(_y_pred)
        else: 
            y_pred = self._clf(X)
        return np.array(y_pred)

    def _clf(self, x):
        distances = get_euclidean_distances(x, self.X_train)
        indices = sort_list(distances, self.k)
        neighbors = [self.y_train[i] for i in indices]
        occurences = get_element_occurences(neighbors)
        prediction = get_most_common_element(occurences)
        return prediction
