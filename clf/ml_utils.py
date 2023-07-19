from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Ignore scalar divide warning 
warnings.filterwarnings("ignore", category=RuntimeWarning)

def load_data(filename):
    """ Load data from CSV file """
    filedata = pd.read_csv(filename, header=None)
    return filedata

# def split_data(data, features, label):
def split_to_separate_datasets(data, features, label):
    X = np.array(data.iloc[:, features].values, dtype='float64')
    y = np.array(data.iloc[:, label].values, dtype='float64')
    return X, y

def sort_list(lst: list, k: int):
    """ Sort list in ascending order and return k elements """
    return np.argsort(lst)[:k]

def standardize(X_train, X_test):
    """ Standardize datasets """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std

    return X_train_std, X_test_std

def train_test_split(X, y, test_size=0.25):
    """ Training/Test split to replicate sklearn's train_test_split function """
    X_length = len(X)
    indices = np.arange(X_length)
    np.random.shuffle(indices)

    # Calculate the split index 
    idx = int(X_length * (1 - test_size))

    # Training and test set
    X_train = np.array(X[indices[ :idx]], dtype='float64')
    y_train = np.array(y[indices[ :idx]], dtype='float64')
    X_test  = np.array(X[indices[idx: ]], dtype='float64')
    y_test  = np.array(y[indices[idx: ]], dtype='float64')

    return X_train, X_test, y_train, y_test

def get_element_occurences(lst):
    count_dict = {}
    for element in lst:
        if element in count_dict:
            count_dict[element] += 1
        else:
            count_dict[element] = 1
    return count_dict

def get_most_common_element(elements):
    """ 
    Get most common element, use this together with 'get_element_occurences'
    This will return the element that is present the most in the passed list
    """
    most_common_element = max(elements, key=elements.get) # pyright: ignore
    return most_common_element

def compute_euclidean_distance(x1, x2):
    """ Compute the euclidean distance between x1 and x2 """
    return np.sqrt(np.sum(( x1 - x2 ) ** 2))

def get_euclidean_distances(x, X):
    """ Get the euclidean distances for passed x and return the distances """
    distances = []
    for x_entry in X:
        euclidean_distance = compute_euclidean_distance(x, x_entry)
        # Check so the passed entry and checked entry isn't the same 
        if np.array_equal(x, x_entry) == False:
            distances.append(euclidean_distance)
    return distances

def calculate_accuracy(y_pred, y_test):
    """ Get accuracy results based on prediction vs test data """
    return np.mean(y_pred == y_test)

def plot_data_classifier(X, y, y_pred, k):
    plt.scatter(X[y_pred == y, 0], X[y_pred == y, 1], c='green', label='Correct')
    plt.scatter(X[y_pred != y, 0], X[y_pred != y, 1], c='red',   label='Wrong')
    # plt.title("Result with k: ", k)
    plt.legend()
    # plt.show()

def get_colormap(color: str):
    """ Get colormap based on string """
    color = color.lower()
    if color == 'light':
        return ListedColormap(['#ffaaaa', '#aaffaa', '#aaaaff'])
    elif color == 'rgb':  # Red Green Blue
        return ListedColormap(['#ff0000', '#00ff00', '#0000ff'])           
    elif color == 'pyc':  # Purple Yellow Cyan
        return ListedColormap(['#800080', '#ffff00', '#00ffff'])            
    elif color == 'opgb': # Orange Pink Gray Brown
        return ListedColormap(['#ffa500', '#ffc0cb', '#808080', '#a52a2a']) 

def plot_decision_boundary(ax, X, y, y_pred, clf, k, h=0.035):
    """ Plot decision boundary for KNN """
    cmap_light = get_colormap('light')
    cmap_redgreenblue = get_colormap('rgb')

    accuracy_score = calc_accuracy_score(y, y_pred)
    training_error = 1 - accuracy_score
    title = f"K: {k}, training error: {training_error:.3f}"

    axis_offset = 0.25
    x_min, x_max = X[:, 0].min() - axis_offset, X[:, 0].max() + axis_offset
    y_min, y_max = X[:, 1].min() - axis_offset, X[:, 1].max() + axis_offset
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    ax.contourf(xx, yy, z, cmap=cmap_light, alpha=0.9)
    ax.scatter(X[y_pred == y, 0], X[y_pred == y, 1], c=y[y_pred == y], cmap=cmap_redgreenblue, marker='o', edgecolor='k', label='Correct')
    ax.scatter(X[y_pred != y, 0], X[y_pred != y, 1], c=y[y_pred != y], cmap=cmap_redgreenblue, marker='X', edgecolor='k', label='Wrong')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(title)
    ax.legend()

def plot_regression(ax, X, y, y_pred, k, mse=-1, cost=-1):
    """ Plot regression curve """
    sorted_indices = np.argsort(X)
    X_sorted = X[sorted_indices] 
    y_sorted = y[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    title = f"K: {k}"
    if mse != -1:
        title += f", mse: {mse:.2f}"
    if cost != -1:
        title += f", cost: {cost:.2f}"

    # print('X_sorted:\n', X_sorted); print('y_pred_sorted:\n', y_pred_sorted)
    ax.scatter(X_sorted, y_sorted, s=24, edgecolor='k', color='green', label='True')
    ax.plot(X_sorted, y_pred_sorted, color='red', label='Predicted')
    # ax.plot(X_sorted, y_sorted, color='red', label='Predicted')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend()

def calc_accuracy_score(y, y_pred):
    """ Calculate the accuracy score """
    correct_predictions = np.sum(y == y_pred)
    acc = correct_predictions / len(y)
    return acc

def extend_matrix(X, ones=False):
    """ Extend matrix """
    if ones:
        Xe = np.c_[np.ones(X.shape[0]), X]
    else:
        Xe = np.c_[np.zeros(X.shape[0]), X]
    return Xe

def calc_beta(X, y):
    """ Calculate beta """
    Xe = extend_matrix(X, ones=True) # Extend with ones
    # B = np.linalg.inv(Xe.T.dot(Xe)).dot(Xe.T).dot(y)
    beta = np.linalg.inv(Xe.T @ Xe) @ Xe.T @ y
    return beta 

def calc_mse(y, y_pred):
    """ MSE, Mean Squared Error """
    s = (y - y_pred) ** 2
    mse = np.mean(s)
    return mse

def calc_cost(X, y):
    """ Cost function """
    Xe = extend_matrix(X, ones=True)
    n = len(y)

    beta = calc_beta(X, y)
    # y_pred = Xe @ beta
    # j = np.dot(Xe, beta) - y
    # J = (j.T.dot(j)) / n
    j = (Xe @ beta) - y
    J = (j.T @ j) / n
    return J

