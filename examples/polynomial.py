import sys, path
_directory = path.Path(__file__).abspath()
print(_directory)
sys.path.append(_directory.parent.parent)
from clf.knn_regressor import *

file = '../datasets/polynomial200.csv'
filedata = load_data(file)

test_size = 0.50
entries = len(filedata.values)
test_set_size = round(entries * test_size)

train_set = filedata.iloc[test_set_size:, ].values
test_set = filedata.iloc[:test_set_size, ].values
print(f'Train set entries: {len(train_set)}\nTest set entries: {len(test_set)}')

X, y = filedata.iloc[:, 0], filedata.iloc[:, 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# Plot training and test set data
plt.scatter(X_train, y_train, s=24, edgecolor='k', color='yellow', label='Training set')
plt.scatter(X_test, y_test, s=24, edgecolor='k', color='magenta', label='Test set')
plt.legend()
plt.title('Training and test set data')
# plt.show()

fig, axs = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("KNN Regression")

# print(train_set); print("\n"); print(test_set)

K = [1,3,5,7,9,11]

current_row = 0
cols_per_row = int(len(K) / 2)
subfigure_idx = 0


for k in K:
    knn = KNNRegressor(k=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    # X_ = np.arange(0, 25, 0.25)
    # Y_pred = knn.predict(X_)
    cost = calc_cost(X_test, y_pred)
    mse = calc_mse(X_test, y_pred)

    if subfigure_idx >= cols_per_row:
        current_row += 1
        subfigure_idx = 0

    ax = axs[current_row, subfigure_idx]
    
    plot_regression(ax, X_test, y_test, y_pred, k, mse=mse, cost=cost)
    subfigure_idx += 1

    #print(f"y_pred:\n{y_pred}")

plt.show()
