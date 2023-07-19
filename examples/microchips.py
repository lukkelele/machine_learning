import sys, path
_directory = path.Path(__file__).abspath()
print(_directory)
sys.path.append(_directory.parent.parent)
from clf.knn_clf import *

print("Starting microchips.py")

file = '../datasets/microchips.csv'
features = [0, 1] # X
label = 2         # y

filedata = load_data(file)
X_train, y_train = split_to_separate_datasets(filedata, features, label)
# print(filedata.info(verbose=True))

fig, axs = plt.subplots(2, 2, figsize=(20, 12))
fig.suptitle("Classification using KNN")
current_row = 0

# Unknown chips
chip1 = [-0.3,  1.0] # Fail
chip2 = [-0.5, -0.1] # OK
chip3 = [ 0.6,  0.0] # OK

X_test = np.array([chip1, chip2, chip3])

# List of k's to iterate 
K = [1, 3, 5, 7]
results = []

cols_per_row = int(len(K) / 2) # 4 / 2 = 2
subfigure_idx = 0

for k in K:
    # Create classifier and fit the training data
    knn = KNNClassifier(k=k)
    knn.fit(X_train, y_train)
    y_train_pred = knn.predict(X_train)
    accuracy_score = calc_accuracy_score(y_train, y_train_pred)
    training_error = 1 - accuracy_score
    #print(f'Accuracy score: {accuracy_score:.3f}\nTraining error: {training_error:.3f}')

    # Predict values for unknown chips
    pred_chip1 = knn.predict(chip1)
    pred_chip2 = knn.predict(chip2)
    pred_chip3 = knn.predict(chip3)
    chip_results = np.array([pred_chip1, pred_chip2, pred_chip3])

    # plt.subplot(current_row, cols_per_row, subfigure_idx)
    # plot_data_classifier(X_train, y_train, y_clf, k) # Plot without decision boundary
    if subfigure_idx >= cols_per_row:
        current_row += 1
        subfigure_idx = 0

    # print(f"subfig idx: {subfigure_idx}\ncurrent row: {current_row}")
    ax = axs[current_row, subfigure_idx]
    plot_decision_boundary(ax, X_train, y_train, y_train_pred, knn, k)
    subfigure_idx += 1

    # Print results
    print(f"====================================\n| K: {k}")
    chip_idx = 1
    for chip_result in chip_results:
        # print(f'Chip result == {chip_result}')
        print(f"""|\t Chip {chip_idx} - {X_test[chip_idx - 1]}: {"OK!" if chip_result == 1 else "Failed..."}""")
        chip_idx += 1
    results.append(chip_results)
print("==================================== ")

plt.show()

print("Exiting...")

