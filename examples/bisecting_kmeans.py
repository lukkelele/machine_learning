from ml_utils import *
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np


def calculate_sse(X, model):
    """ Calculate SSE with the help of a model """
    distance = 0
    labels = model.predict(X)
    centroids = model.cluster_centers_
    for i in range(len(X)):
        distance += np.sum((X[i] - centroids[labels[i]]) ** 2)
    return distance

def calculate_cluster_sse(cluster_points):
    """ Calculate SSE on a specific cluster """
    centroid = cluster_points.mean(axis=0)
    distances = np.linalg.norm(cluster_points - centroid, axis=1)
    sse = (distances ** 2).sum()
    return sse

def select_next_cluster(clusters):
    """ Select next cluster with highest SSE to reduce the total variance """
    clusters_sse = []
    for cluster in clusters:
        cluster_sse = calculate_cluster_sse(cluster[0]) # (data points, indices)
        clusters_sse.append(cluster_sse)
    cluster_idx = np.argmax(clusters_sse)
    return clusters.pop(cluster_idx)

def get_clusters(selected_cluster, labels):
    """ Split selected cluster """
    cluster_points = selected_cluster[0]
    cluster_indices= selected_cluster[1]

    # Cluster 1
    cluster1_points = cluster_points[labels == 0] 
    cluster1_indices = cluster_indices[labels == 0]
    # Cluster 2 
    cluster2_points = cluster_points[labels == 1]
    cluster2_indices = cluster_indices[labels == 1]

    # Create tuple for new clusters with data points and indices
    cluster1 = (cluster1_points, cluster1_indices)
    cluster2 = (cluster2_points, cluster2_indices)
    return cluster1, cluster2
    

def bisecting_kmeans(X, k, iterations=5):
    """
    Bisecting KMeans

    X: data 
    k: cluster size
    iterations: times to apply model to chosen clusters to find best fit
    """
    kmeans_clf = KMeans(n_clusters=2, n_init='auto').fit(X)
    indices = np.arange(X.shape[0])
    clusters = [(X, indices)]

    while len(clusters) < k:
        # Select the cluster with the highest SSE
        selected_cluster = select_next_cluster(clusters)
        # Assign returned tuple (points, indices) to respective variable
        cluster_points = selected_cluster[0]
        cluster_indices= selected_cluster[1]

        # Initiate variables that will be updated with the 'best' values
        best_clf = None
        best_cluster_sse = np.inf # to update first iteration correctly
        best_cluster1 = None
        best_cluster2 = None

        print("> Cluster SSE")
        for i in range(iterations):
            kmeans_clf = KMeans(n_clusters=2, n_init='auto').fit(cluster_points)
            new_labels = kmeans_clf.labels_

            # Split the selected cluster
            cluster1, cluster2 = get_clusters(selected_cluster, new_labels)

            # Calculate SSE for both newly created clusters
            cluster1_sse = calculate_cluster_sse(cluster1[0])
            cluster2_sse = calculate_cluster_sse(cluster2[0]) 
            new_clusters_sse = cluster1_sse + cluster2_sse
            print(f"Iteration {i+1} | 1: {cluster1_sse:.5f} | 2: {cluster2_sse:.5f} | Sum: {new_clusters_sse:.5f}")

            # If the SSE is lower than previously set, update with new data
            if new_clusters_sse < best_cluster_sse:
                best_clf = kmeans_clf              
                best_cluster_sse = new_clusters_sse # Lowest combined SSE
                best_cluster1 = cluster1 
                best_cluster2 = cluster2

        print(f"> Lowest SSE: {best_cluster_sse:.5f}\n")
        clusters.append(best_cluster1) # pyright: ignore
        clusters.append(best_cluster2) # pyright: ignore

    labels = np.zeros(X.shape[0])
    # Iterate through clusters and assign clusters labels for k clusters
    # Cluster (data points, indices)
    for i, cluster in enumerate(clusters):
        indices = cluster[1] # select indices
        labels[indices] = i

    return labels

def plot_clusters(X, labels, k, ax):
    """ Plot clusters """
    # Scatter plot data points with different colors
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=75, edgecolor="k", cmap="magma")

    centroids = []
    for cluster_idx in np.unique(labels):
        cluster_points = X[labels == cluster_idx]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)
    ax.scatter(centroids[:, 0], centroids[:, 1], c="black", s=920, alpha=0.55, marker='o')

def run_and_plot(X, cluster_sizes):
    """ 'Main' function for testing the algorithm """
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    title = f"Bisecting KMeans with cluster sizes "
    for cluster_size in cluster_sizes:
        title += str(cluster_size) + ", "
    # Remove last 2 characters from title string to remove ', ' after last appended size in title
    title = title[:-2] 
    fig.suptitle(title)

    for i, k in enumerate(cluster_sizes):
        ax = axs[i // 2, i % 2]
        ax.set_title(f"Clusters: {k}")
        labels = bisecting_kmeans(X, k)
        plot_clusters(X, labels, k, ax)

    plt.tight_layout()
    plt.show()


print('Starting bkmeans.py\n')
dataset_path = "./datasets/microchips.csv"
filedata = load_data(dataset_path)
X, y = split_to_separate_datasets(filedata, [0,1], 1)
run_and_plot(X, [2, 3, 4, 5])
print('\nExiting bkmeans.py')
