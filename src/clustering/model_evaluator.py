import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score


class ModelEvaluator:
    def __init__(self, data):
        """
        Initializes a ModelEvaluator object with data for evaluation.

        Parameters:
        - data (DataFrame or ndarray): Data for clustering (scaled and post-PCA).
        """
        self.data = data

    def plot_elbow_method(self, max_clusters=10):
        """
        Plots the elbow method chart for a range of clusters from 1 to max_clusters.

        Parameters:
        - max_clusters (int): Maximum number of clusters to consider. Default is 10.
        """
        inertia = []
        range_clusters = range(1, max_clusters + 1)
        for k in range_clusters:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.data)
            inertia.append(kmeans.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(range_clusters, inertia, marker="o", linestyle="--")
        plt.xlabel("Liczba klastrów")
        plt.ylabel("Inercja")
        plt.title("Metoda łokcia dla KMeans")
        plt.xticks(range_clusters)
        plt.grid(True)
        plt.show()

    def plot_silhouette_scores(self, max_clusters=10):
        """
        Plots the silhouette coefficient for a range of clusters from 2 to max_clusters.

        Parameters:
        - max_clusters (int): Maximum number of clusters to consider. Default is 10.
        """
        silhouette_scores = []
        range_clusters = range(2, max_clusters + 1)
        for k in range_clusters:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(self.data)
            score = silhouette_score(self.data, labels)
            silhouette_scores.append(score)

        plt.figure(figsize=(8, 5))
        plt.plot(range_clusters, silhouette_scores, marker="o", linestyle="--")
        plt.xlabel("Liczba klastrów")
        plt.ylabel("Współczynnik silhouette")
        plt.title("Współczynnik silhouette dla różnych liczby klastrów")
        plt.xticks(range_clusters)
        plt.grid(True)
        plt.show()

    def plot_davies_bouldin_scores(self, max_clusters=10):
        """
        Creates a plot of the Davies-Bouldin index for a range of clusters from 2 to max_clusters.

        Parameters:
        - max_clusters (int): Maximum number of clusters to consider. Default is 10.
        """
        dbi_scores = []
        range_clusters = range(2, max_clusters + 1)
        for k in range_clusters:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(self.data)
            score = davies_bouldin_score(self.data, labels)
            dbi_scores.append(score)

        plt.figure(figsize=(8, 5))
        plt.plot(range_clusters, dbi_scores, marker="o", linestyle="--")
        plt.xlabel("Liczba klastrów")
        plt.ylabel("Indeks Daviesa-Bouldina")
        plt.title("Indeks Daviesa-Bouldina dla różnych liczby klastrów")
        plt.xticks(range_clusters)
        plt.grid(True)
        plt.show()

    def plot_variance_explained(self, data_scaled, max_components=5):
        """
        Plots the cumulative explained variance by PCA components.

        Parameters:
        - data_scaled (DataFrame or ndarray): Scaled input data.
        - max_components (int): Maximum number of PCA components to consider. Default is 5.
        """
        pca = PCA(n_components=max_components)
        pca.fit(data_scaled)
        explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, max_components + 1), explained_variance_ratio, marker="o", linestyle="--")
        plt.xlabel("Liczba komponentów PCA")
        plt.ylabel("Skumulowana wariancja wyjaśniona")
        plt.title("Wariancja wyjaśniona przez PCA")
        plt.xticks(range(1, max_components + 1))
        plt.grid(True)
        plt.show()

    def plot_kmeans_clusters(self, data_reduced, labels):
        """
        Visualizes KMeans clusters on reduced-dimensionality data.

        Parameters:
        - data_reduced (ndarray): Data post-dimensionality reduction (e.g., PCA).
        - labels (ndarray): KMeans cluster labels.
        """
        plt.figure(figsize=(10, 6))
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap("viridis", len(unique_labels))

        for k in unique_labels:
            cluster_data = data_reduced[labels == k]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=50, color=colors(k), label=f"Klaster {k}")

        plt.xlabel("Pierwsza składowa PCA")
        plt.ylabel("Druga składowa PCA")
        plt.title("Wizualizacja klastrów KMeans")
        plt.legend()
        plt.show()

    def print_silhouette_score(self, labels):
        """
        Calculates and displays the Davies-Bouldin index for the provided cluster labels.

        Parameters:
        - labels (ndarray): Cluster labels.
        """
        score = silhouette_score(self.data, labels)
        print(f"Współczynnik silhouette: {score:.4f}")
