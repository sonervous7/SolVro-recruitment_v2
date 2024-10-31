import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


class Clusterer:
    def __init__(self, data, n_clusters=7):
        """
        Initializes the Clusterer object.

        Parameters:
        - data (DataFrame or ndarray): The data to be clustered (preprocessed and PCA-reduced).
        - n_clusters (int): The number of clusters. Default is 7.
        """
        self.data = data
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.labels = None
        self.silhouette_score = None

    def cluster_data(self):
        """
        Performs data clustering and stores the cluster labels.

        Returns:
        - labels (ndarray): Cluster labels for each data point.
        """
        self.labels = self.model.fit_predict(self.data)
        return self.labels

    def calculate_silhouette_score(self):
        """
        Calculates the silhouette score for the clustering.

        Returns:
        - score (float): Silhouette score value.

        Raises:
        - ValueError: If clustering has not been performed.
        """
        if self.labels is None:
            raise ValueError("Klasteryzacja nie została jeszcze wykonana. Użyj metody cluster_data().")
        self.silhouette_score = silhouette_score(self.data, self.labels)
        return self.silhouette_score

    def print_davies_bouldin_score(self, lables, n_clusters):
        """
        Calculates and displays the Davies-Bouldin index for the given cluster labels.

        Parameters:
        - labels (ndarray): Cluster labels for each data point.
        - n_clusters (int): The number of clusters.

        Prints:
        - Davies-Bouldin index value for the specified number of clusters.
        """

        dbi_score = davies_bouldin_score(self.data, lables)
        print(f"Indeks Daviesa-Bouldina dla {n_clusters} klastrów: {dbi_score}")

    def plot_elbow_method(self, max_clusters=10):
        """
        Plots the elbow method for a range of clusters from 1 to max_clusters.

        Parameters:
        - max_clusters (int): Maximum number of clusters to consider. Default is 10.

        Displays:
        - Elbow plot showing inertia (sum of squared distances to the closest cluster center)
          as a function of the number of clusters.
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
        plt.title("Metoda łokcia")
        plt.xticks(range_clusters)
        plt.grid(True)
        plt.show()
