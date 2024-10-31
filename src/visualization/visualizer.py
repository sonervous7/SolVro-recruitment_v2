import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    def __init__(self, data):
        """
        Initializes the Visualizer object with data for visualization.

        Parameters:
        - data (DataFrame): Data with assigned clusters (column 'cluster').
        """
        self.data = data.copy()

    def plot_cluster_distribution_by_glass(self):
        """
        Plots the distribution of clusters by glass type ('glass').
        """
        plt.figure(figsize=(12, 6))
        sns.countplot(data=self.data, x="glass", hue="cluster", palette="viridis")
        plt.title("Rozkład klastrów względem typu szkła")
        plt.xlabel("Typ szkła")
        plt.ylabel("Liczba koktajli")
        plt.legend(title="Klaster", loc="upper right")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_cluster_distribution_by_category(self):
        """
        Plots the distribution of clusters by cocktail category ('category').
        """
        plt.figure(figsize=(12, 6))
        sns.countplot(data=self.data, x="category", hue="cluster", palette="viridis")
        plt.title("Rozkład klastrów względem kategorii koktajlu")
        plt.xlabel("Kategoria")
        plt.ylabel("Liczba koktajli")
        plt.legend(title="Klaster", loc="upper right")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_estimated_abv_in_clusters(self):
        """
        Plots the distribution of estimated alcohol content (estimated_abv) within clusters.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.data, x="cluster", y="estimated_abv", palette="viridis")
        plt.title("Rozkład estimated_abv w klastrach")
        plt.xlabel("Klaster")
        plt.ylabel("Szacowana zawartość alkoholu (%)")
        plt.show()

    def plot_alcoholic_ratio_in_clusters(self):
        """
        Plots the distribution of the alcoholic ingredient ratio (alcoholic_ratio) within clusters.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.data, x="cluster", y="alcoholic_ratio", palette="viridis")
        plt.title("Rozkład alcoholic_ratio w klastrach")
        plt.xlabel("Klaster")
        plt.ylabel("Udział składników alkoholowych")
        plt.show()

    def plot_num_ingredients_in_clusters(self):
        """
        Plots the total number of ingredients in cocktails within clusters.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.data, x="cluster", y="num_total_ingredients", palette="viridis")
        plt.title("Rozkład liczby składników w klastrach")
        plt.xlabel("Klaster")
        plt.ylabel("Liczba składników")
        plt.show()

    def plot_num_alcoholic_ingredients_in_clusters(self):
        """
        Plots the number of alcoholic ingredients in cocktails within clusters.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.data, x="cluster", y="num_alcoholic_ingredients", palette="viridis")
        plt.title("Rozkład liczby składników alkoholowych w klastrach")
        plt.xlabel("Klaster")
        plt.ylabel("Liczba składników alkoholowych")
        plt.show()

    def plot_num_non_alcoholic_ingredients_in_clusters(self):
        """
        Plots the number of non-alcoholic ingredients in cocktails within clusters.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.data, x="cluster", y="num_non_alcoholic_ingredients", palette="viridis")
        plt.title("Rozkład liczby składników bezalkoholowych w klastrach")
        plt.xlabel("Klaster")
        plt.ylabel("Liczba składników bezalkoholowych")
        plt.show()

    def plot_pca_clusters(self, data_reduced, labels):
        """
        Plots the visualization of clusters on PCA-reduced data.

        Parameters:
        - data_reduced (ndarray): Dimensionally reduced data (e.g., from PCA).
        - labels (ndarray): Cluster labels.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=labels, cmap="viridis", alpha=0.7)
        plt.xlabel("Pierwsza składowa PCA")
        plt.ylabel("Druga składowa PCA")
        plt.title("Wizualizacja klastrów po PCA")
        plt.colorbar(label="Klaster")
        plt.show()
