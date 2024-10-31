import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class DimensionalityReducer:
    def __init__(self, data, n_components=None):
        """
        Initializes a DimensionalityReducer object.

        Parameters:
        - data (DataFrame or ndarray): Data for dimensionality reduction.
        - n_components (int or None): Number of PCA components. If None, all components are used.
        """
        self.data = data
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)
        self.data_reduced = None

    def reduce_dimensions(self):
        """
        Reduces data dimensionality using PCA and updates self.data_reduced.

        Returns:
        - data_reduced (ndarray): Data transformed into the reduced PCA component space.
        """
        self.data_reduced = self.pca.fit_transform(self.data)
        return self.data_reduced

    def plot_variance(self):
        """
        Plots the cumulative explained variance by the PCA components.

        Note:
        Must be called after reduce_dimensions() has been executed.

        Raises:
        - ValueError: If PCA has not been fitted before calling this method.
        """
        if self.pca.explained_variance_ratio_ is None:
            raise ValueError("PCA musi zostać dopasowane przed wywołaniem tej metody.")

        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_), marker="o", linestyle="--")
        plt.xlabel("Liczba komponentów PCA")
        plt.ylabel("Kumulatywna wariancja wyjaśniona")
        plt.title("Wykres kumulatywnej wariancji wyjaśnionej")
        plt.grid(True)
        plt.show()
