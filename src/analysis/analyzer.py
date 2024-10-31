import pandas as pd


class Analyzer:
    def __init__(self, data):
        """
        Initializes the Analyzer object.

        Parameters:
        - data (DataFrame): Data with assigned cluster labels.
        """
        self.data = data.copy()
        self.cluster_summary = None

    def analyze_clusters(self):
        """
        Exports cluster analysis to an Excel file.

        Parameters:
        - filename (str): Path to the file where the analysis will be saved.

        Raises:
        - ValueError: If cluster analysis has not been performed.
        """
        if "cluster" not in self.data.columns:
            raise ValueError("Dane nie zawierają kolumny 'cluster'. Upewnij się, że klasteryzacja została wykonana.")

        # Wybieramy tylko kolumny numeryczne
        numeric_cols = self.data.select_dtypes(include=["number"]).columns

        # Grupujemy dane po 'cluster' i obliczamy średnie tylko dla kolumn numerycznych
        self.cluster_summary = self.data.groupby("cluster")[numeric_cols].mean()

        return self.cluster_summary

    def export_cluster_analysis(self, filename):
        """
        Eksportuje analizę klastrów do pliku Excel.

        Parametry:
        - filename (str): Ścieżka do pliku, w którym zostanie zapisana analiza.
        """
        if self.cluster_summary is None:
            raise ValueError("Analiza klastrów nie została jeszcze wykonana. Użyj metody analyze_clusters().")

        self.cluster_summary.to_excel(filename, index=True)
        print(f"Analiza klastrów została zapisana do pliku {filename}")

    def export_clustered_cocktails(self, filename):
        """
        Exports the names of cocktails assigned to each cluster in separate sheets.

        Parameters:
        - filename (str): Path to the Excel file where data will be saved.

        Raises:
        - ValueError: If 'name' or 'cluster' columns are missing.
        """
        if "name" not in self.data.columns or "cluster" not in self.data.columns:
            raise ValueError("Dane muszą zawierać kolumny 'name' i 'cluster'.")

        # Filtrujemy tylko kolumny 'name' i 'cluster'
        clustered_cocktails = self.data[["name", "cluster"]]

        # Grupujemy po klastrze i zapisujemy do osobnych arkuszy w pliku Excel
        with pd.ExcelWriter(filename) as writer:
            for cluster in clustered_cocktails["cluster"].unique():
                cluster_df = clustered_cocktails[clustered_cocktails["cluster"] == cluster]
                cluster_df.to_excel(writer, sheet_name=f"Cluster_{cluster}", index=False)

        print(f"Koktajle przypisane do klastrów zostały zapisane do pliku {filename}")

    def export_full_cluster_analysis(self, filename):
        """
        Exports a comprehensive cluster analysis, including mean, median, standard deviation,
        minimum, maximum, and most frequent categorical values, to an Excel file with separate sheets.

        Parameters:
        - filename (str): Path to the Excel file where the full analysis will be saved.

        Raises:
        - ValueError: If 'cluster' column is missing in the data.
        """
        if "cluster" not in self.data.columns:
            raise ValueError("Dane muszą zawierać kolumnę 'cluster'.")

        # Wybieramy kolumny numeryczne i kategoryczne
        numeric_cols = self.data.select_dtypes(include=["number"]).columns
        categorical_cols = self.data.select_dtypes(include=["object", "category"]).columns.drop(
            "cluster", errors="ignore"
        )

        # Słowniki na analizy klastrów dla każdej statystyki
        cluster_analysis_mean = {}
        cluster_analysis_median = {}
        cluster_analysis_std = {}
        cluster_analysis_min = {}
        cluster_analysis_max = {}
        cluster_analysis_categorical = {}

        # Analiza dla każdego klastra
        for cluster_num in self.data["cluster"].unique():
            cluster_data = self.data[self.data["cluster"] == cluster_num]

            # Obliczamy różne statystyki dla danych numerycznych
            cluster_analysis_mean[cluster_num] = cluster_data[numeric_cols].mean()
            cluster_analysis_median[cluster_num] = cluster_data[numeric_cols].median()
            cluster_analysis_std[cluster_num] = cluster_data[numeric_cols].std()
            cluster_analysis_min[cluster_num] = cluster_data[numeric_cols].min()
            cluster_analysis_max[cluster_num] = cluster_data[numeric_cols].max()

            # Najczęstsze wartości kategoryczne
            cluster_analysis_categorical[cluster_num] = cluster_data[categorical_cols].apply(
                lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"
            )

        # Tworzenie DataFrames dla każdej statystyki
        mean_df = pd.DataFrame(cluster_analysis_mean).transpose()
        median_df = pd.DataFrame(cluster_analysis_median).transpose()
        std_df = pd.DataFrame(cluster_analysis_std).transpose()
        min_df = pd.DataFrame(cluster_analysis_min).transpose()
        max_df = pd.DataFrame(cluster_analysis_max).transpose()
        categorical_df = pd.DataFrame(cluster_analysis_categorical).transpose()

        # Zapis do pliku Excel z różnymi arkuszami
        with pd.ExcelWriter(filename) as writer:
            mean_df.to_excel(writer, sheet_name="Mean")
            median_df.to_excel(writer, sheet_name="Median")
            std_df.to_excel(writer, sheet_name="Std Dev")
            min_df.to_excel(writer, sheet_name="Min")
            max_df.to_excel(writer, sheet_name="Max")
            categorical_df.to_excel(writer, sheet_name="Mode Categorical")

        print(f"Pełna analiza klastrów została zapisana do pliku {filename}")

    def export_summary_statistics(self, filename):
        """
        Exports basic descriptive statistics (mean, median, standard deviation,
        minimum, and maximum) for all numeric features to an Excel file.

        Parameters:
        - filename (str): Path to the Excel file where the summary statistics will be saved.
        """
        # Wybieramy wszystkie kolumny numeryczne
        features = self.data.select_dtypes(include=["number"]).columns

        # Obliczamy statystyki dla wybranych kolumn
        summary_statistics = {
            "Cechy": features,
            "Średnia": [self.data[feature].mean() for feature in features],
            "Mediana": [self.data[feature].median() for feature in features],
            "Odchylenie standardowe": [self.data[feature].std() for feature in features],
            "Wartość minimalna": [self.data[feature].min() for feature in features],
            "Wartość maksymalna": [self.data[feature].max() for feature in features],
        }

        # Tworzenie DataFrame z wyników
        summary_df = pd.DataFrame(summary_statistics)

        # Zapis do pliku Excel
        with pd.ExcelWriter(filename) as writer:
            summary_df.to_excel(writer, sheet_name="Summary Statistics", index=False)

        print(f"Podstawowe statystyki opisowe zostały zapisane do pliku {filename}")
