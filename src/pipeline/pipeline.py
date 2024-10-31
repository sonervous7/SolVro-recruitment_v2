from analysis.analyzer import Analyzer
from clustering.cluster import Clusterer
from data_preprocessing.data_augmenter import DataAugmenter
from data_preprocessing.data_cleaner import DataCleaner
from data_preprocessing.data_loading import DataLoader
from data_preprocessing.data_scaler import DataScaler
from dimensionality_reduction.dimensionality_reducer import DimensionalityReducer
from feature_engineering.feature_extraction import FeatureEngineer
from clustering.model_evaluator import ModelEvaluator
from visualization.visualizer import Visualizer


class CocktailClusteringPipeline:
    def __init__(self, data_path):
        """
        Initializes the CocktailClusteringPipeline with a data path for the dataset.

        Parameters:
        - data_path (str): Path to the dataset file.
        """
        self.data_loader = DataLoader(data_path)
        self.data = None
        self.scaled_data = None
        self.data_reduced = None
        self.labels = None

    def run_pipeline(self):
        """
        Executes the full data processing and clustering pipeline, which includes
        loading data, null analysis, data augmentation, data cleaning, feature engineering,
        scaling, dimensionality reduction, evaluation, clustering, and visualization.
        """
        # **1. Loading Data**
        self.data = self.data_loader.load_data()

        # **2. NULL_ANALYZER**

        # Could check simply uncommenting

        # null_and_nan_analyzer = NullAnalyzer(self.data)
        #
        # null_and_nan_analyzer.count_null_overall()
        # null_and_nan_analyzer.count_null_in_columns()
        # null_and_nan_analyzer.count_nulls_in_alcohol()
        # null_and_nan_analyzer.count_nulls_in_percentage()
        # null_and_nan_analyzer.count_nulls_in_type()

        # **3. Data Augmentation**
        augmenter = DataAugmenter(self.data)
        augmenter.fill_missing_ingredient_types()
        augmenter.fill_missing_percentages()
        augmenter.update_soda_water_alcohol()

        # **4. Data Cleaning**
        cleaner = DataCleaner(self.data)
        cleaner.clean_data()

        # **5. NULL ANALYZER AFTER AUGUMENTATION AND CLEANING**

        # Could check simply uncommenting

        # null_and_nan_analyzer = NullAnalyzer(self.data)
        #
        # print("AFTER PREPROCESSING")
        # null_and_nan_analyzer.count_null_overall()
        # null_and_nan_analyzer.count_null_in_columns()
        # null_and_nan_analyzer.count_nulls_in_alcohol()
        # null_and_nan_analyzer.count_nulls_in_percentage()
        # null_and_nan_analyzer.count_nulls_in_type()

        # **6. Feature Engineering**
        engineer = FeatureEngineer(self.data)
        self.data = engineer.get_engineered_data()

        # Check print

        # print(self.data.columns)

        # **7. Data Scaling**
        scaler = DataScaler(data=self.data)
        self.scaled_data = scaler.scale_standard()
        # Different, scaler, no time to implement console gui
        # self.scaled_data = scaler.scale_combined()

        # **8. Dimensionality Reduction**
        reducer = DimensionalityReducer(self.scaled_data, n_components=2)
        self.data_reduced = reducer.reduce_dimensions()
        reducer.plot_variance()

        # **9. Evaluation**
        evaluator = ModelEvaluator(self.data_reduced)

        evaluator.plot_elbow_method()

        evaluator.plot_silhouette_scores()

        evaluator.plot_davies_bouldin_scores()

        evaluator.plot_variance_explained(data_scaled=self.scaled_data)

        # **10. Clustering**

        clusterer = Clusterer(self.data_reduced, n_clusters=8)
        self.labels = clusterer.cluster_data()
        silhouette = clusterer.calculate_silhouette_score()
        print(f"Współczynnik silhouette: {silhouette}")

        # **11. Next Evaluation**
        evaluator.print_silhouette_score(self.labels)
        clusterer.print_davies_bouldin_score(self.labels, n_clusters=8)
        evaluator.plot_kmeans_clusters(data_reduced=self.data_reduced, labels=self.labels)

        # **12. Adding Cluster Labels to Data**
        self.data["cluster"] = self.labels

        # **13. Analysis**
        analyzer = Analyzer(self.data)
        cluster_summary = analyzer.analyze_clusters()
        analyzer.export_cluster_analysis("../data/cluster_analysis.xlsx")
        analyzer.export_clustered_cocktails("../data/clustered_cocktails.xlsx")
        analyzer.export_full_cluster_analysis("../data/full_cluster_analysis.xlsx")
        analyzer.export_summary_statistics("../data/summary_statistics.xlsx")

        # **14. Visualization**
        visualizer = Visualizer(self.data)
        # Visualizations based on the clusters
        visualizer.plot_cluster_distribution_by_glass()
        visualizer.plot_cluster_distribution_by_category()
        visualizer.plot_estimated_abv_in_clusters()
        visualizer.plot_alcoholic_ratio_in_clusters()
        visualizer.plot_num_ingredients_in_clusters()
        visualizer.plot_num_alcoholic_ingredients_in_clusters()
        visualizer.plot_num_non_alcoholic_ingredients_in_clusters()
