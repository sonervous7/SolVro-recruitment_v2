from pipeline.pipeline import CocktailClusteringPipeline


def main():
    """
    Main function to initiate the cocktail clustering process.

    This function loads the dataset path and creates an instance of
    the `CocktailClusteringPipeline` class from `pipeline.py`. It then
    calls the `run_pipeline()` method to execute the full process of
    analysis and clustering of cocktails.
    """
    data_path = r"..\data\cocktail_dataset.json"
    pipeline = CocktailClusteringPipeline(data_path)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
