import pandas as pd


class DataLoader:
    """
    Initializes the DataLoader object.

    Parameters:
    - file_path (str): Path to the JSON file containing data.
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """
        Loads data from a JSON file.

        Returns:
        - data (DataFrame): Loaded data as a pandas DataFrame.
        """
        self.data = pd.read_json(self.file_path)
        return self.data
