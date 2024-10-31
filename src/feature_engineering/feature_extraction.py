import re

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


class FeatureEngineer:
    """
    Initializes the FeatureEngineer object.

    Parameters:
    - data (DataFrame): The dataset to be engineered.
    """

    def __init__(self, data):
        self.data = data.copy()
        self.mlb = MultiLabelBinarizer()
        self.extract_ingredient_names()
        self.extract_measures()
        self.extract_percentages()
        self.extract_alcohol_content()
        self.extract_types()
        self.calculate_abv()
        self.categorize_instruction_lengths()
        self.categorize_measures()
        self.calculate_ingredient_counts()
        self.calculate_alcoholic_ratio()
        self.encode_ingredient_types()
        self.encode_measure_categories()

    def extract_ingredient_names(self):
        """Extracts ingredient names and adds them to the data."""
        self.data["ingredient_names"] = self.data["ingredients"].apply(
            lambda ingredients: [ingredient.get("name") for ingredient in ingredients]
        )

    def extract_types(self):
        """Extracts ingredient types."""
        self.data["ingredient_type"] = self.data["ingredients"].apply(
            lambda ingredients: [ingredient.get("type", "").strip() for ingredient in ingredients]
        )

    def extract_alcohol_content(self):
        """Extracts alcohol content information."""
        self.data["ingredient_alcohol"] = self.data["ingredients"].apply(
            lambda ingredients: [ingredient.get("alcohol", 0) for ingredient in ingredients]
        )

    def extract_measures(self):
        """Extracts ingredient measures and adds them to the data."""
        self.data["ingredient_measure"] = self.data["ingredients"].apply(self._extract_measure_list)

    def _extract_measure_list(self, ingredients):
        """
        Helper function to extract measure list from ingredients.

        Parameters:
        - ingredients (list): List of ingredient dictionaries.

        Returns:
        - measures (list): List of measures for each ingredient.
        """
        measures = []
        for ingredient in ingredients:
            measures.append(ingredient.get("measure", ""))
        return measures

    def categorize_measures(self):
        """Categorizes ingredient measures and adds them to the data."""
        self.data["measure_category"] = self.data["ingredient_measure"].apply(self._categorize_measures_list)

    def _categorize_measures_list(self, measures_list):
        """
        Helper function to categorize measures in a list.

        Parameters:
        - measures_list (list): List of measures.

        Returns:
        - list: List of measure categories.
        """
        return [self._categorize_measure(measure) for measure in measures_list]

    def _categorize_measure(self, measure):
        """
        Categorizes individual measure based on keywords.

        Parameters:
        - measure (str): The measure to categorize.

        Returns:
        - str: Category of the measure ('small', 'medium', 'large', or 'unknown').
        """
        measure = measure.lower().strip()
        if any(
            keyword in measure
            for keyword in [
                "dash",
                "splash",
                "twist",
                "slice",
                "piece",
                "wedge",
                "cube",
                "chunk",
                "grated",
                "drop",
            ]
        ):
            return "small"
        elif any(keyword in measure for keyword in ["1/4", "1/2", "3/4", "tsp", "tblsp"]):
            return "small"
        elif any(keyword in measure for keyword in ["1 oz", "1 1/2 oz", "1 oz", "1 tsp", "1 tblsp"]):
            return "medium"
        elif any(keyword in measure for keyword in ["2 oz", "3 oz", "4 oz", "cup", "juice of 1"]):
            return "large"
        else:
            return "unknown"

    def extract_percentages(self):
        """Extracts alcohol percentage content for each ingredient."""
        self.data["ingredient_percentage"] = self.data["ingredients"].apply(
            lambda ingredients: [
                ingredient.get("percentage", 0) if ingredient.get("alcohol") == 1 else 0 for ingredient in ingredients
            ]
        )

    def calculate_abv(self):
        """Calculates estimated alcohol by volume (ABV) for each cocktail."""
        self.data["estimated_abv"] = self.data.apply(self._calculate_row_abv, axis=1)

    def _calculate_row_abv(self, row):
        """
        Helper function to calculate ABV for a single row.

        Parameters:
        - row (Series): Row of data containing 'ingredient_measure' and 'ingredient_percentage'.

        Returns:
        - float: Estimated ABV for the cocktail.
        """
        ingredient_measures = row["ingredient_measure"]
        ingredient_percentages = row["ingredient_percentage"]

        total_volume = 0
        total_alcohol_volume = 0

        for measure, percentage in zip(ingredient_measures, ingredient_percentages):
            oz_volume = self._measure_to_oz(measure)
            total_volume += oz_volume
            total_alcohol_volume += oz_volume * (percentage / 100)

        return (total_alcohol_volume / total_volume) * 100 if total_volume > 0 else 0

    def _measure_to_oz(self, measure):
        """
        Converts a measure to ounces.

        Parameters:
        - measure (str): Measure string.

        Returns:
        - float: Volume in ounces.
        """
        measure = measure.lower().strip()
        match = re.match(r"(\d+\/\d+|\d+\.\d+|\d+)", measure)
        if match:
            quantity_str = match.group(1)
            # Konwersja ułamka na float
            if "/" in quantity_str:
                numerator, denominator = quantity_str.split("/")
                quantity = float(numerator) / float(denominator)
            else:
                quantity = float(quantity_str)
        else:
            quantity = 0.0  # Domyślna wartość, jeśli nie można wyodrębnić liczby

        # Ustawienie jednostek miar
        if "oz" in measure:
            multiplier = 1.0
        elif "tsp" in measure:
            multiplier = 0.166667  # 1 tsp to około 1/6 uncji
        elif "tblsp" in measure:
            multiplier = 0.5  # 1 tblsp to około 1/2 uncji
        elif "cup" in measure:
            multiplier = 8.0  # 1 cup to około 8 uncji
        elif "dash" in measure or "drop" in measure:
            multiplier = 0.02  # Przybliżenie dla "dash" i "drop"
        elif "cl" in measure:
            multiplier = 0.33814  # 1 cl to około 0.33814 uncji
        elif "ml" in measure:
            multiplier = 0.033814  # 1 ml to około 0.033814 uncji
        else:
            multiplier = 0.0  # Nieznana jednostka

        return quantity * multiplier

    def categorize_instruction_lengths(self):
        """Categorizes the length of instructions."""
        self.data["instruction_length"] = self.data["instructions"].apply(self._categorize_instruction_length)

    def _categorize_instruction_length(self, instruction):
        """
        Categorizes the length of a single instruction.

        Parameters:
        - instruction (str): Instruction string.

        Returns:
        - str: Length category ('short', 'medium', or 'long').
        """
        length = len(instruction)
        if length < 50:
            return "short"
        elif 50 <= length < 150:
            return "medium"
        else:
            return "long"

    def calculate_ingredient_counts(self):
        """Calculates counts of alcoholic, non-alcoholic, and total ingredients."""
        # Liczba składników alkoholowych w każdym koktajlu
        self.data["num_alcoholic_ingredients"] = self.data["ingredient_alcohol"].apply(sum)

        # Całkowita liczba składników w koktajlu
        self.data["num_total_ingredients"] = self.data["ingredient_names"].apply(len)

        # Liczba składników bezalkoholowych w koktajlu
        self.data["num_non_alcoholic_ingredients"] = (
            self.data["num_total_ingredients"] - self.data["num_alcoholic_ingredients"]
        )

    def calculate_alcoholic_ratio(self):
        """Calculates the ratio of alcoholic ingredients in each cocktail."""
        self.data["alcoholic_ratio"] = self.data["num_alcoholic_ingredients"] / self.data["num_total_ingredients"]

    def encode_ingredient_types(self):
        """Encodes 'ingredient_type' using MultiLabelBinarizer and adds result to the data."""
        ingredient_type_matrix = pd.DataFrame(
            self.mlb.fit_transform(self.data["ingredient_type"]), columns=self.mlb.classes_, index=self.data.index
        )
        self.data = pd.concat([self.data, ingredient_type_matrix], axis=1)

    def encode_measure_categories(self):
        """Encodes 'measure_category' using MultiLabelBinarizer and adds result to the data."""
        # Nowy obiekt MLB dla 'measure_category', aby uniknąć konfliktu z poprzednim
        mlb_measure = MultiLabelBinarizer()
        measure_category_matrix = pd.DataFrame(
            mlb_measure.fit_transform(self.data["measure_category"]),
            columns=mlb_measure.classes_,
            index=self.data.index,
        )
        self.data = pd.concat([self.data, measure_category_matrix], axis=1)

    def encode_ingredient_names(self):
        """Encodes 'ingredient_names' using MultiLabelBinarizer and adds result to the data."""
        mlb_ingredients = MultiLabelBinarizer()
        ingredient_matrix = pd.DataFrame(
            mlb_ingredients.fit_transform(self.data["ingredient_names"]),
            columns=mlb_ingredients.classes_,
            index=self.data.index,
        )
        self.data = pd.concat([self.data, ingredient_matrix], axis=1)

    def get_engineered_data(self):
        """Returns the DataFrame with extracted and categorized features."""
        return self.data
