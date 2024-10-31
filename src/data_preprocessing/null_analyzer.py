import pandas as pd


class NullAnalyzer:
    def __init__(self, data):
        """
        Initializes a NullAnalyzer object.

        Parameters:
        - data (DataFrame): Data containing cocktails and ingredients.
        """
        self.data = data

    def count_null_overall(self):
        """
        Counts the total number of null values in the entire dataset.

        Returns:
        - int: Total count of null values.
        """
        total_nulls = self.data.isnull().sum().sum()
        print(f"Ogólna liczba wartości null: {total_nulls}")
        return total_nulls

    def count_null_in_columns(self):
        """
        Counts null values in selected columns ('instructions', 'category', 'name').

        Returns:
        - dict: Dictionary with the count of null values in each selected column.
        """
        columns_to_check = ["instructions", "category", "name"]
        null_counts = self.data[columns_to_check].isnull().sum()
        print("\nLiczba wartości null w wybranych kolumnach:")
        print(null_counts)
        return null_counts

    def count_null_in_ingredients_field(self, field):
        """
        Counts null values in a specified ingredient field for each cocktail.

        Parameters:
        - field (str): Ingredient field to check for null values (e.g., 'alcohol', 'type', 'percentage').

        Returns:
        - DataFrame: DataFrame containing cocktail ID and ingredients with null values in the specified field.
        """
        null_ingredient_data = []
        for _, row in self.data.iterrows():
            cocktail_id = row["id"]  # Zakładamy, że kolumna 'id' jest identyfikatorem koktajlu
            for ingredient in row["ingredients"]:
                if ingredient.get(field) is None:
                    null_ingredient_data.append(
                        {
                            "cocktail_id": cocktail_id,
                            "ingredient_id": ingredient.get("id"),
                            "ingredient_name": ingredient.get("name"),
                            f"{field}_is_null": True,
                        }
                    )

        null_ingredient_df = pd.DataFrame(null_ingredient_data)
        print(f"\nWartości null w polu składnika '{field}':")
        print(null_ingredient_df)
        return null_ingredient_df

    def count_nulls_in_alcohol(self):
        """
        Counts null values in the 'alcohol' field of ingredients.

        Returns:
        - DataFrame: DataFrame containing cocktail ID and ingredients with null values in the 'alcohol' field.
        """
        return self.count_null_in_ingredients_field("alcohol")

    def count_nulls_in_type(self):
        """
        Counts null values in the 'type' field of ingredients.

        Returns:
        - DataFrame: DataFrame containing cocktail ID and ingredients with null values in the 'type' field.
        """
        return self.count_null_in_ingredients_field("type")

    def count_nulls_in_percentage(self):
        """
        Counts null values in the 'percentage' field of ingredients.

        Returns:
        - DataFrame: DataFrame containing cocktail ID and ingredients with null values in the 'percentage' field.
        """
        return self.count_null_in_ingredients_field("percentage")
