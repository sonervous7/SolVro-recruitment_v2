import re


class DataCleaner:
    def __init__(self, data):
        """
        Initializes the DataCleaner object.

        Parameters:
        - data (DataFrame): The data to be processed.
        """
        self.data = data

    def remove_tags_column(self):
        """
        Removes the 'tags' column if it exists, due to a high amount of missing data.
        """
        if "tags" in self.data.columns:
            self.data.drop(columns=["tags"], inplace=True)
            print("Kolumna 'tags' została usunięta.")
        else:
            print("Kolumna 'tags' nie istnieje w danych.")

    def remove_duplicate_ingredients(self):
        """
        Removes duplicate ingredients in the 'ingredients' column for each cocktail.
        Updates the 'ingredients' column in self.data.
        """

        def remove_duplicates(ingredients):
            seen = set()  # Przechowuje unikalne pary (id, name)
            unique_ingredients = []
            for ingredient in ingredients:
                # Utwórz parę (id, name) dla sprawdzenia duplikatu
                identifier = (ingredient.get("id"), ingredient.get("name").lower().strip())
                if identifier not in seen:
                    seen.add(identifier)
                    unique_ingredients.append(ingredient)
            return unique_ingredients

        self.data["ingredients"] = self.data["ingredients"].apply(remove_duplicates)

    def clean_ingredient_names(self):
        """
        Standardizes ingredient names in the 'ingredients' column.
        Updates ingredient names in-place.
        """

        def clean_name(ingredient):
            name = ingredient.get("name", "").lower().strip()
            name = re.sub(r"\s+", " ", name)  # Usunięcie nadmiarowych spacji
            ingredient["name"] = name
            return ingredient

        self.data["ingredients"] = self.data["ingredients"].apply(
            lambda ingredients: [clean_name(ingredient) for ingredient in ingredients]
        )

    def replace_none_types(self):
        """
        Replaces None values in ingredient types with 'Unknown' in the 'ingredients' column.
        Updates 'ingredients' in self.data.
        """

        def replace_none(ingredient):
            """
            Replaces None values in the ingredient type with 'Unknown'.

            Parameters:
            - ingredient (dict): A dictionary containing ingredient details.

            Returns:
            - ingredient (dict): Updated ingredient dictionary with 'Unknown' in place of None in the 'type' field.
            """
            if ingredient.get("type") is None:
                ingredient["type"] = "Unknown"
            return ingredient

        self.data["ingredients"] = self.data["ingredients"].apply(
            lambda ingredients: [replace_none(ingredient) for ingredient in ingredients]
        )

    def replace_synonyms(self):
        """
        Replaces synonyms in ingredient types for consistency.
        Updates 'ingredients' in self.data.
        """
        synonyms_dict = {
            "Whisky": "Whiskey",
            "Bitter": "Bitters",
            "Liquer": "Liqueur",
        }

        def replace_synonym(ingredient):
            """
            Replaces synonyms in the ingredient type for consistency.

            Parameters:
            - ingredient (dict): A dictionary containing ingredient details.

            Returns:
            - ingredient (dict): Updated ingredient dictionary with standardized 'type' field.
            """
            typ = ingredient.get("type", "")
            ingredient["type"] = synonyms_dict.get(typ, typ)
            return ingredient

        self.data["ingredients"] = self.data["ingredients"].apply(
            lambda ingredients: [replace_synonym(ingredient) for ingredient in ingredients]
        )

    def clean_data(self):
        """
        Performs full data cleaning by executing all necessary methods in the correct order.
        """
        self.remove_tags_column()
        self.remove_duplicate_ingredients()
        self.clean_ingredient_names()
        self.replace_none_types()
        self.replace_synonyms()
