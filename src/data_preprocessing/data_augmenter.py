from config.mapping_dicts import ingredient_type_mapping, alcohol_percentage_dict


class DataAugmenter:
    def __init__(self, data):
        """
        Initializes the DataAugmenter object.

        Parameters:
        - data (DataFrame): The data to be augmented.
        """
        self.data = data

    def fill_missing_ingredient_types(self):
        """
        Fills in missing ingredient types in the 'ingredients' column.
        Updates 'ingredients' in self.data.
        """

        def fill_types(ingredients):
            for ingredient in ingredients:
                if ingredient.get("type") is None:
                    ingredient_name = ingredient.get("name")
                    if ingredient_name in ingredient_type_mapping:
                        ingredient["type"] = ingredient_type_mapping[ingredient_name]
                    else:
                        ingredient["type"] = "Unknown"
            return ingredients

        self.data["ingredients"] = self.data["ingredients"].apply(fill_types)

    def fill_missing_percentages(self):
        """
        Fills in missing alcohol percentage values in the ingredients.
        Updates 'ingredients' in self.data.
        """

        def fill_percentages(ingredients):
            for ingredient in ingredients:
                if ingredient.get("percentage") is None:
                    if ingredient.get("alcohol") == 1:
                        ingredient_name = ingredient.get("name")
                        if ingredient_name in alcohol_percentage_dict:
                            ingredient["percentage"] = alcohol_percentage_dict[ingredient_name]
                        else:
                            ingredient["percentage"] = 40  # Domyślna wartość dla alkoholu
                    else:
                        ingredient["percentage"] = 0
            return ingredients

        self.data["ingredients"] = self.data["ingredients"].apply(fill_percentages)

    def update_soda_water_alcohol(self):
        """
        Updates the alcohol information for 'soda water'.
        Sets 'alcohol' to 0 for ingredients with the name 'soda water'.
        """

        def update_alcohol(ingredients):
            for ingredient in ingredients:
                if ingredient.get("name", "").lower() == "soda water":
                    ingredient["alcohol"] = 0
            return ingredients

        self.data["ingredients"] = self.data["ingredients"].apply(update_alcohol)

    def augment_data(self):
        """
        Executes the full data augmentation process by calling all methods in the appropriate order.
        """
        self.update_soda_water_alcohol()
        self.fill_missing_ingredient_types()
        self.fill_missing_percentages()
