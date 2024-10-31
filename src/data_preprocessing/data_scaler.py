import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataScaler:
    def __init__(self, data):
        """
        Inicjalizuje obiekt DataScaler z danymi do skalowania.

        Parametry:
        - data (DataFrame): Dane do skalowania.
        """
        self.data = data.copy()
        self.scaled_data = None

    def scale_standard(self, numeric_features=None):
        """
        Skaluje dane numeryczne przy użyciu StandardScaler.

        Parametry:
        - numeric_features (list): Lista nazw cech numerycznych do skalowania.
          Jeśli None, użyje domyślnych cech.

        Zwraca:
        - scaled_data (DataFrame): DataFrame z przeskalowanymi cechami.
        """
        if numeric_features is None:
            numeric_features = [
                "estimated_abv",
                "num_total_ingredients",
                "num_alcoholic_ingredients",
                "num_non_alcoholic_ingredients",
                "alcoholic_ratio"
            ]

        # Sprawdzenie, czy wszystkie cechy istnieją w danych
        missing_features = [feat for feat in numeric_features if feat not in self.data.columns]
        if missing_features:
            raise ValueError(f"Następujące cechy nie istnieją w danych: {missing_features}")

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(self.data[numeric_features])

        self.scaled_data = pd.DataFrame(data_scaled, columns=numeric_features, index=self.data.index)

        return self.scaled_data

    def scale_combined(self, minmax_features=None, standard_features=None):
        """
        Skaluje dane używając zarówno MinMaxScaler, jak i StandardScaler.

        Parametry:
        - minmax_features (list): Lista cech do skalowania MinMaxScaler.
          Jeśli None, użyje domyślnych cech.
        - standard_features (list): Lista cech do skalowania StandardScaler.
          Jeśli None, użyje domyślnych cech.

        Zwraca:
        - scaled_data (DataFrame): DataFrame z przeskalowanymi cechami.
        """
        if minmax_features is None:
            minmax_features = [
                "estimated_abv",
                "num_total_ingredients",
                "num_alcoholic_ingredients",
                "num_non_alcoholic_ingredients",
            ]

        if standard_features is None:
            standard_features = ["alcoholic_ratio"]

        # Sprawdzenie, czy wszystkie cechy istnieją w danych
        missing_minmax_features = [feat for feat in minmax_features if feat not in self.data.columns]
        missing_standard_features = [feat for feat in standard_features if feat not in self.data.columns]

        if missing_minmax_features:
            raise ValueError(f"Następujące cechy do MinMaxScaler nie istnieją w danych: {missing_minmax_features}")
        if missing_standard_features:
            raise ValueError(f"Następujące cechy do StandardScaler nie istnieją w danych: {missing_standard_features}")

        minmax_scaler = MinMaxScaler()
        standard_scaler = StandardScaler()

        # Skalowanie cech z MinMaxScaler (0-1)
        data_minmax_scaled = minmax_scaler.fit_transform(self.data[minmax_features])
        data_minmax_scaled_df = pd.DataFrame(
            data_minmax_scaled, columns=minmax_features, index=self.data.index
        )

        # Skalowanie cech z StandardScaler (średnia 0, odchylenie 1)
        data_standard_scaled = standard_scaler.fit_transform(self.data[standard_features])
        data_standard_scaled_df = pd.DataFrame(
            data_standard_scaled, columns=standard_features, index=self.data.index
        )

        # Połączenie obu przeskalowanych zestawów cech
        self.scaled_data = pd.concat([data_minmax_scaled_df, data_standard_scaled_df], axis=1)

        return self.scaled_data

    def get_scaled_data(self):
        """
        Zwraca przeskalowane dane.

        Zwraca:
        - scaled_data (DataFrame): Przeskalowane dane.
        """
        return self.scaled_data
