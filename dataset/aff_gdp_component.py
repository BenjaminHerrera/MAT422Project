# Model related imports

# Dataset related imports
from dataset import BaseComponent

# Data augmentation/processing imports

# Evaluation related imports

# PyTorch related imports

# Util related imports

# Execution related imports
import re

# Datastructure related imports
import pandas as pd

# Miscellaneous imports


class AFFGDPComponent(BaseComponent):
    """Agricultural, Forestry, and Fishery GDP statistics for the PalayDataset.
    This class contains the following attributes:

    - Province* (Literal[0, ..., 81])
    - Year* (Literal[1987, ..., 2024])
    - Quarter* (Literal[1, 2, 3, 4])
    - Crop* (Literal["Irrigated Palay", "Rainfed Palay"])
    - AFF GDP (float, "Billions in Philippine Pesos")

    Attribute names with the "*" next to them are considered to be primary keys

    This code is meant for datasource from the following source location:
        https://openstat.psa.gov.ph/PXWeb/pxweb/en/DB/DB__2B__NA__QT__23QNAP__PRD/0982B5CQPQ1.px/?rxid=73b73b2f-5144-4ebd-962b-72e9c721c7bf
    """

    def post_init(self: BaseComponent):
        """Post initialization method to sort out the content for the GDP
        statistics. Applies a history spread and placement into the unknown
        period when specified.

        Args:
            self (ElNinoSSTComponent): Instance method of the class
        """

        # Translate the CSV to a pandas dataframe and crop it
        df = pd.read_csv(self.source_path, skiprows=1)
        df = df.iloc[:, 1:]

        # Copy the key as a provisional and slightly modify it
        self.dataframe = self.key.copy()

        # Initialize new columns for AFF GDP with history spread
        for spread in range(self.history_spread + 1):
            col_name = f"AFF GDP Q_n-{spread}" if spread > 0 else "AFF GDP Q_n"
            self.dataframe[col_name] = float("nan")

        # Create a dictionary to map (year, quarter) to GDP values
        gdp_values = {}
        for col_name, col_series in df.items():
            year, quarter = AFFGDPComponent._extract_year_quarter(col_name)
            if year and quarter:
                gdp_values[(year, quarter)] = col_series.values[0]

        # Apply GDP values to the DataFrame
        for (year, quarter), gdp_value in gdp_values.items():
            mask = (self.dataframe["Year"] == year) & (
                self.dataframe["Quarter"] == quarter
            )
            self.dataframe.loc[mask, "AFF GDP Q_n"] = gdp_value

        # Apply history spread by shifting content down
        for spread in range(1, self.history_spread + 1):
            col_name_prev = (
                f"AFF GDP Q_n-{spread-1}" if spread > 1 else "AFF GDP Q_n"
            )
            col_name_current = f"AFF GDP Q_n-{spread}"
            self.dataframe[col_name_current] = self.dataframe.groupby(
                ["Province", "Crop"]
            )[col_name_prev].shift(1)

        # If the component was constructed with the unknown period omitted,
        #   remove the unknown period (the current reference quarter)
        if not self.include_unknown_period:
            self.dataframe = self.dataframe.drop("AFF GDP Q_n", axis=1)

    @staticmethod
    def _extract_year_quarter(col_name: str) -> tuple[int, int]:
        """Extract year and quarter from column name

        Args:
            col_name (str): The column to extract information from when
                iterating through the columns of the datasource

        Returns:
            tuple[int, int]: Returns either nothing if the col_name does not
                follow the regex. Or it returns the year as the first
                element and the quarter as the second element
        """
        # Create a Regex matcher and check if the regex matches
        match = re.search(r"(\d{4}) Q(\d)", col_name)

        # If it does, parse and return the year and the quarter from the string
        if match:
            return int(match.group(1)), int(match.group(2))

        # If not, return nothing
        return None, None
