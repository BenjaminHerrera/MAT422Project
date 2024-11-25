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


class PalayYieldComponent(BaseComponent):
    """Palay yield information for the PalayDataset. This class
    contains the following components:

    - Province* (Literal[0, ..., 81])
    - Year* (Literal[1987, ..., 2024])
    - Quarter* (Literal[1, 2, 3, 4])
    - Crop* (Literal["Irrigated Palay", "Rainfed Palay"])
    - PalayYield (float, "metric tons")

    Attribute names with the "*" next to them are considered to be primary keys

    This code is meant for datasource from the following source location:
        https://openstat.psa.gov.ph/PXWeb/pxweb/en/DB/DB__2E__CS/0022E4EAHC0.px/?rxid=bdf9d8da-96f1-41-00-ae09-18cb3eaeb313
    """

    def post_init(self: "PalayYieldComponent"):
        """Post initialization method to sort out the content for the yield
        statistics. Applies a history spread and placement into the unknown
        period when specified.

        Args:
            self (PalayYieldComponent): Instance method of the class
        """

        # Read the dataset and format it
        df = pd.read_csv(self.source_path, skiprows=1)
        df["Geolocation"] = df["Geolocation"].str.lstrip(".")
        for col in df.columns[2:]:
            df[col] = df[col].apply(lambda x: 0.0 if x == ".." else x)

        # Copy the key as a provisional and slightly modify it
        self.dataframe = self.key.copy()

        # Initialize new columns for palay yield with history spread
        for spread in range(self.history_spread + 1):
            col_name = (
                f"PALAY YIELD Q_n-{spread}"
                if spread > 0
                else "PALAY YIELD Q_n"
            )
            self.dataframe[col_name] = float("nan")

        # Apply palay yield values to the DataFrame
        for _, row in df.iterrows():
            province = row["Geolocation"]
            crop = row["Ecosystem/Croptype"]
            for col_name, value in row.items():
                if col_name not in ["Geolocation", "Ecosystem/Croptype"]:
                    year, quarter = PalayYieldComponent._extract_year_quarter(
                        col_name
                    )
                    if year and quarter:
                        mask = (
                            (self.dataframe["Year"] == year)
                            & (self.dataframe["Quarter"] == quarter)
                            & (self.dataframe["Province"] == province)
                            & (self.dataframe["Crop"] == crop)
                        )
                        self.dataframe.loc[mask, "PALAY YIELD Q_n"] = value

        # Apply history spread by shifting content down
        for spread in range(1, self.history_spread + 1):
            col_name_prev = (
                f"PALAY YIELD Q_n-{spread-1}"
                if spread > 1
                else "PALAY YIELD Q_n"
            )
            col_name_current = f"PALAY YIELD Q_n-{spread}"
            self.dataframe[col_name_current] = self.dataframe.groupby(
                ["Province", "Crop"]
            )[col_name_prev].shift(1)

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
        match = re.search(r"(\d{4}) Quarter (\d)", col_name)

        # If it does, parse and return the year and the quarter from the string
        if match:
            return int(match.group(1)), int(match.group(2))

        # If not, return nothing
        return None, None
