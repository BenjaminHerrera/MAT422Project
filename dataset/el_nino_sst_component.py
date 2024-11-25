# Model related imports

# Dataset related imports
from dataset import BaseComponent

# Data augmentation/processing imports

# Evaluation related imports

# PyTorch related imports

# Util related imports

# Execution related imports

# Datastructure related imports
import pandas as pd

# Miscellaneous imports


class ElNinoSSTComponent(BaseComponent):
    """El Nino Sea Surface Temperature for the PalayDataset. This class
    contains the following components:

    - Province* (Literal[0, ..., 81])
    - Year* (Literal[1987, ..., 2024])
    - Quarter* (Literal[1, 2, 3, 4])
    - Crop* (Literal["Irrigated Palay", "Rainfed Palay"])
    - El Nino 3.4 SST (float, "Mean °C")

    Attribute names with the "*" next to them are considered to be primary keys

    This code is meant for datasource from the following source location:
        https://www.ncei.noaa.gov/access/monitoring/enso/sst
    """

    def post_init(self: "ElNinoSSTComponent"):
        """Post initialization method to sort out the content for the SST
        statistics. Applies a history spread and placement into the unknown
        period when specified.

        Args:
            self (ElNinoSSTComponent): Instance method of the class
        """

        # Read the dataset
        df = pd.read_csv(self.source_path, delim_whitespace=True)

        # Convert monthly data to quarterly data by averaging
        df["Quarter"] = df["MON"].apply(lambda x: (x - 1) // 3 + 1)
        quarterly_df = df.groupby(["YR", "Quarter"]).mean().reset_index()

        # Initialize new columns for NINO SST with history spread
        self.dataframe = self.key.copy()
        for spread in range(self.history_spread + 1):
            col_name = (
                f"NINO3.4 SST Q_n-{spread}"
                if spread > 0
                else "NINO3.4 SST Q_n"
            )
            self.dataframe[col_name] = float("nan")

        # Create a dictionary to map (year, quarter) to NINO3.4 SST values
        sst_values = {}
        for _, row in quarterly_df.iterrows():
            year = row["YR"]
            quarter = row["Quarter"]
            sst_values[(year, quarter)] = row["NINO3.4"]

        # Apply SST values to the DataFrame
        for (year, quarter), sst_value in sst_values.items():
            mask = (self.dataframe["Year"] == year) & (
                self.dataframe["Quarter"] == quarter
            )
            self.dataframe.loc[mask, "NINO3.4 SST Q_n"] = sst_value

        # Apply history spread by shifting content down
        for spread in range(1, self.history_spread + 1):
            col_name_prev = (
                f"NINO3.4 SST Q_n-{spread-1}"
                if spread > 1
                else "NINO3.4 SST Q_n"
            )
            col_name_current = f"NINO3.4 SST Q_n-{spread}"
            self.dataframe[col_name_current] = self.dataframe.groupby(
                ["Province", "Crop"]
            )[col_name_prev].shift(1)

        # If the component was constructed with the unknown period omitted,
        #   remove the unknown period (the current reference quarter)
        if not self.include_unknown_period:
            self.dataframe = self.dataframe.drop("NINO3.4 SST Q_n", axis=1)
