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


class PalayKeyComponent(BaseComponent):
    """Primary key of the PalayDataset. This class is the key for all other
    components that composes the PalayDataset. Contains the following:

    - Province* (Literal[0, ..., 81])
    - Year* (Literal[1987, ..., 2024])
    - Quarter* (Literal[1, 2, 3, 4])
    - Crop* (Literal["Irrigated Palay", "Rainfed Palay"])

    Attribute names with the "*" next to them are considered to be primary keys

    This code is meant for datasource from the following source location:
        https://openstat.psa.gov.ph/PXWeb/pxweb/en/DB/DB__2E__CS/0022E4EAHC0.px/?rxid=bdf9d8da-96f1-41-00-ae09-18cb3eaeb313

            OR

        https://openstat.psa.gov.ph/PXWeb/pxweb/en/DB/DB__2E__CS/0012E4EVCP0.px/?rxid=bdf9d8da-96f1-41-00-ae09-18cb3eaeb313
    """

    def post_init(self: "PalayKeyComponent"):
        """Post initialization method to sort out the content for the province
        keys

        Args:
            self (PalayKeyComponent): Instance method of the class
        """

        # Translate the CSV to a pandas dataframe
        df = pd.read_csv(self.source_path, skiprows=1)

        # Save information about the time column names with their hierarchy.
        # This is used for sorting the datasamples in time series graphs
        self.time_cols = ["Year", "Quarter"]

        # Get the unique croptypes and trim the dataset
        croptypes = list(df["Ecosystem/Croptype"].unique())
        df = df.head(83)

        # Get a list of provinces and crop the first two columns of the dataset
        provinces = [i[4:] for i in list(df["Geolocation"])]
        df = df.iloc[:, 2:]

        # Create a dict to formulate the pandas dataframe to the component
        main_dict = {"Province": [], "Year": [], "Quarter": [], "Crop": []}

        # Iterate through the dataset
        for row_index in range(len(df)):
            for col_index in range(len(df.columns)):
                # Extract the row index and the column name
                row_name = df.index[row_index]
                col_name = df.columns[col_index]

                # Add to the province in the main_dict
                main_dict["Province"].append(provinces[row_name])

                # Split the column name
                col_name = col_name.split(" ")

                # Add the year and quarter to the main_dict
                main_dict["Year"].append(int(col_name[0]))
                main_dict["Quarter"].append(int(col_name[-1]))

        # Add the other croptypes
        for i in croptypes:
            main_dict["Crop"] += [i] * len(main_dict[next(iter(main_dict))])
        for i in main_dict:
            if i == "Crop":
                continue
            main_dict[i] *= len(croptypes)

        # Formalize the dataframe for the component
        self.dataframe = pd.DataFrame(main_dict)
