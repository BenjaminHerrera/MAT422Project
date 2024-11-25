# Model related imports

# Dataset related imports
from . import BaseDataset, BaseComponent
from torch.utils.data import Subset

# Data augmentation/processing imports
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Evaluation related imports

# PyTorch related imports

# Util related imports

# Execution related imports

# Datastructure related imports
import pandas as pd
import numpy as np
import json  # noqa

# Miscellaneous imports
from typing import Literal, Optional
from pprint import pprint  # noqa


class PalayDataset(BaseDataset):
    """Dataset definition where the time series dataset is defined into one
    instance. For example, a singular reference point at year Y and Q_q
    would look like this:

        Column        | Value
        --------------+---------
        Province      | ...
        Year          | ...
        Quarter       | ...
        Crop          | ...
        Comp_1 Q_q    | ...
        Comp_1 Q_q-1  | ...
        Comp_1 Q_q-2  | ...
        Comp_1 Q_q-3  | ...
        Comp_1 Q_q-4  | ...
        ...           | ...
        Comp_n Q_q    | ...
        Comp_n Q_q-1  | ...
        Comp_n Q_q-2  | ...
        Comp_n Q_q-3  | ...
        Comp_n Q_q-4  | ...

    It can also be defined in a state format. For example, the same format
    above can be translated into this:

        Column   |  Value
        ---------+------------------------------------------------------------
                 |  ┌                                                       ┐
        Province |  │┌   ...   ┐┌   ...   ┐┌   ...   ┐┌   ...   ┐┌   ...   ┐│
        Year     |  ││   ...   ││   ...   ││   ...   ││   ...   ││   ...   ││
        Quarter  |  ││   ...   ││   ...   ││   ...   ││   ...   ││   ...   ││
        Crop     |  ││   ...   ││   ...   ││   ...   ││   ...   ││   ...   ││
        Comp_1   |  ││   ...   ││   ...   ││   ...   ││   ...   ││   ...   ││
        ...      |  ││   ...   ││   ...   ││   ...   ││   ...   ││   ...   ││
        Comp_n   |  │└   ...   ┘└   ...   ┘└   ...   ┘└   ...   ┘└   ...   ┘│
                 |  └                                                       ┘
                 |       Q_q       Q_q-1      Q_q-2      Q_q-3      Q_q-4

    The above sample has a sequence length of 5 quarters, and 4+n features.
    Note that the feature and label components passed will have the same
    key components.
    """

    def __init__(
        self: "PalayDataset",
        key_component: BaseComponent,
        feature_components: list[BaseComponent],
        label_components: list[BaseComponent],
        categorical_encoder: Optional[Literal["one-hot", "label"]] = None,
        data: Optional[pd.DataFrame] = None,
        state_version: bool = False,
    ):
        """Constructor for the PalayDataset where it takes a list
        of components and a primary key component to combine everything
        together.

        Args:
            self (PalayDataset): Instance method of the class
            key_component (BaseComponent): Component instance that houses the
                primary key of the dataset
            feature_components (list[BaseComponent]): Component instance that
                houses the features of the dataset
            label_components (list[BaseComponent]): Component instance that
                houses the labels for the dataset
            categorical_encoder (Literal["one-hot", "label"], optional):
                type of categorical encoder to use on the dataset. If None,
                then the data is going to be kept as is. If not, uses either
                uses one-hot or label encoding. Any other value results in
                an error. Defaults to None.
            state_version (bool): Whether or not to represent the data in
                a state format
        """

        # Pull attributes and methods from the parent class
        super().__init__()

        # Save the components of the dataset
        self.key_component = key_component
        self.feature_components = feature_components
        self.label_components = label_components
        self.state_version = state_version

        # Get column names of the key component
        self.key_cols = self.key_component.dataframe.columns

        # If data is provided, populate the dataset variable with that info
        if isinstance(data, pd.DataFrame):
            self.data = data

        # Combine the components together
        components = feature_components + label_components
        self.data = key_component.dataframe.copy()
        for df in components:
            self.data = pd.merge(
                self.data,
                df.dataframe,
                on=list(self.key_cols),
                how="outer",
            )

        # TODO: Confirm technique for cleaning dataset
        # Remove samples with no or missing data
        self.data.replace(0, np.nan, inplace=True)
        self.data.dropna(inplace=True)

        # Save the clean original data
        self.original_data = self.data.copy()

        # Apply one-hot or label encoding to the columns that are categorical
        self.categorical_encoder = categorical_encoder
        self.label_encoders = {}
        if categorical_encoder == "one-hot":
            original_columns = self.data.columns.tolist()
            self.data = pd.get_dummies(self.data, dtype=int)
            self.original_categorical_columns = self.data.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            new_columns = self.data.columns.tolist()
            added_columns = [
                col for col in new_columns if col not in original_columns
            ]
            self.data = self.data[
                added_columns
                + [col for col in original_columns if col in self.data.columns]
            ]
        elif categorical_encoder == "label":
            for col in self.data.select_dtypes(include=["object"]).columns:
                label_encoder = LabelEncoder()
                self.data[col] = label_encoder.fit_transform(self.data[col])
                self.label_encoders[col] = label_encoder
        else:
            raise TypeError(
                "`categorical_encoder` argument not set properly."
                " Set to `one-hot` or `label`"
            )

        # Get the number of features and the number of labels
        label_cols = set()
        for i in label_components:
            label_cols |= set(i.dataframe.columns)
        self.num_labels = len(label_cols - set(self.key_cols))
        self.num_features = len(self.data.columns) - self.num_labels

        # Scale the data with StandardScaler
        self.feature_scaler = StandardScaler()
        self.label_scaler = StandardScaler()
        features = self.data.iloc[:, : self.num_features]
        labels = self.data.iloc[
            :, self.num_features : self.num_features + self.num_labels
        ]
        scaled_features = pd.DataFrame(
            self.feature_scaler.fit_transform(features),
            columns=features.columns,
        )
        scaled_labels = pd.DataFrame(
            self.label_scaler.fit_transform(labels),
            columns=labels.columns,
        )
        self.data = pd.concat([scaled_features, scaled_labels], axis=1)

        # If the state version is specified, create stats of the components
        #   for quick fetching of state samples
        if self.state_version:
            # Track the columns of each component
            self.component_columns = {}
            for i in feature_components:
                insert = list(set(i.dataframe.columns) - set(self.key_cols))
                insert.sort()
                self.component_columns[i.__class__.__name__] = insert[::-1]

            # Determine the maximum length of the lists
            self.max_non_primary_features = max(
                len(v) for v in self.component_columns.values()
            )

            # Fill in None values to the end of the lists where needed
            for key in self.component_columns:
                while (
                    len(self.component_columns[key])
                    < self.max_non_primary_features
                ):
                    self.component_columns[key].append(None)

            # Set the number of SAMPLE feature and labels
            self.sample_num_features = self[0][0].shape[1]
            self.sample_num_labels = self[0][1].shape[0]

        # If not, simply set the sample label and feature sizes
        else:
            self.sample_num_features = self[0][0].shape[0]
            self.sample_num_labels = self[0][1].shape[0]

    def split_dataset(
        self: "PalayDataset",
        train_size: float,
        valid_size: Optional[float] = None,
        by_year: bool = True,
    ) -> list[Subset]:
        """Splits the dataset into training, validation, and optionally testing
        sets. This is done via a random selection process or via the year. This
        method is partially overridden to accompanying both the original
        definition in `BaseDataset` and the "by year" splitting scheme.

        Args:
            self (BaseDataset): Instance method for the class
            train_size (float): Proportion of the dataset to include in the
                train split.
            valid_size (Optional[float]): Proportion of the dataset to include
                in the validation split. If None, the remainder of the dataset
                after the train split is used as the validation set. If
                provided, the remainder is split into validation and test sets.
            by_year (bool). Whether or not to split the dataset by year. If
                this argument is True, then the method will use the provided
                train size and valid size as percentages for selecting the
                year. If not, it will call the original `split_dataset` method
                in `BaseDataset`. Defaults to True.

        Returns:
            list[Subset]: A list containing datasets for training,
                validation, and optionally testing.
        """

        # If the caller does not want it to be split by the year, call the
        #   original method
        if not by_year:
            return super().split_dataset(train_size, valid_size)

        # Reset the DataFrame index to ensure indices align properly
        proper_indices = self.original_data.reset_index(drop=True)

        # Group the data by year and convert to a dict of lists of indices
        data_by_year = (
            proper_indices.groupby("Year")
            .apply(lambda x: x.index.tolist())
            .to_dict()
        )

        # Sort the years for consistent processing
        years = sorted(data_by_year.keys())
        num_years = len(years)

        # Determine the number of years to be used for training and validation
        num_train_years = int(np.floor(train_size * num_years))
        num_val_years = (
            int(np.floor(valid_size * num_years))
            if valid_size is not None
            else num_years - num_train_years
        )

        # Select the years for training, validation, and testing
        train_years = years[:num_train_years]
        val_years = years[num_train_years : num_train_years + num_val_years]
        test_years = years[num_train_years + num_val_years :]

        # Gather the indices for the training, validation, and testing subsets
        train_indices = [
            idx for year in train_years for idx in data_by_year[year]
        ]
        val_indices = [idx for year in val_years for idx in data_by_year[year]]
        test_indices = [
            idx for year in test_years for idx in data_by_year[year]
        ]

        # Create the Subset objects for training, validation, and testing
        train_subset = Subset(self, train_indices)
        val_subset = Subset(self, val_indices)
        test_subset = (
            Subset(self, test_indices) if valid_size is not None else None
        )

        # Return the Subset objects and include test_subset only if
        #   valid_size is provided
        if test_subset is not None:
            return [train_subset, val_subset, test_subset]
        else:
            return [train_subset, val_subset]

    def inverse_transform(
        self: "PalayDataset",
        value: float | list[float],
        label_only: bool = True,
    ) -> np.ndarray:
        """Method to convert scaled target values into its interpreted values

        Args:
            self (PalayDataset): Instance method for the class
            value (float | list[float]): Value(s) to interpret
            label_only (bool): Whether or not to use the label only scaler.
                Defaults to True.

        Returns:
            np.ndarray: Interpreted values based on the given value and type
        """
        # Restructure the value argument into a list if a float was provided
        if isinstance(value, float):
            value = [value]

        # Convert the value into a dataframe
        slice_specification = (
            slice(-self.num_labels, None)
            if label_only
            else slice(None, -self.num_labels)
        )
        value = pd.DataFrame(
            value, columns=self.original_data.columns[slice_specification]
        )

        # Return the inverse of the value (interpretation) if we are only
        # focused on the label columns
        if label_only:
            return self.label_scaler.inverse_transform(value)

        # If focusing on the feature columns, apply a different inverse
        #   technique respective to the type of encoding done to categorical
        #   columns
        else:
            # Inverse the values with the feature scaler
            value = self.feature_scaler.inverse_transform(value)

            # Recreate the dataframe
            value = pd.DataFrame(
                value,
                columns=self.original_data.columns[slice_specification],
            )

            # Check if the encoding was one-hot encoding
            #! ❌ UNTESTED
            if self.categorical_encoder == "one-hot":
                # Create a copy of the DataFrame to store the original
                #   categories
                original_df = pd.DataFrame(index=value.index)

                # Loop through each original categorical column
                for column in self.original_categorical_columns:
                    # Find the one-hot encoded columns that correspond to this
                    #   original column
                    one_hot_cols = [
                        col
                        for col in value.columns
                        if col.startswith(column + "_")
                    ]

                    if one_hot_cols:
                        # Reverse the one-hot encoding
                        original_df[column] = (
                            value[one_hot_cols]
                            .idxmax(axis=1)
                            .str[len(column) + 1 :]
                        )

                        # Drop the one-hot encoded columns from the value
                        #   DataFrame
                        value = value.drop(columns=one_hot_cols)

                # Combine the remaining numerical columns with the original
                #   categories
                value = pd.concat([value, original_df], axis=1)

            # If the encoding was label encoding
            elif self.categorical_encoder == "label":
                # Apply the inverse of the encoding to categorical columns
                for column, encoder in self.label_encoders.items():
                    if column in value.columns:
                        value[column] = encoder.inverse_transform(
                            value[column].round().astype(int)
                        )

            # Return the NumPy version of the finalized inversions
            return value.to_numpy()
