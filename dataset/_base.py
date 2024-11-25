# Model related imports

# Dataset related imports
from torch.utils.data import Dataset, random_split, Subset

# Data augmentation/processing imports
import pickle

# Evaluation related imports

# PyTorch related imports
import torch

# Util related imports

# Execution related imports
import os

# Datastructure related imports
import pandas as pd
import numpy as np
import json  # noqa

# Miscellaneous imports
from typing import Any, Optional
from pprint import pprint  # noqa
import pyfiglet


class BaseComponent:
    """Parent class that handles basic operations for components"""

    def __init__(
        self: "BaseComponent",
        source_path: str = "",
        history_spread: int = 0,
        include_unknown_period: bool = False,
        key: Optional[pd.DataFrame] = None,
        save_path: str | bool = "",
    ):
        """Constructor for the base component class or for its children

        Args:
            self (BaseComponent): Instance method of the class
            source_path (str): Path to the source of the component that is
                trying to parse it. IF loading from a file, simply ignore this
                argument. Defaults to empty string.
            history_spread (int): If set, apply a spread from the start of the
                reference point to the `history_spread`'s max time period.
                Defaults to 0.
            include_unknown_period (bool): Boolean value that states whether or
                not the statistics of the component should be seen in the
                future/targeted quarters for prediction/forecasting. True if
                the statistics will be on the same future/targeted quarters.
                During inference, this will assume that the input provided for
                the component are forecasted. False if the statistic will not
                be in the quarter of the future/target quarter. This assumes
                that during inference, the passed-through values to the
                component are known already. Defaults to False.
            key (pd.DataFrame, optional): Dataframe that provides the full list
                of primary keys to ground components to a common keyset. This
                helps standardizes the dataset creation and ensures consistency
                for ease of development. Defaults to None.
            save_path (str): Path to the save file that holds previously
                compiled dataframe information. To load previous data, simply
                set the path value. If a True value is passed, the default path
                to the save file will be loaded. For the opposite, simply
                ignore assigning a value to the path. Defaults to "".
        """

        # Save the values of the passed arguments
        self.source_path = source_path
        self.history_spread = history_spread
        self.include_unknown_period = include_unknown_period
        self.key = key

        # Create an empty variable that holds the component's main dataframe
        self.dataframe: pd.DataFrame = None

        # Call a post initialization method to custom the construction of
        #   the component
        if not save_path:
            self.post_init()

        # If the save path is specified, either get the default file or the
        #   specified file
        else:
            self.load_component(
                "" if isinstance(save_path, bool) else save_path
            )

    def post_init(self: "BaseComponent"):
        """Method to process additional information past the creation of the
        component

        Args:
            self (BaseComponent): Instance method of the class

        Raises:
            NotImplementedError: Raised when the post_init is not defined
                by the children class
        """
        raise NotImplementedError

    def save_component(self: "BaseComponent", path: str = ""):
        """Method to save the component content after compilation

        Args:
            self (BaseComponent): Instance method for the class
            path (str, optional): Path to save the component information.
                If no path is provided, the method will save the component
                using the instance's name. Defaults to "".
        """
        # If the path is None, set it to the default location the caller's
        #   script path
        if not path:
            path = self.__class__.__name__ + ".comp"

        # Save the data
        self.dataframe.to_csv(path, index=False)

    def load_component(self: "BaseComponent", path: str = ""):
        """Method to load the component content from a given path

        Args:
            self (BaseComponent): Instance method for the class
            path (str, optional): Path to save the component information.
                If no path is provided, the method will save the component
                using the instance's name. Defaults to "".
        """
        # If the path is None, set it to the default location the caller's
        #   script path
        if not path:
            path = self.__class__.__name__ + ".comp"

        # Read the component data
        self.dataframe = pd.read_csv(path)


class BaseDataset(Dataset):
    """Base class for data handling"""

    def __init__(
        self: "BaseDataset",
    ):
        """Initializes the BaseDataset class.

        NOTE:
            Implementation of the `BaseDataset` class for children datasets
            must define these variables below in order for the compilation,
            training, evaluation scripts to work. Read the description of these
            variable so that you can get an understanding of what to define
            these variables for an easy use of this framework.

        Args:
            self (BaseDataset): Instance method for the class
        """
        # Generate dataframes for the data. One is the active data that is for
        #   training (self.data). The other dataframe is the cleansed version
        #   which is used for data analysis and model performance evaluation.
        self.data: pd.DataFrame = None
        self.original_data: pd.DataFrame = None

        # Create a basis for the number of features and labels. This is the
        #   number of features and samples used in the `self.data` dataframe.
        #   Used for data analysis and model performance evaluation.
        self.num_features = None
        self.num_labels = None

        # Create variables for the number of SAMPLE features and labels. This
        #   is the number of features and labels (feature and label dimension)
        #   used when training the model. This is different from the previous
        #   because this is the model input representation of the data.
        # NOTE: These numbers is going to be different than the
        #   `self.num_features` and the `self.num_labels` if the dataset is
        # represented in a state format (used for RNNs, LSTMs, etc.)
        self.sample_num_features = None
        self.sample_num_labels = None

    def __len__(self: "BaseDataset") -> int:
        """
        Returns the total number of samples in the dataset.

        Args:
            self (BaseDataset): Instance method for the class

        Returns:
            int: The size of the dataset.
        """
        return len(self.data)

    def __getitem__(
        self: "BaseDataset", idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieves a data sample from the dataset. Assumes that the data
        is represented with a pandas dataframe by default AND that the data is
        scaled after creation of the instance data. Override with a different
        definition if this is not the case.

        Args:
            self (BaseDataset): Instance method for the class
            idx (int): Index of the data sample to retrieve

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The data sample
                corresponding to the given index or the given dataset source.
                Returns a feature, label, and non-state feature, respectively.
        """
        # Grab features and labels as normal if the dataset is not state mode
        features = self.data.iloc[idx, : self.num_features].values
        if not self.state_version or True:
            label = self.data.iloc[idx, -self.num_labels :].values
        non_state_features = features

        # If not, get features as state version
        if self.state_version:
            # Extract and modify the feature return
            sample = self.data.iloc[idx]
            features = []
            for i in range(self.max_non_primary_features):
                insert = sample[self.key_cols].values
                for j in self.component_columns:
                    target_col = self.component_columns[j][i]
                    if target_col:
                        insert = np.hstack(
                            [insert, sample[[target_col]].values]
                        )
                    else:
                        insert = np.hstack([insert, [0]])
                features.append(insert)
            features = np.array(features)

            # Extract the labels
            label = self.data.iloc[idx, -self.num_labels :].values

        # Tensorize the features and labels
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        non_state_features = torch.tensor(
            non_state_features, dtype=torch.float32
        )

        # Return the selection
        return features_tensor, label_tensor, non_state_features

    @staticmethod
    def collater(batch: list | tuple) -> Any:
        """Classmethod that contains the processing functions to collate
        the contents of the batch

        Args:
            batch (list): Batch content from the dataloader.

        Returns:
            Any: Returns the processed batch from the dataloader

        Raises:
            NotImplementedError1: Raised if not implemented
        """
        raise NotImplementedError

    def inprocess(self: "BaseDataset", input_info: dict | list | str) -> Any:
        """Classmethod that contains the inprocessing functions to process
        batch content on the fly. The dataset instance must have the
        tokenizer defined or set before using this method.

        Args:
            self (BaseDataset): Instance method for the class
            input_info (dict | list | str): Content to inprocess

        Returns:
            Any: Returns the processed batch after inprocessing

        Raises:
            NotImplementedError: Raised if not implemented
        """
        raise NotImplementedError

    def preprocess(self: "BaseDataset", **kwargs):
        """Preprocessing step for the dataset. The dataset instance must have
        the tokenizer defined or set before using this method.

        Args:
            self (BaseDataset): Instance method for the class

        Raises:
            NotImplementedError: Raised if not implemented
        """
        raise NotImplementedError

    def split_dataset(
        self: "BaseDataset",
        train_size: float,
        valid_size: Optional[float] = None,
    ) -> list[Subset]:
        """Splits the dataset into training, validation, and optionally testing
        sets. This is done via a random selection process

        Args:
            self (BaseDataset): Instance method for the class
            train_size (float): Proportion of the dataset to include in the
                train split.
            valid_size (Optional[float]): Proportion of the dataset to include
                in the validation split. If None, the remainder of the dataset
                after the train split is used as the validation set. If
                provided, the remainder is split into validation and test sets.

        Returns:
            list[Subset]: A list containing datasets for training,
                validation, and optionally testing.
        """
        # If the val_size is not provided, use the given train_size value to
        # specify the training dataset size and the remainder as the validation
        # dataset size
        if valid_size is None:
            train_len = int(len(self) * train_size)
            val_len = len(self) - train_len
            return random_split(self, [train_len, val_len])

        # If provided, split up the portion of the data for training dictated
        # by train_size. Afterwards, split it up for the validation dataset by
        # the val_size value. The remainder is left for the testing dataset
        else:
            train_len = int(len(self) * train_size)
            val_len = int(len(self) * valid_size)
            test_len = len(self) - train_len - val_len
            if test_len < 0:
                raise ValueError(
                    "train_size and val_size sum exceed dataset size"
                )
            return random_split(self, [train_len, val_len, test_len])

    def save_dataset(self: "BaseDataset", path: str):
        """Method to save the dataset as a pickle file

        Args:
            self (BaseDataset): Instance method for the class
            path (str): Path of where to save the dataset
        """
        torch.save(self, path)

    @staticmethod
    def load_dataset(path: str) -> "BaseDataset":
        """Method to save the dataset as a pickle file. This method is static.

        Args:
            path (str): Path of where to load the dataset from
        """
        return torch.load(path)

    def save_split(
        self: "BaseDataset", split: Subset, name: str, directory: str
    ):
        """Method to save split information about a dataset

        Args:
            self (BaseDataset): Instance method for the class
            split (Subset): Split to save
            name (str): Name of the split
            directory (str): Directory of where to save the split
        """
        # Make the directory if not made already
        os.makedirs(directory, exist_ok=True)

        # Get the indices of the split
        split_indices = split.indices if isinstance(split, Subset) else split

        # Save the split
        with open(os.path.join(directory, f"split_{name}.pkl"), "wb") as f:
            pickle.dump(split_indices, f)

    def load_split(
        self: "BaseDataset", dataset: "BaseDataset", split_path: str
    ) -> Subset:
        """Method to load a split to the dataset

        Args:
            self (BaseDataset): Instance method for the class
            dataset (BaseDataset): Dataset instance to assign subsets to
            split_path (str): Path to the split

        Returns:
            Subset: Split information pulled from the file
        """
        with open(split_path, "rb") as f:
            return Subset(dataset, pickle.load(f))

    def _info(self: "BaseDataset"):
        """Debugging method for the dataset instance. Assumes that the dataset
        information is in a pandas DataFrame.

        Args:
            self (BaseDataset): Instance method for the class
        """
        print(pyfiglet.figlet_format("INFO", font="big"))
        print(
            "<!> Some information may be incorrect if called before critical "
            "points\n\n"
        )
        print("Size:", self.__len__())
        print("# Features:", self.num_features)
        print("# Labels:", self.num_labels)
        print("\n")
