# Dataset related imports
from dataset import BaseDataset

# Evaluation related imports
import evaluation

# PyTorch related imports
import torch

# Util related imports
from utils.dict_obj import DictObj

# Execution related imports
import torch.distributed as dist
from accelerate import Accelerator
import random
import os

# Datastructure related imports
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing  # noqa
import pandas as pd  # noqa
import json  # noqa

# Miscellaneous imports
from typing import Optional
import re


def _clean_config(cfg: dict | list) -> dict:
    """Recursively cleans the config dictionary by removing empty keys and
    promoting nested values to the nearest named key level

    Args:
        cfg (dict): The JSON-like dictionary to clean

    Returns:
        dict: The cleaned dictionary with no empty keys and flattened structure
    """

    # Check if the input is a dictionary
    if isinstance(cfg, dict):
        # Create a tracking dictionary
        clean_data = {}

        # Iterate through each key-value pair in the dictionary
        for k, v in cfg.items():
            # If the key is empty, clean its value and update the clean_data
            if k == "":
                if isinstance(v, dict):
                    clean_data.update(_clean_config(v))

            # Otherwise, clean the value and add it to clean_data
            else:
                clean_data[k] = _clean_config(v)

        # Return the cleaned dictionary
        return clean_data

    # If the input is a list, recursively clean each item in the list
    elif isinstance(cfg, list):
        return [_clean_config(item) for item in cfg]

    # If the input is neither a dictionary nor a list, return it as is
    else:
        return cfg


def prepare_config(cfg: DictConfig) -> DictConfig:
    """Function to prepare the config of a run session before execution

    Args:
        cfg (DictConfig): Config to cleanse, prepare, and organize

    Returns:
        DictConfig: Prepared configuration for usage
    """

    # Register any custom resolvers
    OmegaConf.register_new_resolver(
        "env", lambda var, default="": os.getenv(var, default)
    )
    OmegaConf.register_new_resolver(
        "random", lambda a, b: random.randint(a, b)
    )

    # Cleanse the configuration
    cfg = _clean_config(OmegaConf.to_container(cfg))
    cfg = OmegaConf.create(cfg)

    # Return the prepared configuration
    return cfg


def create_output_paths(
    base_path: str,
    provided_number: Optional[int | str] = None,
    sub_directories: Optional[list] = None,
    make_dirs: bool = True,
) -> str:
    """Creates the output directory with its necessary sub-directories

    Args:
        base_path (str): Base path of the output path.
        provided_number (int | str, optional): Override number if a number
            is provided for the folder
        sub_directories (list, optional): Additional subdirectories.
            Defaults to None.
        make_dirs (bool): Whether or not to make the directories. This is used
            to prevent non-main processes from creating multiple folders for a
            single session. Defaults to True.

    Returns:
        str: The final output path for the script
    """

    # Extract the base name from the base path
    base_name = os.path.basename(base_path)

    # Define the pattern to search for existing directories
    pattern = re.compile(rf"{re.escape(base_name)}___(\d+)")

    # List existing directories that match the pattern
    parent_path = os.path.dirname(base_path)
    os.makedirs(parent_path, exist_ok=True)
    dirs = [d for d in os.listdir(parent_path) if pattern.match(d)]

    # Determine the next suffix number
    if provided_number:
        next_number = str(provided_number)
    elif not dirs:
        next_number = 1
    else:
        existing_numbers = [int(pattern.match(d).group(1)) for d in dirs]
        next_number = max(existing_numbers) + 1

    # Construct the result path
    result_path = os.path.join(parent_path, f"{base_name}___{next_number}")

    # Create main directory if required
    if make_dirs:
        # Make the base directory
        os.makedirs(result_path, exist_ok=True)

        # Create sub-directories
        if sub_directories:
            if not isinstance(sub_directories, list):
                raise ValueError(
                    "Values passed into `sub_directories` needs to be a `list`"
                )

            for item in sub_directories:
                os.makedirs(os.path.join(result_path, item), exist_ok=True)

    # Return the final path
    return result_path


def dataset_reader(base_path: str) -> DictObj[str, BaseDataset]:
    """Method to return dataset classes of the dataset that were compiled

    Args:
        base_path (str): Path of the directory that has these datasets

    Returns:
        DictObj[str, BaseDataset]: Dictionary class of the compiled datasets
    """
    # Variable declaration
    total = None
    train = None
    valid = None
    test = None

    # Traverse the directory
    for root, _, files in os.walk(base_path):
        for file in files:
            # Read and save the dataset content
            if (
                file.endswith(".pth")
                and not any([i in file for i in ["train", "valid", "test"]])
                and total is None
            ):
                total = BaseDataset.load_dataset(os.path.join(root, file))
            if file.endswith(".train.pth") and train is None:
                train = BaseDataset.load_dataset(os.path.join(root, file))
            if file.endswith(".valid.pth") and valid is None:
                valid = BaseDataset.load_dataset(os.path.join(root, file))
            if file.endswith(".test.pth") and test is None:
                test = BaseDataset.load_dataset(os.path.join(root, file))

            # If all files are found, break the loop
            if (
                total is not None
                and train is not None
                and valid is not None
                and test is not None
            ):
                break

    # Return the results
    return DictObj(
        {
            "total": total,
            "train": train,
            "valid": valid,
            "test": test,
        }
    )


def broadcast_string(
    value: str,
    accelerator: Accelerator,
    src: int = 0,
) -> str:
    """Broadcasts a string from the source process to all other processes

    Args:
        value (str): The string value to broadcast
        accelerator (Accelerator): The accelerator instance to apply
            broadcast and other communication between processes
        src (int): The source process rank. Defaults to 0.

    Returns:
        str: The broadcasted string value received by all processes.
    """

    # Convert the string to a byte tensor
    value_tensor = torch.tensor(
        bytearray(value, "utf-8"), dtype=torch.uint8
    ).to(accelerator.device)

    # Create a tensor containing the length of the byte tensor
    length_tensor = torch.tensor([len(value_tensor)], dtype=torch.int).to(
        accelerator.device
    )

    # Broadcast the length tensor from the source process to all processes
    dist.broadcast(length_tensor, src)

    # Resize the byte tensor in all processes to match the length from the
    # source process
    value_tensor.resize_(length_tensor.item())

    # Broadcast the byte tensor data from the source process to all processes
    dist.broadcast(value_tensor, src)

    # Convert the byte tensor back to a string and return
    return value_tensor.cpu().numpy().tobytes().decode("utf-8")


def build_evaluation_list(
    config: DictConfig,
) -> list[evaluation.BaseEvaluation]:
    """Builds a list of evaluation classes to test a model's performance

    Args:
        config (DictConfig): Configurations for the transform function

    Returns:
        list[evaluation.BaseEvaluation]: Resulting list of evaluation systems
    """

    # Define a tracker variable
    evaluation_list = []

    # Iterate through the config
    for item in config:
        evaluation_class = getattr(evaluation, item.name)
        args = item.get("args", {})
        transform_component = evaluation_class(**args)
        evaluation_list.append(transform_component)

    # Return the evaluation list
    return evaluation_list
