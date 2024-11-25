# Set the python path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# Execution related imports
from hydra.core.hydra_config import HydraConfig  # noqa
import hydra

# Datastructure related imports
from omegaconf import DictConfig, OmegaConf

# ! PLAYGROUND IMPORTS
import random
import json  # noqa


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


@hydra.main(version_base=None, config_path="./config/", config_name="main")
def my_app(cfg: DictConfig):
    # Resolve any system placeholder in the config and configurations
    OmegaConf.register_new_resolver(
        "env", lambda var, default="": os.getenv(var, default)
    )
    OmegaConf.register_new_resolver(
        "random", lambda a, b: random.randint(a, b)
    )
    print(json.dumps(OmegaConf.to_container(cfg), indent=4))
    cfg = OmegaConf.create(
        _clean_config(OmegaConf.to_container(cfg))
    )
    print("\n\n" * 3)
    print(OmegaConf.to_yaml(cfg))
    print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    my_app()
