# Set the python path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# Model related imports

# Dataset related imports
import dataset

# Data augmentation/processing imports

# Evaluation related imports

# PyTorch related imports

# Util related imports
from utils.scripts import create_output_paths

# Execution related imports
import hydra
import json
import copy

# Datastructure related imports
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Miscellaneous imports
import random


@hydra.main(
    version_base=None,
    config_path="../configs/",
    config_name="fcnn_6",
)
def main(cfg: DictConfig):
    """Main run function with config wrapper

    Args:
        cfg (DictConfig): config object
    """

    #
    #   ? 🥾 SETUP PROCESS
    #   region
    #

    # Resolve any system placeholder in the config
    OmegaConf.register_new_resolver(
        "env", lambda var, default="": os.getenv(var, default)
    )
    OmegaConf.register_new_resolver(
        "random", lambda a, b: random.randint(a, b)
    )

    # Save the model's output path
    OUTPUT_PATH = create_output_paths(
        os.path.join(cfg.output_path, "dataset_compilation")
    )

    #   endregion

    #
    #   ? 💽 DATASET COMPILATION
    #   region
    #

    # Generate the key component
    component_key: dataset.BaseComponent = getattr(
        dataset, cfg.dataset.components.key.component
    )(**cfg.dataset.components.key.args)
    component_key.save_component(f"{OUTPUT_PATH}/PalayKeyComponent.comp")

    # Generate feature components
    feature_components = []
    for item in tqdm(
        cfg.dataset.components.features,
        desc="Creating dataset FEATURE components",
    ):
        feature_component: dataset.BaseComponent = getattr(
            dataset, item.component
        )(**item.args, key=component_key.dataframe)
        feature_component.save_component(
            f"{OUTPUT_PATH}/{item.component}.comp"
        )
        feature_components.append(feature_component)

    # Generate label components
    label_components = []
    for item in tqdm(
        cfg.dataset.components.labels,
        desc="Creating dataset LABEL components",
    ):
        label_component: dataset.BaseComponent = getattr(
            dataset, item.component
        )(**item.args, key=component_key.dataframe)
        label_component.save_component(f"{OUTPUT_PATH}/{item.component}.comp")
        label_components.append(label_component)

    # Create a dataset out of the components
    data: dataset.BaseDataset = getattr(dataset, cfg.dataset.dataset)(
        key_component=component_key,
        feature_components=feature_components,
        label_components=label_components,
        categorical_encoder=cfg.dataset.encoding,
        state_version=cfg.dataset.state_version,
    )

    # Save the entire dataset
    data.data.to_csv(f"{OUTPUT_PATH}/scaled_dataset.csv", index=False)
    data.original_data.to_csv(
        f"{OUTPUT_PATH}/original_dataset.csv", index=False
    )
    data.save_dataset(os.path.join(OUTPUT_PATH, "dataset.pth"))

    # Split the dataset
    train, valid, test = data.split_dataset(**cfg.dataset.split_args)

    # Save the splits in both .pkl format and the formatted dataset in .pth
    data.save_split(train, "train", OUTPUT_PATH)
    data.save_split(valid, "valid", OUTPUT_PATH)
    data.save_split(test, "test", OUTPUT_PATH)
    train_data = copy.deepcopy(data)
    valid_data = copy.deepcopy(data)
    test_data = copy.deepcopy(data)
    train_data.data = train.dataset.data.iloc[train.indices]
    valid_data.data = valid.dataset.data.iloc[valid.indices]
    test_data.data = test.dataset.data.iloc[test.indices]
    train_data.original_data = train.dataset.original_data.iloc[train.indices]
    valid_data.original_data = valid.dataset.original_data.iloc[valid.indices]
    test_data.original_data = test.dataset.original_data.iloc[test.indices]
    train_data.save_dataset(os.path.join(OUTPUT_PATH, "dataset.train.pth"))
    valid_data.save_dataset(os.path.join(OUTPUT_PATH, "dataset.valid.pth"))
    test_data.save_dataset(os.path.join(OUTPUT_PATH, "dataset.test.pth"))

    # Save the scaled splits
    train.dataset.data.iloc[train.indices].to_csv(
        f"{OUTPUT_PATH}/scaled_dataset.train.csv", index=False
    )
    valid.dataset.data.iloc[valid.indices].to_csv(
        f"{OUTPUT_PATH}/scaled_dataset.valid.csv", index=False
    )
    test.dataset.data.iloc[test.indices].to_csv(
        f"{OUTPUT_PATH}/scaled_dataset.test.csv", index=False
    )

    # Save the original splits
    train.dataset.original_data.iloc[train.indices].to_csv(
        f"{OUTPUT_PATH}/original_dataset.train.csv", index=False
    )
    valid.dataset.original_data.iloc[valid.indices].to_csv(
        f"{OUTPUT_PATH}/original_dataset.valid.csv", index=False
    )
    test.dataset.original_data.iloc[test.indices].to_csv(
        f"{OUTPUT_PATH}/original_dataset.test.csv", index=False
    )

    # Output information about the split
    with open(f"{OUTPUT_PATH}/split_info.txt", "w") as file:
        file.write(f"Train Size: {len(train)}\n")
        file.write(f"Valid Size: {len(valid)}\n")
        file.write(f"Test Size: {len(test)}")

    # Write the input config:
    with open(f"{OUTPUT_PATH}/config.json", "w") as file:
        file.write(json.dumps(dict(OmegaConf.to_container(cfg)), indent=4))

    # Print the information about the dataset curation
    data._info()

    #   endregion


# Run the main function
if __name__ == "__main__":
    main()
