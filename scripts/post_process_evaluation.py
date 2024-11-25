# Set the python path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Model related imports
import models

# Dataset related imports
from torch.utils.data import DataLoader

# Evaluation related imports
import evaluation

# PyTorch and training related imports
import torch

# Util imports
from utils.scripts import (
    create_output_paths,
    prepare_config,
    dataset_reader,
    broadcast_string,
    build_evaluation_list,
)
from utils.evaluation import post_process
from utils.dict_obj import DictObj

# Execution related imports
from accelerate import Accelerator
import itertools

# Datastructure related imports
from omegaconf import DictConfig, OmegaConf
from collections import OrderedDict
import geopandas as gpd
import pandas as pd  # noqa
import numpy as np
import json  # noqa

# Miscellaneous imports
from pprint import pprint  # noqa
from tqdm import tqdm
import pyfiglet
import hydra


# ! 🧰 DEBUG CONTROL VALUES
# !     NOTE: THESE ARE HARDCODED!
INFERENCE_CONTROL = None


def execute_analysis(
    selection: pd.DataFrame,
    df: pd.DataFrame,
    dir: str,
    num_labels: int,
    num_features: int,
    evaluation_list: list[evaluation.BaseEvaluation],
    key_values: list[str],
    area_type: str,
    missing_area_keys: set[str],
    area_gdf: gpd.GeoDataFrame,
    area_mapping: dict[str, str],
    group: list[str],
):
    """Analysis execution function. This function is made to make code clean

    Args:
        selection (pd.DataFrame): Selected portion of the dataset based on
            a selection (group by) provided by some for loop iterating over
            an enumeration of subset of primary keys
        df (pd.DataFrame): Larger dataframe to analyze columns of the dataset
        dir (str): Main output path of the analyzed dataframe
        num_labels (int): Number of labels in the dataset
        num_features (int): Number of features in the dataset
        evaluation_list (list[evaluation.BaseEvaluation]): List of evaluators
        key_values (list[str]): Values of the enumerated columns. This is
            important as it is used to figure out what selections of the
            dataframe are made for analysis
        area_type (str): Column name that pertains to areas in the dataset
        missing_area_keys (set[str]): Set of missing areas to delete
        area_gdf (gpd.GeoDataFrame): GeoDataFrame to make maps
        area_mapping (dict[str, str]): Dictionary that maps result dataframe
            names to what is provided in the GeoDataFrame
        group (list[str]): List of columns in the iterated enumeration
    """

    # Iterate through every label in the dataset
    labels = selection.iloc[
        :,
        df.shape[1] - (num_labels * 2) - 1 : df.shape[1] - num_labels - 1,
    ].columns
    for label in labels:
        # Create an area tracking dictionary
        area_tracking_data = {}

        # Calculate the performance of the runs
        evaluation_results = {}
        for item in evaluation_list:
            evaluation_results.update(
                item(
                    selection[label + " PREDICTION"],
                    selection[label],
                    pd.DataFrame(
                        columns=[f"col_{i}" for i in range(num_features)]
                    ),
                )
            )
        evaluation_results["n"] = len(selection[label + " PREDICTION"])

        # Calculate the base naming for the files
        if key_values:
            base_file_name = f"{'; '.join(key_values)} -- {label}"
        else:
            base_file_name = f"{label}"

        # Create area performances
        for area in df[area_type.capitalize()].unique():
            # Get a dataframe of the selection based on the iterated area
            area_selection = selection[
                selection[area_type.capitalize()] == area
            ]

            # Create evaluation results for the area selection
            area_evaluation_results = {}
            for item in evaluation_list:
                area_evaluation_results.update(
                    item(
                        area_selection[label + " PREDICTION"],
                        area_selection[label],
                        pd.DataFrame(
                            columns=[f"col_{i}" for i in range(num_features)]
                        ),
                    )
                )
            area_evaluation_results["n"] = len(
                area_selection[label + " PREDICTION"]
            )

            # Add the evaluations to the tracking data
            area_tracking_data[area] = area_evaluation_results

        # Create the area performance dataframe
        area_performances_df = pd.DataFrame.from_dict(
            area_tracking_data, orient="index"
        ).reset_index()
        area_performances_df = area_performances_df.rename(
            columns={"index": area_type.capitalize()}
        )
        area_performances_df = area_performances_df[
            ~area_performances_df[area_type.capitalize()].isin(
                missing_area_keys
            )
        ]

        # > SCATTER PLOT
        post_process.ground_truth_to_pred_graph(
            os.path.join(dir, f"{base_file_name} -- SCATTER.png"),
            selection[label].tolist(),
            selection[label + " PREDICTION"].tolist(),
            evaluation_results,
        )

        # > DISTRIBUTION GRAPH
        post_process.error_distribution_graph(
            os.path.join(dir, f"{base_file_name} -- DISTRIBUTION.png"),
            selection[label].tolist(),
            selection[label + " PREDICTION"].tolist(),
            evaluation_results,
        )

        # Create maps if the area is not in the group
        if area_type.capitalize() not in group:
            # Create area performances
            for area in df[area_type.capitalize()].unique():
                # Get a dataframe of the selection based on the iterated area
                area_selection = selection[
                    selection[area_type.capitalize()] == area
                ]

                # Create evaluation results for the area selection
                area_evaluation_results = {}
                for item in evaluation_list:
                    area_evaluation_results.update(
                        item(
                            area_selection[label + " PREDICTION"],
                            area_selection[label],
                            pd.DataFrame(
                                columns=[
                                    f"col_{i}" for i in range(num_features)
                                ]
                            ),
                        )
                    )
                area_evaluation_results["n"] = len(
                    area_selection[label + " PREDICTION"]
                )

                # Add the evaluations to the tracking data
                area_tracking_data[area] = area_evaluation_results

            # Create the area performance dataframe
            area_performances_df = pd.DataFrame.from_dict(
                area_tracking_data, orient="index"
            ).reset_index()
            area_performances_df = area_performances_df.rename(
                columns={"index": area_type.capitalize()}
            )
            area_performances_df = area_performances_df[
                ~area_performances_df[area_type.capitalize()].isin(
                    missing_area_keys
                )
            ]

            # > AREA PERFORMANCE MAP
            post_process.generate_map(
                area_performances_df,
                area_gdf,
                area_mapping,
                os.path.join(
                    dir,
                    f"{base_file_name}" " -- MAP ({metric}).png",
                ),
                True,
                area_type.capitalize(),
            )


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

    # Resolve any system placeholder in the config and configurations
    cfg = prepare_config(cfg)

    # Print the configuration
    print(OmegaConf.to_yaml(cfg))

    #
    #   ? 🥾 SETUP PROCESS
    #   region
    #

    # Initialize Accelerator
    accelerator = Accelerator()

    # Simplify reference to the action configuration
    action = cfg.action.post_process_analysis

    # Save and create the model train session's output path
    OUTPUT_PATH = ""
    os.makedirs(cfg.output_path, exist_ok=True)
    if accelerator.is_main_process:
        OUTPUT_PATH = create_output_paths(
            os.path.join(cfg.output_path, "post_process_evaluation"),
            cfg.get("number", None),
        )
    accelerator.wait_for_everyone()
    OUTPUT_PATH = broadcast_string(OUTPUT_PATH, accelerator)

    # Extrapolate the datasets and get number of features, labels, etc.
    datasets = dataset_reader(cfg.dataset_path)
    columns = datasets.total.original_data.columns
    key_cols = set(datasets.total.key_component.dataframe.columns)
    time_cols = datasets.total.key_component.time_cols
    num_cols = len(columns)  # noqa
    num_features = datasets.total.num_features
    num_labels = datasets.total.num_labels

    # Create data loaders
    dataloaders = DictObj({})
    for i in datasets:
        dataloaders[i] = DataLoader(
            datasets[i], batch_size=cfg.action.train.single.batch_size
        )

    # Load the GeoJSON file and the mapping
    area_gdf = gpd.read_file(action.geo_data.geojson_url)
    area_type: str = action.geo_data.area_key
    area_gdf_areas = list(area_gdf[area_type.upper()].tolist())
    total_dataset_areas = list(
        datasets.total.original_data[area_type.capitalize()].unique()
    )
    similar_areas = set(area_gdf_areas) & set(total_dataset_areas)
    additional_area_mappings = OmegaConf.to_container(
        action.geo_data.additional_mapping
    )
    missing_area_keys = (
        set(total_dataset_areas)
        - set(area_gdf_areas)
        - set(additional_area_mappings.keys())
    )
    area_mapping = {i: i for i in similar_areas}
    area_mapping.update(additional_area_mappings)

    # Make the evaluation list for validation evaluations
    evaluation_list = build_evaluation_list(cfg.evaluation.components)

    #   endregion

    #
    #   ? 🧪 MODEL INFERENCING
    #   region
    #

    # Create a dictionary of dataframes
    test_dfs = {}
    valid_dfs = {}

    # Get the columns of the dataset and extract the label columns
    label_columns = list(columns[-num_labels:])

    # Iterate through the models list for evaluation
    if cfg.get("specific_run", False):
        iteration = [cfg.get("specific_run")]
    else:
        iteration = os.listdir(action.target_analysis_path)
    for run in tqdm(iteration[:INFERENCE_CONTROL], desc="Inferencing models"):
        # Create dataframes for all of the iterations
        test_dfs[run] = pd.DataFrame(
            columns=list(datasets.total.original_data.columns)
            + [i + " PREDICTION" for i in label_columns]
            + ["run"]
        )
        valid_dfs[run] = pd.DataFrame(
            columns=list(datasets.total.original_data.columns)
            + [i + " PREDICTION" for i in label_columns]
            + ["run"]
        )

        # Get their absolute path
        path = os.path.join(action.target_analysis_path, run)

        # Extract the best model checkpoint
        for file in os.listdir(path + "/checkpoints/"):
            if "best" in file:
                checkpoint = torch.load(path + "/checkpoints/" + file)

        # Initialize a skeleton model
        model = getattr(models, f"{cfg.model.model.upper()}Model")(
            input_size=datasets.total.sample_num_features,
            output_size=datasets.total.sample_num_labels,
            **cfg.model.args,
        )

        # Load the model and set to evaluation mode
        new_model_state_dict = OrderedDict()
        for k, v in checkpoint["model_state_dict"].items():
            name = k[7:] if k.startswith("module.") else k
            new_model_state_dict[name] = v
        model.load_state_dict(new_model_state_dict, strict=False)
        model.eval()

        # Run Inference
        with torch.no_grad():
            # ? Inference on the TEST dataset
            for batch in dataloaders.test:
                # Dissect, pass, and convert the data from the model output
                inputs, targets, non_state_inputs = batch
                outputs = model(inputs)
                outputs = datasets.total.inverse_transform(outputs)
                targets = datasets.total.inverse_transform(targets)
                non_state_inputs = datasets.total.inverse_transform(
                    non_state_inputs, False
                )

                # Iterate through the outputs
                for i in range(len(outputs)):
                    test_dfs[run].loc[len(test_dfs[run])] = np.append(
                        non_state_inputs[i],
                        list(targets[i]) + list(outputs[i]) + [run],
                    )

            # ? Inference on the VALID dataset
            for batch in dataloaders.valid:
                # Dissect, pass, and convert the data from the model output
                inputs, targets, non_state_inputs = batch
                outputs = model(inputs)
                outputs = datasets.total.inverse_transform(outputs)
                targets = datasets.total.inverse_transform(targets)
                non_state_inputs = datasets.total.inverse_transform(
                    non_state_inputs, False
                )

                # Iterate through the outputs
                for i in range(len(outputs)):
                    valid_dfs[run].loc[len(valid_dfs[run])] = np.append(
                        non_state_inputs[i],
                        list(targets[i]) + list(outputs[i]) + [run],
                    )

    #   endregion

    #
    #   ? 📑 MODEL EVALUATION
    #   region
    #

    # Combine the dataframes together
    performance_df = {
        "test": pd.concat(test_dfs.values(), ignore_index=True),
        "valid": pd.concat(valid_dfs.values(), ignore_index=True),
    }

    # > SAVE THE DFs
    os.makedirs(os.path.join(OUTPUT_PATH, "test"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, "valid"), exist_ok=True)
    performance_df["test"].to_csv(
        os.path.join(OUTPUT_PATH, "test", "results.csv")
    )
    performance_df["valid"].to_csv(
        os.path.join(OUTPUT_PATH, "valid", "results.csv")
    )

    # Create an enumeration of subsets for all non-chronal primary keys
    target_keys = list(key_cols - set(time_cols))
    groupings = [
        list(comb)
        for i in range(1, len(target_keys) + 1)
        for comb in itertools.combinations(target_keys, i)
    ]

    # Evaluate model performance by every split, group of columns, and value
    for split in performance_df:
        print(pyfiglet.figlet_format(split, font="rectangles"))
        for group in groupings:
            for key_values, selection in tqdm(
                performance_df[split].groupby(group),
                desc=f"{', '.join(group)} "
                "// Generating Scatter Plots Per Primary "
                "Key",
            ):
                # Create directories for each iteration
                if isinstance(key_values, tuple):
                    dir_path = f"{OUTPUT_PATH}/{split}/" + "/".join(
                        f"{col}/{val}" for col, val in zip(group, key_values)
                    )
                else:
                    # For a single column, create a simple structure
                    dir_path = f"{OUTPUT_PATH}/{split}/{group[0]}/{key_values}"
                os.makedirs(dir_path, exist_ok=True)

                # Apply data analysis
                execute_analysis(
                    selection,
                    performance_df[split],
                    dir_path,
                    num_labels,
                    num_features,
                    evaluation_list,
                    key_values,
                    area_type,
                    missing_area_keys,
                    area_gdf,
                    area_mapping,
                    group,
                )

        # Evaluate on the whole iterated dataframe
        execute_analysis(
            performance_df[split],
            performance_df[split],
            f"{OUTPUT_PATH}/{split}/",
            num_labels,
            num_features,
            evaluation_list,
            [],
            area_type,
            missing_area_keys,
            area_gdf,
            area_mapping,
            [],
        )

    # endregion


""" #! ⚠️ NEEDS WORK

    # > WHOLE COUNTRY PERFORMANCE OVER TIME
    # region

    # Create the timelines
    post_process.timeline_mae_graph(
        OUTPUT_PATH + "/valid/mae_timeline.png", valid_df
    )
    post_process.timeline_mae_graph(
        OUTPUT_PATH + "/test/mae_timeline.png", test_df
    )

    # endregion

    #   endregion

"""

# Run the main function
if __name__ == "__main__":
    main()
