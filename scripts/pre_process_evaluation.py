# Set the python path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Util imports
from utils.scripts import create_output_paths, dataset_reader, prepare_config
from utils.evaluation import pre_process
from utils.dict_obj import DictObj

# Miscellaneous imports
from omegaconf import DictConfig
from tqdm import tqdm
import hydra


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

    # Simplify reference to the action configuration
    action = cfg.action.pre_process_analysis

    # Save the model's output path
    OUTPUT_PATH = create_output_paths(
        os.path.join(cfg.output_path, "pre_process_evaluation")
    )

    # Extrapolate the datasets
    datasets = dataset_reader(cfg.dataset_path)

    # Get the key, feature, and label column names of the dataset
    extract = lambda x: set(sum([list(i.dataframe.columns) for i in x], []))
    key_cols = set(datasets.total.key_component.dataframe.columns)
    feature_cols = extract(datasets.total.feature_components)
    label_cols = extract(datasets.total.label_components) - key_cols
    time_cols = datasets.total.key_component.time_cols

    # Copy the original dataset dataframe
    original_data = DictObj({})
    for i in datasets:
        original_data[i] = datasets[i].original_data.copy()

    #
    #   ? 📈 Overtime Analysis
    #   region
    #

    # Output control
    if action.analysis.overtime:
        # Iterate over the dataset splits
        for split in original_data:
            # Iterate over the combination of key columns
            for remaining_key_values, primary_group in tqdm(
                original_data[split].groupby(list(key_cols - set(time_cols))),
                desc=f"{split.upper()} // Creating time series plot "
                "information",
            ):
                # Convert remaining_key_values to a list
                remaining_key_str = [str(val) for val in remaining_key_values]

                # Iterate over each feature column
                for feature in (feature_cols - key_cols).union(label_cols):
                    # Create the directory for the resulting graph results
                    output_dir = (
                        f"{OUTPUT_PATH}/time_series/{split}/"
                        f"{'/'.join(remaining_key_str)}/"
                    )
                    os.makedirs(output_dir, exist_ok=True)

                    # Define output file name and path
                    output_file = os.path.join(output_dir, f"{feature}.png")

                    # Generate plot for the entire time range
                    pre_process.generate_time_series_plot(
                        dataframe=primary_group,
                        x_column_sort_order=time_cols,
                        y_columns=[feature],
                        output_file=output_file,
                    )

    #   endregion

    #
    #   ? 📊 Count Distribution Analysis
    #   region
    #

    # Output control for distribution of number of samples
    if action.analysis.distribution:
        # Iterate over the splits in the dataset
        for split in original_data:
            # Ensure the output directory exists
            os.makedirs(
                os.path.join(OUTPUT_PATH, "distribution", split),
                exist_ok=True,
            )

            # Iterate over each key column to generate plots
            for feature in tqdm(
                key_cols,
                desc=f"{split.upper()} // Creating count "
                "distribution plots",
            ):
                # Define the output file name and path
                output_file = os.path.join(
                    OUTPUT_PATH,
                    "distribution",
                    split,
                    f"{feature} Key.png",
                )

                # Generate the distribution for the specific feature column
                pre_process.generate_discrete_distribution_plot(
                    dataframe=original_data[split],
                    column=feature,
                    output_file=output_file,
                    output_size=(9, 15),
                    max_label_length=500,
                    swap_axis=True,
                )

            # Create a new column for combined key values
            combined_column = "Combined Non-Chronal Keys"
            original_data[split][combined_column] = original_data[split][
                list(key_cols - set(time_cols))
            ].apply(lambda row: "_".join(row.values.astype(str)), axis=1)
            output_file = os.path.join(
                OUTPUT_PATH,
                "distribution",
                split,
                f"{combined_column}.png",
            )
            pre_process.generate_discrete_distribution_plot(
                dataframe=original_data[split],
                column=combined_column,
                output_file=output_file,
                output_size=(9, 15),
                max_label_length=500,
                swap_axis=True,
            )

    #   endregion

    #
    #   ? 📊 Value Distribution Analysis
    #   ! ❌ DEFUNCT
    #   region
    #

    # # Output control for value distribution plots
    # if action.analysis.value_distribution:
    #     # Iterate over the splits in the dataset
    #     for split in original_data:
    #         # Create the folder
    #         os.makedirs(
    #             os.path.join(OUTPUT_PATH, "value_distribution", split),
    #             exist_ok=True,
    #         )

    #         # Group by the non-time-related key columns
    #         for remaining_key_values, primary_group in tqdm(
    #             original_data[split].groupby(list(key_cols - set(time_cols))), # noqa
    #             desc=f"{split.upper()} // Creating value distribution "
    #             "plot information (Combined Keys)",
    #         ):
    #             # Convert remaining_key_values to a list of strs for labeling
    #             remaining_key_str = [str(val) for val in remaining_key_values] # noqa

    #             # Iterate over each feature column to generate plots
    #             for feature in feature_cols:
    #                 # Create the directory path for storing the graph results
    #                 output_dir = (
    #                     f"{OUTPUT_PATH}/value_distribution/{split}/"
    #                     f"{'/'.join(remaining_key_str)}/"
    #                 )
    #                 os.makedirs(output_dir, exist_ok=True)

    #                 # Define the output file name and path
    #                 output_file = os.path.join(output_dir, f"{feature}.png")

    #                 # Generate the value distribution plot
    #                 pre_process.generate_continuous_distribution_plot(
    #                     dataframe=primary_group,
    #                     column=feature,
    #                     output_file=output_file,
    #                 )

    #         # Iterate over each key column to group data and generate plots
    #         for key in tqdm(
    #             key_cols,
    #             desc=f"{split.upper()} // Creating value distribution "
    #             "plot information (Individual Keys)",
    #         ):
    #             # Group by the current key column
    #             for key_value, group in original_data[split].groupby(key):
    #                 # Convert the key value to a string for directory naming
    #                 key_value_str = str(key_value)

    #                 # Create a directory for the current key and key value
    #                 output_dir = os.path.join(
    #                     OUTPUT_PATH,
    #                     "value_distribution",
    #                     split,
    #                     key,
    #                     key_value_str,
    #                 )
    #                 os.makedirs(output_dir, exist_ok=True)

    #                 # Iterate over each feature column to generate continuous
    #                 # distribution plots
    #                 for feature in feature_cols:
    #                     # Define the output file name and path for each col
    #                     output_file = os.path.join(
    #                         output_dir, f"{feature}.png"
    #                     )

    #                     # Generate the continuous distribution plot for the
    #                     # current group
    #                     pre_process.generate_continuous_distribution_plot(
    #                         dataframe=group,
    #                         column=feature,
    #                         output_file=output_file,
    #                     )

    #   endregion


# Run the main function
if __name__ == "__main__":
    main()
