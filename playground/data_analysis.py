#

# !
# ! TEMPORARY `pre_process_evaluation.py` SCRIPT
# !

# Set the python path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Util imports
from utils.scripts import dataset_reader, prepare_config, build_evaluation_list
from utils.dict_obj import DictObj
import matplotlib.pyplot as plt
import utils.evaluation.post_process as post_process

# Datastructure related imports
import geopandas as gpd
import pandas as pd

# Miscellaneous imports
from omegaconf import DictConfig, OmegaConf
import pyfiglet
import hydra

# ! UNORGANIZED IMPORTS
from dataset import (  # noqa
    PalayKeyComponent,
    AFFGDPComponent,
    ElNinoSSTComponent,
    AreaHarvestedComponent,
    PalayYieldComponent,
    PalayDataset,
)
import evaluation


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

    # Save the model's output path
    os.makedirs("./evaluation", exist_ok=True)
    OUTPUT_PATH = "./evaluation"

    # Extrapolate the datasets
    datasets = dataset_reader(cfg.dataset_path)
    # columns = datasets.total.original_data.columns
    # key_cols = set(datasets.total.key_component.dataframe.columns)
    # time_cols = datasets.total.key_component.time_cols
    # num_cols = len(columns)
    num_features = datasets.total.num_features
    num_labels = datasets.total.num_labels

    # Get the key, feature, and label column names of the dataset
    # extract = lambda x: set(sum([list(i.dataframe.columns) for i in x], []))
    # key_cols = set(datasets.total.key_component.dataframe.columns)
    # feature_cols = extract(datasets.total.feature_components)
    # label_cols = extract(datasets.total.label_components) - key_cols
    # time_cols = datasets.total.key_component.time_cols

    # Make the evaluation list for validation evaluations
    evaluation_list = build_evaluation_list(cfg.evaluation.components)

    # Copy the original dataset dataframe
    original_data = DictObj({})
    for i in datasets:
        original_data[i] = datasets[i].original_data.copy()

    # Set plot theme
    plt.style.use("seaborn-v0_8-deep")

    # Get area mappings
    area_gdf = gpd.read_file(
        cfg.action.post_process_analysis.geo_data.geojson_url
    )
    area_type: str = cfg.action.post_process_analysis.geo_data.area_key
    area_gdf_areas = list(area_gdf[area_type.upper()].tolist())
    total_dataset_areas = list(
        datasets.total.original_data[area_type.capitalize()].unique()
    )
    similar_areas = set(area_gdf_areas) & set(total_dataset_areas)
    additional_area_mappings = OmegaConf.to_container(
        cfg.action.post_process_analysis.geo_data.additional_mapping
    )
    missing_area_keys = (
        set(total_dataset_areas)
        - set(area_gdf_areas)
        - set(additional_area_mappings.keys())
    )
    area_mapping = {i: i for i in similar_areas}
    area_mapping.update(additional_area_mappings)

    # ! DEBUG
    print(pyfiglet.figlet_format("Differences", font="big"))
    print(len(area_gdf_areas), area_gdf_areas, "\n" * 3)
    print(len(total_dataset_areas), total_dataset_areas, "\n" * 3)
    print(set(total_dataset_areas) - set(area_gdf_areas), "\n" * 3)
    print(set(area_gdf_areas) - set(total_dataset_areas), "\n" * 3)
    print(datasets.total.feature_components)

    #
    #   ? MODEL ANALYSIS // RNN_4
    #   region
    #

    # Get the dataframe of the model performances
    model_perf = pd.read_csv("./results.csv")

    # Evaluate on the whole iterated dataframe
    execute_analysis(
        model_perf,
        model_perf,
        "./evaluation",
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

    #

    #   endregion

    #
    #   ? DISTRIBUTION ANALYSIS // PROVINCES
    #   region
    #

    # Grouping the data by province and counting the number of samples
    focus: pd.DataFrame = datasets.total.label_components[-1].dataframe
    focus.dropna(subset=["PALAY YIELD Q_n"])
    focus = focus[focus["PALAY YIELD Q_n"] != 0]
    province_counts = focus["Province"].value_counts()

    # Finding the minimum year for each province
    min_years = focus.groupby("Province")["Year"].min()

    # > Plotting
    plt.figure(figsize=(10, 14))
    ax = province_counts.plot(kind="barh")
    plt.xlabel("Number of Samples", fontsize=16)
    plt.ylabel("Province", fontsize=16)
    for i in ax.patches:
        province = i.get_y() + i.get_height() / 2
        samples = int(i.get_width())
        lowest_year = min_years.loc[
            ax.get_yticklabels()[int(province)].get_text()
        ]

        ax.text(
            i.get_width() + 0.1,
            i.get_y() + i.get_height() / 2,
            f"{samples}, {lowest_year}",
            fontsize=10,
            color="black",
            verticalalignment="center",
        )
    plt.tight_layout()
    plt.xlim(0, 350)

    # Save the plot
    plt.savefig(os.path.join(OUTPUT_PATH, "province_sample_distribution.png"))

    # > Map Plotting
    province_summary = pd.DataFrame(
        {"Sample Count": province_counts}
    ).reset_index()
    post_process.generate_map(
        province_summary,
        area_gdf,
        area_mapping,
        os.path.join(OUTPUT_PATH, "province_sample_distribution_map.png"),
        False,
        "Province",
    )

    #   endregion

    #
    #   ? OVERTIME ANALYSIS // AREA-HARVESTED
    #   region
    #

    # Aggregating data over all provinces for each croptype
    df_grouped = (
        original_data.total.groupby(["Year", "Quarter", "Crop"])[
            "Area Harvested Q_n"
        ]
        .mean()
        .reset_index()
    )

    # Creating a new 'Year-Quarter' column for detailed labels and 'Year' for
    # x-axis
    df_grouped["Year_Quarter"] = (
        df_grouped["Year"].astype(str)
        + "-Q"
        + df_grouped["Quarter"].astype(str)
    )
    df_grouped["Year_Label"] = df_grouped["Year"].astype(str)

    # Aggregate the data to ensure unique Year-Quarter-Crop combinations
    df_grouped_year_qtr = (
        df_grouped.groupby(["Year_Quarter", "Year_Label", "Crop"])[
            "Area Harvested Q_n"
        ]
        .mean()
        .reset_index()
    )

    # Pivoting the table using the aggregated data
    df_pivot_year_qtr = df_grouped_year_qtr.pivot(
        index="Year_Quarter", columns="Crop", values="Area Harvested Q_n"
    )

    # ! Plotting
    plt.figure(figsize=(14, 8))
    for croptype in df_pivot_year_qtr.columns:
        plt.plot(
            df_pivot_year_qtr.index,
            df_pivot_year_qtr[croptype],
            marker="o",
            label=croptype,
        )
    plt.title("Rice Harvested Over Time by Crop Type")
    plt.xlabel("Year")
    plt.ylabel("Area Harvested (Ha)")
    plt.legend(title="Crop Type")
    plt.grid(True)
    year_labels = df_grouped_year_qtr["Year_Label"].unique()
    plt.xticks(
        ticks=range(0, len(df_pivot_year_qtr), 4),
        labels=year_labels,
        rotation=45,
    )
    plt.ticklabel_format(style="plain", axis="y")
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(OUTPUT_PATH, "area_harvested_timeline.png"))

    #   endregion

    #
    #   ? OVERTIME ANALYSIS // PALAY OUTPUT
    #   region
    #

    # Aggregating data over all provinces for each croptype
    df_grouped = (
        original_data.total.groupby(["Year", "Quarter", "Crop"])[
            "PALAY YIELD Q_n"
        ]
        .mean()
        .reset_index()
    )

    # Creating a new 'Year-Quarter' column for detailed labels and 'Year' for
    # x-axis
    df_grouped["Year_Quarter"] = (
        df_grouped["Year"].astype(str)
        + "-Q"
        + df_grouped["Quarter"].astype(str)
    )
    df_grouped["Year_Label"] = df_grouped["Year"].astype(str)

    # Aggregate the data to ensure unique Year-Quarter-Crop combinations
    df_grouped_year_qtr = (
        df_grouped.groupby(["Year_Quarter", "Year_Label", "Crop"])[
            "PALAY YIELD Q_n"
        ]
        .mean()
        .reset_index()
    )

    # Pivoting the table using the aggregated data
    df_pivot_year_qtr = df_grouped_year_qtr.pivot(
        index="Year_Quarter", columns="Crop", values="PALAY YIELD Q_n"
    )

    # ! Plotting
    plt.figure(figsize=(14, 8))
    for croptype in df_pivot_year_qtr.columns:
        plt.plot(
            df_pivot_year_qtr.index,
            df_pivot_year_qtr[croptype],
            marker="o",
            label=croptype,
        )
    plt.title("Rice Output Over Time by Crop Type")
    plt.xlabel("Year")
    plt.ylabel("Rice Output Volume (MT)")
    plt.legend(title="Crop Type")
    plt.grid(True)
    year_labels = df_grouped_year_qtr["Year_Label"].unique()
    plt.xticks(
        ticks=range(0, len(df_pivot_year_qtr), 4),
        labels=year_labels,
        rotation=45,
    )
    plt.ticklabel_format(style="plain", axis="y")
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(OUTPUT_PATH, "rice_timeline.png"))

    #   endregion

    #
    #   ? OVERTIME ANALYSIS // YIELD OUTPUT
    #   region
    #

    # Aggregating data over all provinces for each croptype
    df_grouped = (
        original_data.total.groupby(["Year", "Quarter", "Crop"])[
            ["Area Harvested Q_n", "PALAY YIELD Q_n"]
        ]
        .sum()
        .reset_index()
    )

    # Calculate the yield
    df_grouped["Yield"] = (
        df_grouped["PALAY YIELD Q_n"] / df_grouped["Area Harvested Q_n"]
    )

    # Creating a new 'Year-Quarter' column for detailed labels and 'Year' for
    # x-axis
    df_grouped["Year_Quarter"] = (
        df_grouped["Year"].astype(str)
        + "-Q"
        + df_grouped["Quarter"].astype(str)
    )
    df_grouped["Year_Label"] = df_grouped["Year"].astype(str)

    # Aggregate the data to ensure unique Year-Quarter-Crop combinations
    df_grouped_year_qtr = (
        df_grouped.groupby(["Year_Quarter", "Year_Label", "Crop"])["Yield"]
        .mean()
        .reset_index()
    )

    # Pivoting the table using the aggregated data
    df_pivot_year_qtr = df_grouped_year_qtr.pivot(
        index="Year_Quarter", columns="Crop", values="Yield"
    )

    # Plotting
    plt.figure(figsize=(12, 6))
    for croptype in df_pivot_year_qtr.columns:
        plt.plot(
            df_pivot_year_qtr.index,
            df_pivot_year_qtr[croptype],
            marker="o",
            label=croptype,
        )

    plt.title("Yield Over Time by Crop Type")
    plt.xlabel("Year")
    plt.ylabel("Rice Yield (MT/Ha)")
    plt.legend(title="Crop Type")
    plt.grid(True)

    # Set the x-axis ticks to show only the years
    year_labels = df_grouped_year_qtr["Year_Label"].unique()
    plt.xticks(
        ticks=range(0, len(df_pivot_year_qtr), 4),
        labels=year_labels,
        rotation=45,
    )

    # Format the y-axis to avoid scientific notation
    plt.ticklabel_format(style="plain", axis="y")

    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(OUTPUT_PATH, "yield_timeline.png"))

    #   endregion

    #
    #   ? OVERTIME ANALYSIS // AFF GDP OUTPUT
    #   region
    #

    # ! GET CUSTOM GDP COMPONENT
    aff_gdp_component = AFFGDPComponent(save_path=True)

    # Aggregating data over all provinces, quarters, and crops
    df_grouped = (
        aff_gdp_component.dataframe.groupby(["Year", "Quarter"])["AFF GDP Q_n"]
        .mean()
        .reset_index()
    )

    # Creating a new 'Year-Quarter' column for detailed labels and 'Year' for
    # the x-axis
    df_grouped["Year_Quarter"] = (
        df_grouped["Year"].astype(str)
        + "-Q"
        + df_grouped["Quarter"].astype(str)
    )
    df_grouped["Year_Label"] = df_grouped["Year"].astype(str)

    # Aggregate the data to ensure unique Year-Quarter combinations
    df_grouped_year_qtr = (
        df_grouped.groupby(["Year_Quarter", "Year_Label"])["AFF GDP Q_n"]
        .mean()
        .reset_index()
    )

    # Plotting
    plt.figure(figsize=(14, 8))
    plt.plot(
        df_grouped_year_qtr["Year_Quarter"],
        df_grouped_year_qtr["AFF GDP Q_n"],
        marker="o",
        label="AFF GDP Q_n",
    )

    plt.title("AFF GDP Over Time")
    plt.xlabel("Year")
    plt.ylabel("AFF GDP (Million Philippine Pesos)")
    plt.grid(True)

    year_labels = df_grouped_year_qtr["Year_Label"].unique()
    plt.xticks(
        ticks=range(0, len(df_grouped_year_qtr), 4),
        labels=year_labels,
        rotation=45,
    )
    plt.ticklabel_format(style="plain", axis="y")
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(OUTPUT_PATH, "aff_gdp_timeline.png"))

    #   endregion

    #
    #   ? OVERTIME ANALYSIS // SST OUTPUT NINO3.4 SST Q_n-1
    #   region
    #

    # ! GET CUSTOM SST COMPONENT
    el_nino_sst_component = ElNinoSSTComponent(save_path=True)

    # Aggregating data over all provinces, quarters, and crops
    df_grouped = (
        el_nino_sst_component.dataframe.groupby(["Year", "Quarter"])[
            "NINO3.4 SST Q_n"
        ]
        .mean()
        .reset_index()
    )

    # Creating a new 'Year-Quarter' column for detailed labels and 'Year' for
    # the x-axis
    df_grouped["Year_Quarter"] = (
        df_grouped["Year"].astype(str)
        + "-Q"
        + df_grouped["Quarter"].astype(str)
    )
    df_grouped["Year_Label"] = df_grouped["Year"].astype(str)

    # Aggregate the data to ensure unique Year-Quarter combinations
    df_grouped_year_qtr = (
        df_grouped.groupby(["Year_Quarter", "Year_Label"])["NINO3.4 SST Q_n"]
        .mean()
        .reset_index()
    )

    # Plotting
    plt.figure(figsize=(14, 8))
    plt.plot(
        df_grouped_year_qtr["Year_Quarter"],
        df_grouped_year_qtr["NINO3.4 SST Q_n"],
        marker="o",
        label="NINO3.4 SST Q_n",
    )

    plt.title("El Nino 3.4 SST Over Time")
    plt.xlabel("Year")
    plt.ylabel("El Nino 3.4 SST (Degrees Centigrade)")
    plt.grid(True)

    year_labels = df_grouped_year_qtr["Year_Label"].unique()
    plt.xticks(
        ticks=range(0, len(df_grouped_year_qtr), 4),
        labels=year_labels,
        rotation=45,
    )
    plt.ticklabel_format(style="plain", axis="y")
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(OUTPUT_PATH, "el_nino_sst_timeline.png"))

    #   endregion


# Run the main function
if __name__ == "__main__":
    main()
