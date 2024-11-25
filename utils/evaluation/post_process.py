# Imports
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from matplotlib.ticker import FuncFormatter
from adjustText import adjust_text
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
import geopandas
import os


def ground_truth_to_pred_graph(
    result_path: str,
    labels: list | np.ndarray,
    predictions: list | np.ndarray,
    evaluations: dict[str, float],
):
    """Function to plot ground truth data against predicted information

    Args:
        result_path (str): Output path of the graph
        labels (list | np.ndarray): Enumerable of label information
        predictions (list | np.ndarray): Enumerable of predicted values
        evaluations (dict[str, float]): A dictionary of various evaluation
            metrics to list with the scatter plot
    """

    # Ensure labels and predictions are numpy arrays
    labels = np.array(labels).astype(np.float64)
    predictions = np.array(predictions).astype(np.float64)

    # Plot and create the graph
    plt.figure(figsize=(10, 6))
    plt.cla()
    plt.clf()
    plt.scatter(
        labels,
        predictions,
        alpha=0.5,
        color="blue",
        label="Predicted vs Actual",
    )
    plt.plot(
        [min(labels), max(labels)],
        [min(labels), max(labels)],
        "k--",
        lw=3,
        label="Ideal Fit",
    )

    # Add statistics of the performance
    plt.text(
        0.95,
        0.05,
        "\n".join(
            [
                f"{key.upper()}: {value:.3f}"
                for key, value in evaluations.items()
            ]
        ),
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", alpha=0.1),
    )

    # Add all other items
    plt.title(f"Ground Truth vs. Predicted // {os.path.basename(result_path)}")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predicted")
    plt.legend()
    plt.grid(True)
    formatter = FuncFormatter(lambda x, pos: f"{x:,.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.tight_layout()

    # Save the graph
    plt.savefig(result_path, format="png", dpi=300)
    plt.close()


def error_distribution_graph(
    result_path: str,
    labels: list | np.ndarray,
    predictions: list | np.ndarray,
    evaluations: dict[str, float],
):
    """Function to plot the distribution of MAE and RMSE

    Args:
        result_path (str): Output path of the graph
        labels (list | np.ndarray): Enumerable of label information
        predictions (list | np.ndarray): Enumerable of predicted values
        evaluations (dict[str, float]): A dictionary of various evaluation
            metrics to list with the scatter plot
    """
    # Ensure labels and predictions are numpy arrays
    labels = np.array(labels).astype(np.float64)
    predictions = np.array(predictions).astype(np.float64)

    # Calculate errors
    errors = predictions - labels
    mae = mean_absolute_error(labels, predictions)
    sem = np.std(np.abs(errors)) / np.sqrt(len(errors))
    mae_ci = stats.t.ppf((1 + 0.95) / 2, df=len(errors) - 1) * sem
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    error_df = pd.DataFrame(errors, columns=["Errors"])

    # Plotting the distribution of errors
    plt.figure(figsize=(15, 9))
    _ = sns.histplot(
        error_df["Errors"],
        kde=True,
        color="blue",
        element="step",
        stat="density",
    )

    # Add statistics of the performance
    plt.text(
        0.95,
        0.05,
        "\n".join(
            [
                f"{key.upper()}: {value:.3f}"
                for key, value in evaluations.items()
            ]
        ),
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", alpha=0.1),
    )

    # Add MAE and RMSE lines
    plt.axvline(
        mae,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"MAE: {mae:.2f} ± {mae_ci:.2f} @ 95% Confidence",
    )
    plt.axvline(
        rmse,
        color="green",
        linestyle="dashed",
        linewidth=2,
        label=f"RMSE: {rmse:.2f}",
    )

    # Include labels to the graph
    plt.title(
        f"Distribution of Prediction Errors // {os.path.basename(result_path)}"
    )
    plt.xlabel("Prediction Error")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the graph
    plt.savefig(result_path, format="png", dpi=300)
    plt.close()


# ! FIGURE OUT
def timeline_mae_graph(result_path: str, df: pd.DataFrame):
    """Function to plot the timeline of MAE by year

    Args:
        result_path (str): Base absolute/relative path to the result directory
        df (pd.DataFrame): DataFrame containing 'year', 'ground_truth', and
            'prediction' columns
    """

    # Get the MAE per year
    mae_per_year = (
        df.groupby("Year")
        .apply(
            lambda x: mean_absolute_error(
                x["PALAY YIELD"], x["PALAY YIELD PREDICTION"]
            )
        )
        .to_dict()
    )

    # Plot the MAEs
    years = sorted(mae_per_year.keys())
    maes = [mae_per_year[year] for year in years]
    plt.figure(figsize=(10, 5))
    plt.plot(years, maes, marker="o")
    plt.xlabel("Year")
    plt.ylabel("MAE")
    plt.title("MAE Overtime")
    plt.grid(True)
    plt.xticks(years, rotation=45)
    plt.savefig(result_path)
    plt.close()


def generate_map(
    result_df: pd.DataFrame,
    regions_gdf: geopandas.GeoDataFrame,
    region_mapping: dict,
    output_path: str,
    reversed_color: bool,
    area_type: str,
):
    """Function to generate the visualized map results of the palay forecaster

    Args:
        result_df (pd.DataFrame): Regionalized performance results
        regions_gdf (geopandas.GeoDataFrame): Regionalized information
        region_mapping (dict): Mapping of the names from the result of the
            model to the geojson content
        output_path (str): Path of the output of the map image
        column (str): Column to visualize
        reversed_color (bool): Should the color be in reverse?
        area_type (str): Column name that pertains to areas in the result df
    """

    # Process the contents of the dataframes
    result_df[area_type] = result_df[area_type].map(region_mapping).str.strip()
    regions_gdf[area_type.upper()] = regions_gdf[area_type.upper()].str.strip()

    # Merge the GeoDataFrame with the MAE data
    merged_gdf = regions_gdf.merge(
        result_df, left_on=area_type.upper(), right_on=area_type, how="left"
    )

    # Iterate through all columns of the result dataframe except for the area
    for metric in [col for col in result_df.columns if col != area_type]:
        # Determine the min and max values for the metric
        vmin = result_df[metric].min()
        vmax = result_df[metric].max()

        # Plot the GeoDataFrame with a heatmap based on MAE values
        fig, ax = plt.subplots(1, 1, figsize=(12, 15))
        merged_gdf.plot(
            column=metric,
            cmap="RdYlGn" + ("_r" if reversed_color else ""),
            # linewidth=0.8,
            ax=ax,
            edgecolor="0.8",
            legend=False,
            vmin=vmin,
            vmax=vmax,
        )

        # Create a colorbar with integrated min, max, and range indication
        sm = mpl.cm.ScalarMappable(
            cmap="RdYlGn" + ("_r" if reversed_color else ""),
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
        )
        sm._A = []  # Required for ScalarMappable
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(f"{metric.upper()} Value")

        # Add min and max labels to the colorbar
        cbar.ax.text(
            1.59,
            0,
            f"Min: {vmin}",
            ha="center",
            va="bottom",
            transform=cbar.ax.transAxes,
        )
        cbar.ax.text(
            1.65,
            1,
            f"Max: {vmax}",
            ha="center",
            va="top",
            transform=cbar.ax.transAxes,
        )

        # Add labels for each area and store them for adjustment
        texts = []
        for idx, row in merged_gdf.iterrows():
            texts.append(
                plt.text(
                    x=row.geometry.centroid.x,
                    y=row.geometry.centroid.y,
                    s=row[area_type],
                    fontsize=8,
                    color="black",
                    ha="center",
                )
            )

        # Adjust text to avoid overlap
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="->", color="black"),
            force_text=(0.5, 0.5),
        )

        # Set title and labels
        ax.set_title(f"Heatmap of {metric.upper()} by {area_type}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Save the plot
        plt.savefig(
            output_path.format(metric=metric.upper()), bbox_inches="tight"
        )
