# Imports
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import matplotlib.pyplot as plt
from typing import Optional
import seaborn as sns
import pandas as pd


def generate_time_series_plot(
    dataframe: pd.DataFrame,
    x_column_sort_order: list[str],
    y_columns: list[str],
    output_file: str,
    line_column: Optional[str] = None,
    category_indexes: list[int] = None,
    output_size: tuple[float, float] | list[float, float] = (20, 9),
):
    """Function to save a plot of a time series plot

    Args:
        dataframe (pd.DataFrame): Dataframe to extrapolate the plot information
        x_column_sort_order (list[str]): X Columns to combine the several
            columns. Lower indexed elements are prioritized first with
            proceeding ones ben done in succession.
        y_columns (list[str]): The column to be the y-axis.
        output_file (str): The save path of the developed column
        line_column (Optional[str], optional): The column that has categories
            to compile lines from. Defaults to None.
        category_indexes (list[int], optional): Specific categories from the
            `line_column` to plot. Defaults to None.
        output_size (tuple[float, float] | list[float, float]):
            The size of the figure. Defaults to (20, 9).
    """
    # Create a combined x-axis column based on the ranking of x_cols
    dataframe["combined_x"] = (
        dataframe[x_column_sort_order].astype(str).agg("-".join, axis=1)
    )

    # Set the figure size
    fig, ax = plt.subplots(figsize=output_size)

    # If the line column is specified, plot separate lines
    if line_column:
        unique_categories = dataframe[line_column].unique()
        if category_indexes is not None:
            selected_categories = [
                unique_categories[i]
                for i in category_indexes
                if i < len(unique_categories)
            ]
        else:
            selected_categories = unique_categories
        for label, grp in dataframe.groupby(line_column):
            if label in selected_categories:
                for y_col in y_columns:
                    ax.plot(
                        grp["combined_x"],
                        grp[y_col],
                        label=f"{label} ({y_col})",
                    )

    # If not, just plot straight away
    else:
        for y_col in y_columns:
            ax.plot(dataframe["combined_x"], dataframe[y_col], label=y_col)

    # Develop the plot
    if line_column:
        plt.legend(title=line_column)
    else:
        plt.legend()
    plt.xlabel("-".join(x_column_sort_order))
    plt.xticks(rotation=75)
    plt.ylabel(", ".join(y_columns))
    plt.title(f"{' vs '.join(y_columns)} vs {'-'.join(x_column_sort_order)}")
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style="plain", axis="y")
    plt.grid(True)
    plt.tight_layout()

    # Adjust x-axis limits
    x_min, x_max = dataframe["combined_x"].min(), dataframe["combined_x"].max()
    ax.set_xlim(x_min, x_max)

    # Adjust y-axis limits
    y_min, y_max = min(dataframe[y_columns].min()), max(
        dataframe[y_columns].max()
    )
    ax.set_ylim(y_min, y_max)

    # Save the plot
    plt.savefig(output_file)
    plt.close()


def generate_discrete_distribution_plot(
    dataframe: pd.DataFrame,
    column: str,
    output_file: str,
    output_size: tuple[float, float] | list[float, float] = (20, 9),
    max_label_length: int = 12,
    max_categories: int = 500,
    swap_axis: bool = False,
):
    """Function to save a plot of a distribution graph of the number of
    occurrences of categories in the x-axis.

    Args:
        dataframe (pd.DataFrame): Dataframe to extrapolate the plot information
        column (str): Column to be used for the x-axis categories
        output_file (str): The save path of the developed column
        output_size (tuple[float, float] | list[float, float]):
            The size of the figure. Defaults to (20, 9)
        max_label_length (int): Maximum length of the label
        max_categories (int): Maximum number of categories to display. Defaults
            to 500.
        swap_axis (bool): If True, swaps the x and y axis in the plot. Defaults
            to False.
    """
    # Count occurrences of each category in the specified column
    counts = dataframe[column].value_counts().sort_index()

    # If too many categories, display only the top `max_categories`
    if len(counts) > max_categories:
        counts = counts.head(max_categories)

    # Truncate long labels
    truncated_labels = [
        (
            (str(label)[:max_label_length] + "...")
            if len(str(label)) > max_label_length
            else str(label)
        )
        for label in counts.index
    ]

    # Set the figure size
    fig, ax = plt.subplots(figsize=output_size)

    if swap_axis:
        # Plot the distribution with swapped axes
        ax.barh(truncated_labels, counts.values)
        plt.ylabel(column)
        plt.xlabel("Occurrences")
        if any([len(str(label)) > 3 for label in counts.index]):
            plt.yticks(rotation=0)
        plt.title(f"Distribution of {column} (Swapped Axes)")
    else:
        # Plot the distribution with standard axes
        ax.bar(truncated_labels, counts.values)
        plt.xlabel(column)
        plt.ylabel("Occurrences")
        if any([len(str(label)) > 3 for label in counts.index]):
            plt.xticks(rotation=90, ha="right")
        plt.title(f"Distribution of {column}")

    plt.grid(True, axis="y" if not swap_axis else "x")
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file)
    plt.close()


def generate_continuous_distribution_plot(
    dataframe: pd.DataFrame,
    column: str,
    output_file: str,
    output_size: tuple[float, float] | list[float, float] = (20, 9),
):
    """Function to save a plot of a distribution graph of continous values in
    a given dataframe and a given column.

    Args:
        dataframe (pd.DataFrame): Dataframe to extrapolate the plot information
        column (str): Column to be used for the x-axis categories
        output_file (str): The save path of the developed column
        output_size (tuple[float, float] | list[float, float]):
            The size of the figure. Defaults to (20, 9)
    """
    # Extract the data
    data = dataframe[column]

    # Plot the distribution
    plt.figure(figsize=output_size)
    ax = sns.histplot(
        data, kde=True, color="blue", element="step", stat="density"
    )
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Percentage")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))

    # Save the plot
    plt.savefig(output_file)
    plt.close()
