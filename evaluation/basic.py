# Evaluation related imports
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from evaluation._base import BaseEvaluation

# PyTorch related imports
import torch

# Util related imports
from utils.metrics import calculate_adjusted_r2

# Datastructure related imports
import json  # noqa
import numpy as np


class BasicRegressionEvaluations(BaseEvaluation):
    """Basic regression evaluation class that hosts the following metrics:

        1. Mean Absolute Error (MAE)
        1. Mean Absolute Percentage Error (MAPE)
        2. Root Mean Squared Error (RMSE)
        3. R^2
        4. R^2 Adjusted

    These are the base metrics for all regression tasks
    """

    def __init__(self: "BasicRegressionEvaluations"):
        """Constructor to initialize a basic regression evaluation instance

        Args:
            self (BasicRegressionEvaluations): Instance method for the class
        """
        # Inherit the attributes and methods of the parent class
        super(BasicRegressionEvaluations).__init__()

    def __call__(
        self: "BasicRegressionEvaluations",
        predictions: list,
        labels: list,
        inputs: torch.Tensor,
    ) -> dict[str, float]:
        """Updater method to update the MeanAveragePrecision instance

        Args:
            self (BasicRegressionEvaluations): Instance method for the class
            predictions (list): List of model outputs from the given data
            labels (list): List of ground truths from the model information
            inputs (torch.Tensor): Tensor of inputs that contribute to the
                output of the model (`predictions`)

        Returns:
            dict[str, float]: The calculated MAE, RMSE, R^2, and R^2 Adjusted
        """
        # If the predictions are empty, pass a NaN value for everything
        if not len(predictions):
            result = {
                "mae": np.nan,
                "mape": np.nan,
                "rmse": np.nan,
                "r2": np.nan,
                "r2_adjusted": np.nan,
            }

        # If not, conduct calculations as normal
        else:
            result = {
                "mae": float(mean_absolute_error(labels, predictions)),
                "mape": float(
                    mean_absolute_percentage_error(labels, predictions)
                ),
                "rmse": float(
                    np.sqrt(mean_squared_error(labels, predictions))
                ),
                "r2": float(r2_score(labels, predictions)),
            }
            result["r2_adjusted"] = float(
                calculate_adjusted_r2(
                    result["r2"], len(labels), inputs.shape[1]
                )
            )

        # Return the result
        return result
