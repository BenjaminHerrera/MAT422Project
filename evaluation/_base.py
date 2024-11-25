# PyTorch related imports
import torch


class BaseEvaluation:
    """Base class for evaluation classes

    All child classes MUST have a __call__(input, target) method that intakes
        the input and the target. Afterwards, it compiles the information at
        the end.
    """

    def __init__(self: "BaseEvaluation"):
        """Constructor for the base evaluation class

        Args:
            self (BaseEvaluation): Instance method for the class
        """
        pass

    def __call__(
        self: "BaseEvaluation",
        predictions: list,
        labels: list,
        inputs: torch.Tensor,
    ) -> dict[str, float]:
        """Call method to override in order to be a workable evaluation method

        Args:
            self (BaseEvaluation): Instance method for the class
            predictions (list): List of model outputs from the given data
            labels (list): List of ground truths from the model information
            inputs (torch.Tensor): Tensor of inputs that contribute to the
                output of the model (`predictions`)

        Raises:
            NotImplementedError: Raised if the child method does not
                implement the method

        Returns:
            dict[str, float]: A dictionary of the model's evaluation
        """
        raise NotImplementedError
