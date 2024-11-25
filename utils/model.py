# Imports
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float | torch.Tensor,
    filename: str,
):
    """Helper function to save the checkpoint

    Args:
        model (torch.nn.Module): Model to save
        optimizer (torch.optim.Optimizer): Optimizer state to save
        epoch (int): Epoch to list
        loss (float | torch.Tensor): Loss to list
        filename (str): Target directory to save the checkpoint
    """

    # Compile the checkpoint
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    # Save the checkpoint
    torch.save(checkpoint, filename)
