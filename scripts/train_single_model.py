# Set the python path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Model related imports
import models

# Dataset related imports
from torch.utils.data import DataLoader

# PyTorch related imports
import torch

# Util related imports
from utils.scripts import (
    create_output_paths,
    broadcast_string,
    dataset_reader,
    build_evaluation_list,
    prepare_config,
)
from utils.model import save_checkpoint
from utils.dict_obj import DictObj

# Execution related imports
from accelerate import Accelerator
from tqdm import tqdm
import operator
import random
import wandb
import os

# Datastructure related imports
from dotenv import dotenv_values, find_dotenv
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing  # noqa
import pandas as pd  # noqa
import numpy as np
import json  # noqa

# Miscellaneous imports
from colorama import Fore, Style
import hydra
import re

# Constant definitions
OPS = {"min": operator.lt, "max": operator.gt}


@hydra.main(version_base=None, config_path="../configs/", config_name="fcnn_6")
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

    # Get the dotenv information
    dot_env = DictObj(dotenv_values(find_dotenv()))

    # Set random seed
    random.seed(random.randint(0, 2**32 - 1))
    np.random.seed(random.randint(0, 2**32 - 1))
    torch.manual_seed(random.randint(0, 2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random.randint(0, 2**32 - 1))

    # Check if the model type is legit. Based from the filenames in ./models/
    filenames = []
    model_path = os.chdir(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/")
    )
    for filename in os.listdir(model_path):
        filename_without_extension, _ = os.path.splitext(filename)
        filenames.append(filename_without_extension)
    if cfg.model.model not in filenames:
        raise ValueError(f"Invalid model type. Can only be {filenames}")

    # Save and create the model train session's output path
    OUTPUT_PATH = ""
    RUN_NAME = ""
    CHECKPOINT_PATH = ""
    model_name = cfg.model.model
    base_path = os.path.join(cfg.output_path, cfg.get("save_folder", ""))
    os.makedirs(base_path, exist_ok=True)
    if accelerator.is_main_process:
        OUTPUT_PATH = create_output_paths(
            os.path.join(base_path, model_name),
            cfg.get("number", None),
            ["checkpoints", "wandb"],
        )
        RUN_NAME = (
            f"{cfg.config_name} // {cfg.dataset.dataset} - "
            + re.findall(r"\d+", OUTPUT_PATH)[-1]
        )
        CHECKPOINT_PATH = OUTPUT_PATH + "/checkpoints/"
    accelerator.wait_for_everyone()
    OUTPUT_PATH = broadcast_string(OUTPUT_PATH, accelerator)
    RUN_NAME = broadcast_string(RUN_NAME, accelerator)
    CHECKPOINT_PATH = broadcast_string(CHECKPOINT_PATH, accelerator)

    # Connect to WandB
    if cfg.use_wandb and accelerator.is_main_process:
        # Set Environment variables for WandB
        os.environ["WANDB_DIR"] = OUTPUT_PATH
        os.environ["WANDB_PROJECT"] = dot_env.PROJECT_NAME

        # Figure out the tag system
        tags = [cfg.model.model]
        tags += (cfg.get("tags")) if cfg.get("tags", False) else None

        # Boot up the WandB connection
        wandb.login(key=dot_env.MODEL_TRAIN_KEY)
        wandb.init(
            project=dot_env.PROJECT_NAME,
            name=RUN_NAME,
            config=dict(OmegaConf.to_container(cfg, resolve=True)),
            entity=dot_env.ENTITY,
            tags=tags,
            monitor_gym=True,
        )

    # Disable WandB if not provided
    else:
        wandb.init(mode="disabled")

    #   endregion

    #
    #   ? 💽 DATASET PROCESSING
    #   region
    #

    # Read the datasets
    datasets = dataset_reader(cfg.dataset_path)

    # Create data loaders
    dataloaders = DictObj({})
    for i in datasets:
        dataloaders[i] = DataLoader(
            datasets[i],
            batch_size=cfg.action.train.single.batch_size,
            shuffle=True,
        )

    #   endregion

    #
    #   ? 🏃‍♂️ TRAIN PROCESS
    #   region
    #

    # Initialize the model
    model = getattr(models, f"{cfg.model.model.upper()}Model")(
        input_size=datasets.total.sample_num_features,
        output_size=datasets.total.sample_num_labels,
        **cfg.model.args,
    )

    # Define the loss function, optimizer, and the scheduler
    criterion = getattr(torch.nn.modules, cfg.criterion.loss)()
    optimizer = getattr(torch.optim, cfg.optimizer.optimizer)(
        model.parameters(), **cfg.optimizer.args
    )
    scheduler = getattr(torch.optim.lr_scheduler, cfg.scheduler.scheduler)(
        optimizer, **cfg.scheduler.args
    )

    # Prepare model, optimizer, and data loaders with Accelerator
    model, optimizer = accelerator.prepare(model, optimizer)
    for i in dataloaders:
        dataloaders[i] = accelerator.prepare(dataloaders[i])

    # Initialize tracker(s)
    best_metric_score = float("inf")
    best_metric: str = cfg.action.train.single.best_model_metric.metric
    best_metric_name = best_metric.replace("evaluation/", "").upper()
    metric_strategy = cfg.action.train.single.best_model_metric.strategy

    # Make the evaluation list for validation evaluations
    evaluation_list = build_evaluation_list(cfg.evaluation.components)

    # Run the training loop
    num_epochs = cfg.action.train.single.epochs
    for epoch in tqdm(range(num_epochs)):

        #
        #   ? 📉 TRAIN STEP
        #   region
        #

        # Initialize the model to train mode
        model.train()

        # Reset and track the current running loss for the current epoch
        train_loss = 0

        # Iterate through all batches
        for inputs, labels, _ in dataloaders.train:

            # Forward propagate
            optimizer.zero_grad()
            outputs = model(inputs)

            # Backward propagate
            loss = criterion(outputs, labels)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            # Keep track of the loss for the epoch
            train_loss += loss.item() * inputs.size(0)

        # Calculate the final train loss
        train_loss /= len(dataloaders.train)

        #   endregion

        #
        #   ? ✔️ VALIDATION STEP
        #   region
        #

        # Turn the model into evaluation mode
        model.eval()

        # Initialize a loss and evaluation trackers
        valid_loss = 0.0
        all_predictions = []
        all_labels = []

        # Iterate through the validation dataset and
        # inference the model's ability
        with torch.no_grad():
            for inputs, labels, _ in dataloaders.valid:

                # Get the outputs of the model and count that towards the loss
                outputs = model(inputs.float())
                loss = criterion(outputs, labels.float())
                valid_loss += loss.item() * inputs.size(0)

                # Get the converted values for RMSE, MAE, R2, and Adjusted R2
                converted_predictions = datasets.total.inverse_transform(
                    outputs.cpu()
                )
                converted_labels = datasets.total.inverse_transform(
                    labels.cpu()
                )

                # Dump labels and predictions into a pool
                all_predictions.extend(converted_predictions)
                all_labels.extend(converted_labels)

        # Calculate the validation loss
        valid_loss /= len(dataloaders.valid.dataset)

        # Combine loss and evaluation metrics for dynamic tracking
        evaluation_results = {}
        for item in evaluation_list:
            evaluation_results.update(
                item(all_predictions, all_labels, inputs)
            )
        combined_eval_metrics = {
            "valid/valid_loss": valid_loss,
            **{f"evaluation/{k}": v for k, v in evaluation_results.items()},
        }

        # If the validation loss is better than the best val loss, track it
        # and save the model
        if accelerator.is_main_process:
            banner = ""
            target_metric = combined_eval_metrics[best_metric]
            if OPS[metric_strategy](target_metric, best_metric_score):
                # Save the new valid loss
                best_metric_score = target_metric

                # Delete the saved model
                for filename in os.listdir(CHECKPOINT_PATH):
                    if "best_model" in filename:
                        file_path = os.path.join(CHECKPOINT_PATH, filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)

                # Save the model
                save_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    valid_loss,
                    CHECKPOINT_PATH + f"/best_model_{epoch + 1}.ckpt",
                )

                # Set the banner
                banner = (
                    f"{Fore.RED}[ NEW BEST {best_metric_name} ]"
                    f"{Style.RESET_ALL}"
                )
        accelerator.wait_for_everyone()

        #   endregion

        #
        #   ? ⚙️ COMPILATION STEP
        #   region
        #

        # Extract the learning rate
        learning_rate = optimizer.param_groups[0]["lr"]

        # Print the loss
        if accelerator.is_main_process:
            print(
                f"\nEPOCH {epoch + 1}/{num_epochs} "
                + f">> Learning Rate: {learning_rate:.9f} | "
                + f"Train Loss: {train_loss:.4f} | "
                + f"Valid Loss: {valid_loss:.4f} | "
                + " | ".join(
                    [
                        (
                            f"{Fore.CYAN}{key.upper()}: {value:.4f}"
                            f"{Style.RESET_ALL}"
                            if key.upper() == best_metric_name.upper()
                            else f"{key.upper()}: {value:.4f}"
                        )
                        for key, value in evaluation_results.items()
                    ]
                )
                + f" | {banner}"
            )

            # Log to wandb
            wandb.log(
                {
                    "train/epoch": epoch + 1,
                    "train/learning_rate": learning_rate,
                    "train/train_loss": train_loss,
                    **combined_eval_metrics,
                }
            )

            # Save the current model
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                valid_loss,
                CHECKPOINT_PATH + "/last_model.ckpt",
            )
        accelerator.wait_for_everyone()

        #   endregion

    #   endregion

    # Finish up with WandB
    wandb.finish()


# Run the main function
if __name__ == "__main__":
    main()
