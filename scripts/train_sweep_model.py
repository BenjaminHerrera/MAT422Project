# Set the python path
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Util related imports
from utils.scripts import prepare_config

# Execution related imports
import subprocess
import hydra

# Datastructure related imports
from omegaconf import DictConfig
import json

# Miscellaneous imports
from itertools import product


@hydra.main(version_base=None, config_path="../configs/", config_name="fcnn_6")
def sweep_initializer(cfg: DictConfig):
    """Main execution function to run the multiple

    Args:
        cfg (DictConfig): Configuration for the run
    """
    # Define constants and trackers
    HYPERPARAMETERS = ["model", "optimizer"]

    # Get the config values from the targeted parameters
    configs = {}
    if "SLURM_JOB_ID" not in os.environ:
        relative_path = "../configs/"
    else:
        if os.environ["SLURM_JOB_NAME"] == "interactive":
            relative_path = "../configs/"
        else:
            relative_path = "../../configs/"
    for item in HYPERPARAMETERS:
        if item == "model":
            item += f"/{cfg.model.model}"
        configs[item] = [
            i.split(".")[0] for i in os.listdir(relative_path + item)
        ]

    # Resolve any system placeholder in the config and configurations
    cfg = prepare_config(cfg)

    # Get a combination of the sweep parameters
    keys = configs.keys()
    values_list = [configs[key] for key in keys]
    combinations = product(*values_list)
    sweep_parameters = [dict(zip(keys, combo)) for combo in combinations]

    # Iterate through the sweep parameter combinations
    os.chdir(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "sbatch")
    )
    for idx, i in enumerate(sweep_parameters):
        # Create overrides
        override = ""
        for j in i:
            override += (
                f"{j.split('/')[0]}="
                + "/".join(j.split("/")[1:] + [i[j]])
                + " "
            )

        # Run a sweep run
        values = [
            cfg.action.train.sweep.execution_values.slurm_output,
            cfg.action.train.sweep.execution_values.gpu_type,
            cfg.action.train.sweep.execution_values.gpu_count,
            cfg.action.train.sweep.execution_values.partition,
            cfg.action.train.sweep.execution_values.qos,
            cfg.action.train.sweep.execution_values.use_wandb,
            cfg.action.train.sweep.execution_values.execution_time,
            cfg.action.train.sweep.execution_values.config_name,
            cfg.action.train.sweep.execution_values.batch_size,
            cfg.action.train.sweep.execution_values.unique_port,
            cfg.action.train.sweep.execution_values.save_folder,
            "+tags=[sweep] ++action.train.single.epochs="
            f"{cfg.action.train.sweep.max_sweep_count} {override}"
            f"++number={idx+1}",
        ]
        values = [str(i) for i in values]
        input_str = "\n".join(values) + "\n"
        process = subprocess.Popen(
            ["bash", "train_single_model.sh"], stdin=subprocess.PIPE
        )
        process.communicate(input=input_str.encode())

        # Print success
        print("Executed the following run with parameter values: ")
        print(json.dumps(i, indent=4))
        print(json.dumps(values, indent=4))
        print("\n" * 2)


# Run the sweep function
if __name__ == "__main__":
    sweep_initializer()
