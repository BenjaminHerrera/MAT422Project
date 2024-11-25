#!/bin/bash

# Prompt the user the slurm output
read -p "Where should the slurm output be (default is /scratch/%u/projects/palay/slurm_dump/train_sweep_model_%j/): " parition_type
slurm_output=${slurm_output:-/scratch/%u/projects/palay/slurm_dump/train_sweep_model_%j/}

# Prompt the user for GPU type
read -p "Enter the GPU type (leave empty for default allocation): " gpu_type
gpu_type=${gpu_type:-}

# Prompt the user for the number of GPUs to allocate
read -p "Enter the number of GPUs to allocate (default is 1): " gpu_count
gpu_count=${gpu_count:-1}

# Prompt the user for the use of which partition
read -p "What partition should we use (default is general): " parition_type
parition_type=${parition_type:-general}

# Prompt the user for the use which QOS
read -p "What QOS should we use (default is private): " qos_type
qos_type=${qos_type:-private}

# Prompt the user whether to use wandb (default is true)
read -p "Use wandb? (default is 'true', enter 'false' to disable): " use_wandb
use_wandb=${use_wandb:-true}

# Prompt the user for the execution time (default is 1-00:00:00)
read -p "Enter the execution time (default is '1-00:00:00', format d-hh:mm:ss): " exec_time
exec_time=${exec_time:-1-00:00:00}

# Prompt the user for the number of runs
read -p "Enter the number of runs to execute (default is 10): " number_of_runs
number_of_runs=${number_of_runs:-10}

# Prompt the user for the specific model to run multiple training sessions
read -p "Enter the specific model to train (default is what is set by the execution config): " model_type

# Prompt the user for the specific optimizer to run multiple training sessions
read -p "Enter the specific optimizer to use (default is what is set by the execution config): " optimizer_type

# Prompt the execution config to run
read -p "Provide the configuration file to execute the training: " execution_config

# Prompt the batch size for the training
read -p "What batch size would you like to use (default is what is set by the execution config): " batch_size

# Prompt the save folder
read -p "Provide the save folder to save executions (default is multiple_runs): " save_dir
save_dir=${save_dir:-multiple_runs}

# Prompt any additional overrides or configurations
read -p "Provide additional overrides or configurations (leave empty for no additional information): " additional_args
additional_args=${additional_args:-}

# Determine the argument to override the model type
if [ -z "$model_type" ]; then
    model_type_arg=""
else
    model_type_arg="model=$model_type"
fi

# Determine the argument to override the optimizer type
if [ -z "$optimizer_type" ]; then
    optimizer_type_arg=""
else
    optimizer_type_arg="optimizer=$optimizer_type"
fi

# Make multiple runs of the execution
for ((i = 1; i <= number_of_runs; i++)); do
    # Create a random port
    random_port=$((10000 + RANDOM % 50001))

    # Create additional argument string
    additional_args_string="$model_type_arg $optimizer_type_arg $additional_args +tags=[multiple_run] ++number=$i"

    # Dispatch the run
    bash train_single_model.sh <<EOT
$slurm_output
$gpu_type
$gpu_count
$parition_type
$qos_type
$use_wandb
$exec_time
$execution_config
$batch_size
$random_port
$save_dir
$additional_args_string
EOT

done
