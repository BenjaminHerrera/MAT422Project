#!/bin/bash

# Prompt the user the slurm output
read -p "Where should the slurm output be (default is /scratch/%u/projects/palay/slurm_dump/train_model_%j/): " parition_type
slurm_output=${slurm_output:-/scratch/%u/projects/palay/slurm_dump/train_model_%j/}

# Prompt the user for GPU type
read -p "Enter the GPU type (leave empty for default allocation): " gpu_type

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

# Prompt the execution config to run
read -p "Provide the configuration file to execute the training: " execution_config

# Prompt the batch size for the training
read -p "What batch size would you like to use (default is what is set by the execution config): " batch_size

# Read the unique port number passed from the main script
read -p "Enter the unique port number (leave empty for default port): " port

# Prompt the save folder
read -p "Provide the save folder to save executions (leave empty for no folder path): " save_dir

# Prompt any additional overrides or configurations
read -p "Provide additional overrides or configurations (leave empty for no additional information): " additional_args
additional_args=${additional_args:-}

# Determine the save_dir arg string
if [ -z "$save_dir" ]; then
    save_dir_arg=""
else
    save_dir_arg="++save_folder=$save_dir"
fi

# Determine the GPU allocation string
if [ -z "$port" ]; then
    port_arg=""
else
    port_arg="--main_process_port $port"
fi

# Determine the GPU allocation string
if [ -z "$gpu_type" ]; then
    gpu_allocation="--gpus=$gpu_count"
else
    gpu_allocation="--gpus=$gpu_type:$gpu_count"
fi

# Determine the batchsize argument
if [ -z "$batch_size" ]; then
    batch_size_arg=""
else
    batch_size_arg="++action.train.single.batch_size=$batch_size"
fi

# Create and submit the SLURM job
sbatch <<EOT
#!/bin/bash

#SBATCH -n 1                                                                                # number of nodes
#SBATCH -c 2                                                                                # number of cores to allocate
#SBATCH --mem=4G                                                                            # RAM allocation
#SBATCH $gpu_allocation                                                                     # GPU Allocation
#SBATCH -t $exec_time                                                                       # time in d-hh:mm:ss
#SBATCH -p $parition_type                                                                   # partition 
#SBATCH -q $qos_type                                                                        # QOS
#SBATCH -o $slurm_output/slurm.%j.out                                                       # file to save job's STDOUT (%j = JobId)
#SBATCH -e $slurm_output/slurm.%j.out                                                       # file to save job's STDERR (%j = JobId)
#SBATCH --export=NONE                                                                       # keep environment clean
#!BATCH --mail-type=ALL                                                                     # send an e-mail when a job starts, stops, or fails
#!BATCH --mail-user=%u@asu.edu                                                              # notify email (%u expands -> username)
#SBATCH --job-name="PALAY // Train Model - $execution_config"                               # job name

# Purge any remaining modules
module purge

# Load the mamba module 
module load mamba

# Activate the inference environment
source activate palay

# Log the system resources that the job sees
echo "Printing resources for job ID: \$SLURM_JOB_ID"
scontrol show job \$SLURM_JOB_ID
echo ""
nvidia-smi 
echo ""

# Run the python script
accelerate launch $port_arg --config_file ../../configs/accelerate/gpu${gpu_count}.yaml ../train_single_model.py --config-name=$execution_config ++use_wandb=$use_wandb $batch_size_arg $save_dir_arg $additional_args


# Echo that the job is finished
echo "Finished"
EOT