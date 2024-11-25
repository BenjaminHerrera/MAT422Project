#!/bin/bash

# Prompt the user the slurm output
read -p "Where should the slurm output be (default is /scratch/%u/projects/palay/slurm_dump/post_process_%j/): " parition_type
slurm_output=${slurm_output:-/scratch/%u/projects/palay/slurm_dump/post_process_%j/}

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

# Prompt the user for the execution time (default is 1-00:00:00)
read -p "Enter the execution time (default is '1-00:00:00', format d-hh:mm:ss): " exec_time
exec_time=${exec_time:-1-00:00:00}

# Prompt the user for the specific model to run multiple training sessions
read -p "Enter the specific model to train (default is what is set by the execution config): " model_type

# Prompt the execution config to run
read -p "Provide the configuration file to execute the training: " execution_config

# Determine the argument to override the model type
if [ -z "$model_type" ]; then
    model_type_arg=""
else
    model_type_arg="model=$model_type"
fi

# Determine the GPU allocation string
if [ -z "$gpu_type" ]; then
    gpu_allocation="--gpus=$gpu_count"
else
    gpu_allocation="--gpus=$gpu_type:$gpu_count"
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
#SBATCH --job-name="PALAY // Post Process Evaluation - $execution_config"                   # job name

# Purge any remaining modules
module purge

# Load the mamba module 
module load mamba

# Activate the inference environment
source activate palay

# Generate a random port number between 10000 and 60000
PORT=\$((10000 + RANDOM % 50001))

# Log the system resources that the job sees
echo "Printing resources for job ID: \$SLURM_JOB_ID"
scontrol show job \$SLURM_JOB_ID
echo ""
nvidia-smi 
echo ""

# Run the python script
accelerate launch --main_process_port \$PORT --config_file ../../configs/accelerate/gpu${gpu_count}.yaml ../post_process_evaluation.py --config-name=$execution_config $model_type_arg


# Echo that the job is finished
echo "Finished"
EOT
