#!/bin/bash

# Prompt the user the slurm output
read -p "Where should the slurm output be (default is /scratch/%u/projects/palay/slurm_dump/pre_process_evaluation_%j/): " parition_type
slurm_output=${slurm_output:-/scratch/%u/projects/palay/slurm_dump/pre_process_evaluation_%j/}

# Prompt the execution config to run
read -p "Provide the configuration file to execute the training: " execution_config

# Create and submit the SLURM job
sbatch <<EOT
#!/bin/bash

#SBATCH -n 1                                                                                # number of nodes
#SBATCH -c 1                                                                                # number of cores to allocate
#SBATCH --mem=4G                                                                            # RAM allocation
#SBATCH -t 18:00:00                                                                         # time in d-hh:mm:ss
#SBATCH -q private                                                                          # QoS 
#SBATCH -p general                                                                          # Partition
#!BATCH --exclude=pcg008                                                                    # Exclude faulty nodes (SCRATCH)
#!BATCH --exclude=g050,g049                                                                 # Exclude faulty nodes (SOL)
#SBATCH -o $slurm_output/slurm.%j.out                                                       # file to save job's STDOUT (%j = JobId)
#SBATCH -e $slurm_output/slurm.%j.out                                                       # file to save job's STDERR (%j = JobId)
#SBATCH --export=NONE                                                                       # keep environment clean
#SBATCH --mail-type=ALL                                                                     # send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=%u@asu.edu                                                              # notify email (%u expands -> username)
#SBATCH --job-name="PALAY // Pre Process Evaluation - $execution_config"                    # job name

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
python ../pre_process_evaluation.py --config-name=$execution_config

# Echo that the job is finished
echo "Finished"
EOT
