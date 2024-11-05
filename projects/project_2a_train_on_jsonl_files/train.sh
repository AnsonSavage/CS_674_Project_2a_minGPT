#!/bin/bash

#SBATCH --time=12:00:00           # Walltime
#SBATCH --ntasks=1              # Number of processor cores (i.e., tasks)
#SBATCH --gpus=1                  # Number of GPUs
#SBATCH --mem-per-cpu=262144M       # Memory per CPU core
#SBATCH -J "train_gpt2"   # Job name
#SBATCH --mail-user=ansonsav@byu.edu             # Email address
#SBATCH --mail-type=BEGIN,END,FAIL              # Email types
#SBATCH --qos=dw87                                # Quality of Service

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# Load CPU information
lscpu | grep "^CPU"

echo "(Number of CPUs)"
grep -c ^processor /proc/cpuinfo

mamba activate anson_utils_test
cd /home/ansonsav/cs_674/project_2a_minGPT/minGPT/projects/project_2a_train_on_jsonl_files
python3 ./train_on_jsonl.py