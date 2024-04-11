# For use only when attempting to run on multiple GPUS
# This script is used to set the environment variables for DDP (Distributed Data Parallel) training

export MASTER_ADDR=$(hostname)
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_PORT=29500 # default from torch launcher