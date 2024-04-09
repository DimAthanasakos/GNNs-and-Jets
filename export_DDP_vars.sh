export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

export MASTER_ADDR='localhost'
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=4
export MASTER_PORT=29500 # default from torch launcher