#!/usr/bin/bash

# Load pytorch w/GPU support
module load pytorch/2.1.0-cu12 
module list

# Install additional packages
# The modulefiles automatically set the $PYTHONUSERBASE environment variable for you, 
#   so that you will always have your custom packages every time you load that module.
pip install --user \
seaborn==0.11.2 \
silx==1.1.2 \
numba==0.57.0 \
numpy==1.24.4 \
cython==0.29.30 \
blosc2==2.0.0 \
triton \
energyflow \
vector \
awkward \
uproot 

echo "Before second module load"
module load pytorch/2.1.0-cu12
module list
echo "After second module load"


cd  
module use heppy/modules
module load heppy 

echo "Modules loaded"

export PYTHONPATH="/global/common/software/nersc/pm-stable/sw/pytorch/2.1.0/lib/python3.10:$PYTHONPATH"
cd GNNs-and-Jets
alias python='python3.10'
echo "Alias set"

# The following packages are already installed by the pytorch module
# matplotlib==3.5.1 \
# networkx==2.7.1 \
# numpy==1.21.2 \
# pandas==1.4.1 \
# pyyaml==6.0 \
# scikit-learn==1.0.2 \
# torch==1.11 \
# torch-geometric==2.0.4 \
# torch-scatter==2.0.9 \
# torch-sparse==0.6.13
