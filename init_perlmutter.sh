#!/usr/bin/bash

# Load PyTorch module (provides Python 3.12)
module load pytorch/2.6.0
module list

# Upgrade setuptools first, as it helps with modern package installations
pip install --user --upgrade setuptools

# Install your packages - removed numpy pin, updated silx, removed build flags
echo "Installing Python packages..."
pip install --user \
    seaborn==0.11.2 \
    silx==2.2.1 \
    numpy \
    cython \
    blosc2 \
    triton \
    energyflow \
    vector \
    awkward \
    uproot \
    matplotlib==3.7.2 \
    mplhep==0.3.31 \
    PyYAML==6.0.1 \
    torchinfo==1.8.0 \
    pyjet \
    coffea \
    qpth \
    h5py # Explicitly listing h5py doesn't hurt

echo "Installation complete."