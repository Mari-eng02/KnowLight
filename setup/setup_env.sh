#!/bin/bash

echo "========================================================"
echo "   Setting up KnowLight Conda Environment"
echo "========================================================"

echo "Cleaning corrupted conda package archives..."
conda clean --tarballs -y

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
conda env create -f "$SCRIPT_DIR/environment.yml"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================"
    echo "   Environment setup complete!"
    echo "   To activate run: conda activate knowlight"
    echo "========================================================"
else
    echo "[ERROR] Failed to create conda environment."
    exit 1
fi
