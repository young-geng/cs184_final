#! /bin/bash

echo "Creating conda environment..."
conda env create -f environment.yml
conda env update
