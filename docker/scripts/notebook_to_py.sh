#!/bin/bash

# Define the directories
notebook_dir="../notebooks"
python_dir="../py_notebooks"

# Convert relative paths to absolute paths
notebook_dir=$(realpath "$notebook_dir")
python_dir=$(realpath "$python_dir")

# Remove trailing slash from python_dir if present
python_dir="${python_dir%/}"

# Loop over all notebooks in the notebook directory
for notebook in "$notebook_dir"/*.ipynb
do
    # Get the base name of the notebook
    base_name=$(basename "$notebook" .ipynb)

    # Pair the notebook with a Python file in the Python directory
    jupytext --to py:percent --output "$python_dir/$base_name.py" "$notebook"
done
