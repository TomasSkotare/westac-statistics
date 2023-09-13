#!/bin/bash

# Define the directories
notebook_dir="../work/notebooks"
python_dir="../py_notebooks"

# Convert relative paths to absolute paths
python_dir=$(realpath "$python_dir")
notebook_dir=$(realpath "$notebook_dir")

# Remove trailing slash from notebook_dir if present
notebook_dir="${notebook_dir%/}"

# Loop over all Python files in the Python directory
for python_file in "$python_dir"/*.py
do
    # Get the base name of the Python file
    base_name=$(basename "$python_file" .py)

    # Check if the corresponding notebook already exists
    if [ -e "$notebook_dir/$base_name.ipynb" ]
    then
        echo "Error: $notebook_dir/$base_name.ipynb already exists"
        continue
    fi

    # Convert the Python file to a notebook
    jupytext --to notebook --output "$notebook_dir/$base_name.ipynb" "$python_file"
done
