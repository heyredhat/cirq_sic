#!/bin/bash

# --- Configuration ---
# The file containing the argument combinations, one per line.
# Defaults to "args.txt".
ARGS_FILE=${1:-"args.txt"}

# The Python script to execute.
# Defaults to "run.py".
PYTHON_SCRIPT=${2:-"run.py"}

# --- Pre-flight Checks ---
# Check if the arguments file exists
if [ ! -f "$ARGS_FILE" ]; then
    echo "Error: Arguments file not found at '$ARGS_FILE'"
    exit 1
fi

# Check if the python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at '$PYTHON_SCRIPT'"
    exit 1
fi

# --- Main Execution ---
echo "Starting parallel execution of '$PYTHON_SCRIPT'..."
echo "Reading arguments from '$ARGS_FILE'..."
echo "----------------------------------------------------"

cat "$ARGS_FILE" | xargs -P 0 -I {} sh -c "python3 $PYTHON_SCRIPT {}"

echo "----------------------------------------------------"
echo "All processes launched."

