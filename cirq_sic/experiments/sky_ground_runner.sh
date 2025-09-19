#!/bin/bash

# --- Configuration ---
# The file containing the argument combinations, one per line.
# Fallback to "znorm_search_args.txt" if no first argument is provided.
ARGS_FILE=${1:-"sky_ground_args.txt"}

# The Python script to execute.
# Fallback to "znorm_search.py" if no second argument is provided.
PYTHON_SCRIPT=${2:-"run_sky_ground.py"}

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

# The core command for parallel execution.
# - `cat $ARGS_FILE`: Reads the content of the arguments file.
# - `|`: Pipes each line of the file to the xargs command.
# - `xargs -P 0`: Executes the following command in parallel.
#   -P 0 tells xargs to use as many processes as there are CPU cores.
# - `xargs -I {}`: Replaces `{}` with the entire line of input.
# - `sh -c "..."`: This is the crucial part. It starts a new shell for each
#   line and executes the command inside the quotes. This ensures that the
#   argument string from the file is correctly parsed by the shell before
#   being passed to the Python script.
# - `python3 $PYTHON_SCRIPT {}`: The command to be executed for each line.
cat "$ARGS_FILE" | xargs -P 0 -I {} sh -c "python3 $PYTHON_SCRIPT {}"

echo "----------------------------------------------------"
echo "All processes launched."

