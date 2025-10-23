#!/bin/bash

# Path to your Python script
SCRIPT="run_LebwohlLasher_cython.py"

# Fixed parameters
TEMP=0.5
PLOTFLAG=0
CSV_OUT="results.csv"

# Create or overwrite CSV file with header
echo "iterations,size,temperature,order,time" > "$CSV_OUT"

# Generate 15 log-spaced values from 1 to 500 for iterations
ITERATIONS=($(python3 -c "import numpy as np; print(' '.join(map(str, map(int, np.logspace(0, np.log10(500), 15, base=10).round()))))"))

# Generate 15 log-spaced values from 1 to 100 for size
SIZES=($(python3 -c "import numpy as np; print(' '.join(map(str, map(int, np.logspace(0, np.log10(100), 15, base=10).round()))))"))

# Loop over all combinations
for iter in "${ITERATIONS[@]}"; do
  for size in "${SIZES[@]}"; do
    # Run the simulation and capture output
    OUTPUT=$(python "$SCRIPT" "$iter" "$size" "$TEMP" "$PLOTFLAG")
    
    # Extract values using regex
    ORDER=$(echo "$OUTPUT" | ggrep -oP 'Order: \K[0-9.]+')
    TIME=$(echo "$OUTPUT" | ggrep -oP 'Time: \K[0-9.]+')

    # Append to CSV
    echo "$iter,$size,$TEMP,$ORDER,$TIME" >> "$CSV_OUT"
    
    echo "Logged: iter=$iter, size=$size, order=$ORDER, time=$TIME"
  done
done

