#!/bin/bash

# Path to your Python script
SCRIPT="run_LebwohlLasher_numpyXcython.py"

# Fixed parameters
SIZE=20
ITER=1000
PLOTFLAG=0
CSV_OUT="numpyXcython_temp.txt"

# Create or overwrite CSV file with header
echo "temperature,order" > "$CSV_OUT"

# Generate a list of temperatures using Python (float values from 0 to 1)
TEMPS=($(python3 -c "import numpy as np; print(' '.join(map(str, np.linspace(0.05, 1.6, 30))))"))

# Loop over all combinations
for ((i=1; i<20; i++)); do
  for TEMP in "${TEMPS[@]}"; do
    # Run the simulation and capture output
    OUTPUT=$(python3 "$SCRIPT" "$ITER" "$SIZE" "$TEMP" "$PLOTFLAG")

    # Extract values using grep (use standard grep with Perl mode)
    ORDER=$(echo "$OUTPUT" | ggrep -oP 'Order: \K[0-9.]+')
    TIME=$(echo "$OUTPUT" | ggrep -oP 'Time: \K[0-9.]+')

    # Append to CSV
    echo "$TEMP,$ORDER" >> "$CSV_OUT"

    echo "Logged: iter=$ITER, size=$SIZE, temp=$TEMP, order=$ORDER, time=$TIME"
  done
done

