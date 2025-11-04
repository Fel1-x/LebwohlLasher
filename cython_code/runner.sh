#!/bin/bash

SCRIPT="run_LebwohlLasher_cython.py"
TEMP=0.5
PLOTFLAG=0
CSV_OUT="results.csv"

# Create the csv file
echo "iterations,size,temperature,order,time" > "$CSV_OUT"

# Generate 15 values from 1 to 500 for iterations, split in a logarithmic manner for even plotting.
# Python is imported to generate the spacings.
ITERATIONS=($(python3 -c "import numpy as np; print(' '.join(map(str, map(int, np.logspace(0, np.log10(500), 15, base=10).round()))))"))

# Similar to above but with sizes between 0 and 100
SIZES=($(python3 -c "import numpy as np; print(' '.join(map(str, map(int, np.logspace(0, np.log10(100), 15, base=10).round()))))"))

# Loop over sizes
for size in "${SIZES[@]}"; do
  iter=500
  OUTPUT=$(python3 "$SCRIPT" "$iter" "$size" "$TEMP" "$PLOTFLAG")
  ORDER=$(echo "$OUTPUT" | ggrep -oP 'Order: \K[0-9.]+')
  TIME=$(echo "$OUTPUT" | ggrep -oP 'Time: \K[0-9.]+')
  echo "$iter,$size,$TEMP,$ORDER,$TIME" >> "$CSV_OUT"
  echo "Logged: iter=$iter, size=$size, order=$ORDER, time=$TIME"
done

# Loop over iterations
for iter in "${ITERATIONS[@]}"; do
  size=100
  OUTPUT=$(python3 "$SCRIPT" "$iter" "$size" "$TEMP" "$PLOTFLAG")
  ORDER=$(echo "$OUTPUT" | ggrep -oP 'Order: \K[0-9.]+')
  TIME=$(echo "$OUTPUT" | ggrep -oP 'Time: \K[0-9.]+')
  echo "$iter,$size,$TEMP,$ORDER,$TIME" >> "$CSV_OUT"
  echo "Logged: iter=$iter, size=$size, order=$ORDER, time=$TIME"
done

