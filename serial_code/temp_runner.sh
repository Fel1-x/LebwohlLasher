#!/bin/bash

SCRIPT="LebwohlLasher.py"
SIZE=20
ITER=1000
PLOTFLAG=0
CSV_OUT="results.csv"

# Create the csv file
echo "temperature,order" > "$CSV_OUT"

# A list of temperatures using python
TEMPS=($(python3 -c "import numpy as np; print(' '.join(map(str, np.linspace(0.05, 1.6, 30))))"))

# Loop over all combinations
for ((i=1; i<20; i++)); do
  for TEMP in "${TEMPS[@]}"; do
    OUTPUT=$(python3 "$SCRIPT" "$ITER" "$SIZE" "$TEMP" "$PLOTFLAG")
    ORDER=$(echo "$OUTPUT" | ggrep -oP 'Order: \K[0-9.]+')
    TIME=$(echo "$OUTPUT" | ggrep -oP 'Time: \K[0-9.]+')
    echo "$TEMP,$ORDER" >> "$CSV_OUT"

    echo "Logged: iter=$ITER, size=$SIZE, temp=$TEMP, order=$ORDER, time=$TIME"
  done
done

