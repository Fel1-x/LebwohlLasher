#!/bin/bash

SCRIPT="LebwohlLasher_mpi4pi.py"
SIZE=20
ITER=1000
PLOTFLAG=0
CSV_OUT="mpi4py_3_temp.txt"
TASKS=4

echo "temperature,order" > "$CSV_OUT"

TEMPS=($(python3 -c "import numpy as np; print(' '.join(map(str, np.linspace(0.05, 1.6, 30))))"))

for ((i=1; i<20; i++)); do
  for TEMP in "${TEMPS[@]}"; do
    OUTPUT=$(mpiexec -n "$TASKS" python "$SCRIPT" "$ITER" "$SIZE" "$TEMP" "$PLOTFLAG")
    ORDER=$(echo "$OUTPUT" | ggrep -oP 'Order: \K[0-9.]+')
    TIME=$(echo "$OUTPUT" | ggrep -oP 'Time: \K[0-9.]+')
    echo "$TEMP,$ORDER" >> "$CSV_OUT"
    echo "Logged: iter=$ITER, size=$SIZE, temp=$TEMP, order=$ORDER, time=$TIME"
  done
done

