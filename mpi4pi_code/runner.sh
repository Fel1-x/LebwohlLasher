#!/bin/bash

SCRIPT="LebwohlLasher_mpi4pi.py"
TEMP=0.5
PLOTFLAG=0
CSV_OUT="results3.csv"
TASKS=4

echo "iterations,size,temperature,order,time" > "$CSV_OUT"

ITERATIONS=($(python3 -c "import numpy as np; print(' '.join(map(str, map(int, np.logspace(0, np.log10(500), 15, base=10).round()))))"))

SIZES=($(python3 -c "import numpy as np; print(' '.join(map(str, map(int, np.logspace(0, np.log10(100), 15, base=10).round()))))"))

for size in "${SIZES[@]}"; do
  iter=500
  OUTPUT=$(mpiexec -n "$TASKS" python "$SCRIPT" "$iter" "$size" "$TEMP" "$PLOTFLAG")
  ORDER=$(echo "$OUTPUT" | ggrep -oP 'Order: \K[0-9.]+')
  TIME=$(echo "$OUTPUT" | ggrep -oP 'Time: \K[0-9.]+')
  echo "$iter,$size,$TEMP,$ORDER,$TIME" >> "$CSV_OUT"
  echo "Logged: iter=$iter, size=$size, order=$ORDER, time=$TIME"
done
for iter in "${ITERATIONS[@]}"; do
  size=100
  OUTPUT=$(mpiexec -n "$TASKS" python "$SCRIPT" "$iter" "$size" "$TEMP" "$PLOTFLAG")
  ORDER=$(echo "$OUTPUT" | ggrep -oP 'Order: \K[0-9.]+')
  TIME=$(echo "$OUTPUT" | ggrep -oP 'Time: \K[0-9.]+')
  echo "$iter,$size,$TEMP,$ORDER,$TIME" >> "$CSV_OUT"
  echo "Logged: iter=$iter, size=$size, order=$ORDER, time=$TIME"
done
