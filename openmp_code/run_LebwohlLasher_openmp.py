import sys
from LebwohlLasher_openmp import main

if int(len(sys.argv)) == 6:
    main(sys.argv[0],int(sys.argv[1]),int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
else:
    print("Usage: {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOGFLAG> <THREADS>".format(sys.argv[0]))

