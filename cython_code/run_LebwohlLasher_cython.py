import sys
from LebwohlLasher_cython import main

if int(len(sys.argv)) == 5:
    main(sys.argv[0],int(sys.argv[1]),int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))
else:
    print("Usage: {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOGFLAG>".format(sys.argv[0]))

