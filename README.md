# Accelerated Python Approaches: Speeding Up A Lebwohl-Lasher Simulation

This repository contains seven methods to accelerate a serial Python implementation of the Lebwohl-Lasher model; simulation of liquid crystal behaviour. Each method implements a different optimisation strategy with combined approaches also explored.

---

## Project Structure

### 1. Serial Version
A basic serial implementation, provided by Dr. Simon Hanna.

```bash
python LebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>
```

### 2. Numba
Uses the [Numba](https://numba.pydata.org/) just-in-time compiler, converting the script to machine code at compilation time.

```bash
python LebwohlLasher_numba.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>
```


### 3. Cython
Compiles critical sections (inefficient loops) using [Cython](https://cython.readthedocs.io/en/latest/src/quickstart/install.html).

#### Install Cython:
```bash
pip install Cython
```

#### Compile:
```bash
CC=gcc-15 python setup_LebwohlLasher_cython.py build_ext -fi
```

#### Run:
```bash
python run_LebwohlLasher_cython.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>
```


### 4. NumPy
Vectorized implementation using NumPy arrays for efficient operations.

```bash
python LebwohlLasher_numpy.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>
```


### 5. OpenMP (via Cython)
Parallelised using OpenMP within Cython.

#### Install OpenMP:

```bash
brew install llvm
brew install libomp
```

#### Compile:
```bash
CC=gcc-15 python setup_LebwohlLasher_openmp.py build_ext -fi
```

#### Run:
```bash
python LebwohlLasher_openmp.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> <THREADS>
```


### 6. MPI (via mpi4py)
Distributed memory parallelisation using [mpi4py](https://mpi4py.readthedocs.io/en/stable/).

#### Install MPI:
```bash
brew install open-mpi
```

#### Install mpi4py:
```bash
pip install mpi4py
```

#### Run:
```bash
mpiexec -n <TASK_COUNT> python LebwohlLasher_mpi4py.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>
```


### 7. Combined Approaches
Implementations combining multiple acceleration approaches:

- `LebwohlLasher_numpyxmpi4py.py` — NumPy + MPI
- `LebwohlLasher_numpyxcython.py` — NumPy + Cython

Usage mirrors the respective base methods.


## Input Parameters

| Argument     | Description                                          |
|--------------|------------------------------------------------------|
| `ITERATIONS` | The number of MC steps for a given simulation        |
| `SIZE`       | Lattice size, integer input for a side length        |
| `TEMPERATURE`| Reduced temperature, T*, for a simulation            |
| `PLOTFLAG`   | 0 for no plot, 1 for energy plot and 2 for angle plot|
| `THREADS`    | (OpenMP only) Number of threads                      |
| `TASK_COUNT` | (MPI only) Number of parallel processes - must be >1 |

---

## Notes
- All implementations assume Python 3.
- Plotting requires `matplotlib`.
- Task count, for mpi implementations must be 2 or greater, as the master task is not a worker task.
