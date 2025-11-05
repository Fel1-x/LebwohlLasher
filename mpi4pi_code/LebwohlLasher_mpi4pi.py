"""
Mpi4Py Python Lebwohl-Lasher code.  Based on the paper
P.A. Lebwohl and G. Lasher, Phys. Rev. A, 6, 426-429 (1972).
This version in 2D.

Run at the command line with openmpi by typing:

mpiexec -n <TASK_COUNT> python LebwohlLasher_mpi4py.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

where:
  TASK_COUNT = total number of tasks to be used, must be 2 or greater.
  ITERATIONS = number of Monte Carlo steps, where 1MCS is when each cell
      has attempted a change once on average (i.e. SIZE*SIZE attempts)
  SIZE = side length of square lattice
  TEMPERATURE = reduced temperature in range 0.0 - 2.0.
  PLOTFLAG = 0 for no plot, 1 for energy plot and 2 for angle plot.
  
The initial configuration is set at random. The boundaries
are periodic throughout the simulation.  During the
time-stepping, an array containing two domains is used; these
domains alternate between old data and new data.

SH 16-Oct-23
"""
import os
from os.path import split

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from mpi4py import MPI
import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy.random import default_rng

#=======================================================================
def initdat(nmax):
    """
    Arguments:
      nmax (int) = size of lattice to create (nmax,nmax).
    Description:
      Function to create and initialise the main data array that holds
      the lattice.  Will return a square lattice (size nmax x nmax)
	  initialised with random orientations in the range [0,2pi].
	Returns:
	  arr (float(nmax,nmax)) = array to hold lattice.
    """
    arr = np.random.random_sample((nmax,nmax))*2.0*np.pi
    return arr
#=======================================================================
def plotdat(arr,pflag,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  pflag (int) = parameter to control plotting;
      nmax (int) = side length of square lattice.
    Description:
      Function to make a pretty plot of the data array.  Makes use of the
      quiver plot style in matplotlib.  Use pflag to control style:
        pflag = 0 for no plot (for scripted operation);
        pflag = 1 for energy plot;
        pflag = 2 for angles plot;
        pflag = 3 for black plot.
	  The angles plot uses a cyclic color map representing the range from
	  0 to pi.  The energy plot is normalised to the energy range of the
	  current frame.
	Returns:
      NULL
    """
    if pflag==0:
        return
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax,nmax))
    if pflag==1: # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        for i in range(nmax):
            for j in range(nmax):
                cols[i,j] = one_energy(arr,i,j,nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag==2: # colour the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = arr%np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0,pivot='middle',headwidth=1,scale=1.1*nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()
#=======================================================================
def savedat(arr,nsteps,Ts,runtime,ratio,energy,order,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  nsteps (int) = number of Monte Carlo steps (MCS) performed;
	  Ts (float) = reduced temperature (range 0 to 2);
	  ratio (float(nsteps)) = array of acceptance ratios per MCS;
	  energy (float(nsteps)) = array of reduced energies per MCS;
	  order (float(nsteps)) = array of order parameters per MCS;
      nmax (int) = side length of square lattice to simulated.
    Description:
      Function to save the energy, order and acceptance ratio
      per Monte Carlo step to text file.  Also saves run data in the
      header.  Filenames are generated automatically based on
      date and time at beginning of execution.
	Returns:
	  NULL
    """
    # Create filename based on current date and time.
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "LL-Output-{:s}.txt".format(current_datetime)
    FileOut = open(filename,"w")
    # Write a header with run parameters
    print("#=====================================================",file=FileOut)
    print("# File created:        {:s}".format(current_datetime),file=FileOut)
    print("# Size of lattice:     {:d}x{:d}".format(nmax,nmax),file=FileOut)
    print("# Number of MC steps:  {:d}".format(nsteps),file=FileOut)
    print("# Reduced temperature: {:5.3f}".format(Ts),file=FileOut)
    print("# Run time (s):        {:8.6f}".format(runtime),file=FileOut)
    print("#=====================================================",file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
    print("#=====================================================",file=FileOut)
    # Write the columns of data
    for i in range(nsteps+1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i,ratio[i],energy[i],order[i]),file=FileOut)
    FileOut.close()
#=======================================================================
def one_energy(arr,ix,iy,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  ix (int) = x lattice coordinate of cell;
	  iy (int) = y lattice coordinate of cell;
      nmax (int) = side length of square lattice.
    Description:
      Function that computes the energy of a single cell of the
      lattice taking into account periodic boundaries.  Working with
      reduced energy (U/epsilon), equivalent to setting epsilon=1 in
      equation (1) in the project notes.
	Returns:
	  en (float) = reduced energy of cell.
    """
    en = 0.0
    ixp = (ix+1)%nmax # These are the coordinates
    ixm = (ix-1)%nmax # of the neighbours
    iyp = (iy+1)%nmax # with wraparound
    iym = (iy-1)%nmax #
#
# Add together the 4 neighbour contributions
# to the energy
#
    ang = arr[ix,iy]-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iym]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    return en
#=======================================================================
def all_energy(arr,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to compute the energy of the entire lattice. Output
      is in reduced units (U/epsilon).
	Returns:
	  enall (float) = reduced energy of lattice.
    """
    enall = 0.0
    for i in range(nmax):
        for j in range(nmax):
            enall += one_energy(arr,i,j,nmax)
    return enall
#=======================================================================
def all_energy_MPI(arr,nmax,taskid,comm,row_start,row_end):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
      taskid (int) = task id of the MPI process.
      comm (MPI.COMM_WORLD) = MPI communicator object.
      row_start (int) = row start index of the MPI process.
      row_end (int) = row end index of the MPI process.
    Description:
      Function to compute the energy of the entire lattice. Output
      is in reduced units (U/epsilon).
	Returns:
	  enall_global (float) = reduced energy of lattice.
    """

    # Compute individual enall for each task
    enall = 0.0
    if taskid > 0:
        for i in range(row_start, row_end):
            for j in range(nmax):
                enall += one_energy(arr,i,j,nmax)
    # Sum the individual enalls on the MASTER task
    enall_global = comm.reduce(enall, op=MPI.SUM, root=0)

    # Only return enall_global if on the MASTER task.
    if taskid == 0:
        return enall_global
    else:
        return None
#=======================================================================
def get_order(arr,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to calculate the order parameter of a lattice
      using the Q tensor approach, as in equation (3) of the
      project notes.  Function returns S_lattice = max(eigenvalues(Q_ab)).
	Returns:
	  max(eigenvalues(Qab)) (float) = order parameter for lattice.
    """
    Qab = np.zeros((3,3))
    delta = np.eye(3,3)
    #
    # Generate a 3D unit vector for each cell (i,j) and
    # put it in a (3,i,j) array.
    #
    lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)
    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]
    Qab = Qab/(2*nmax*nmax)
    eigenvalues,eigenvectors = np.linalg.eig(Qab)
    return eigenvalues.max()
#=======================================================================
def get_order_MPI(arr,nmax,taskid,comm,row_start,row_end):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
      taskid (int) = task id of the MPI process.
      comm (MPI.COMM_WORLD) = MPI communicator object.
      row_start (int) = row start index of the MPI process.
      row_end (int) = row end index of the MPI process.
    Description:
      Function to calculate the order parameter of a lattice
      using the Q tensor approach, as in equation (3) of the
      project notes.  Function returns S_lattice = max(eigenvalues(Q_ab)).
	Returns:
	  max(eigenvalues(Qab)) (float) = order parameter for lattice.
    """
    Qab_local = np.zeros((3,3))
    delta = np.eye(3,3)
    #
    # Generate a 3D unit vector for each cell (i,j) and
    # put it in a (3,i,j) array.
    #
    lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)

    # Calculate the order for individual MPI tasks
    if taskid > 0:
        for a in range(3):
            for b in range(3):
                for i in range(row_start,row_end):
                    for j in range(nmax):
                        Qab_local[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]

    Qab = comm.reduce(Qab_local, op=MPI.SUM, root=0)

    # Sum the orders on the MASTER task
    if taskid == 0:
        Qab = Qab/(2*nmax*nmax)
        eigenvalues,eigenvectors = np.linalg.eig(Qab)
        return eigenvalues.max()
    else:
        return None
#=======================================================================
def MC_step(arr,Ts,nmax,taskid,comm,row_start,row_end):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  Ts (float) = reduced temperature (range 0 to 2);
      nmax (int) = side length of square lattice.
      taskid (int) = task id of the MPI process.
      comm (MPI.COMM_WORLD) = MPI communicator object.
      row_start (int) = row start index of the MPI process.
      row_end (int) = row end index of the MPI process.
    Description:
      Function to perform one MC step, which consists of an average
      of 1 attempted change per lattice site.  Working with reduced
      temperature Ts = kT/epsilon.  Function returns the acceptance
      ratio for information.  This is the fraction of attempted changes
      that are successful.  Generally aim to keep this around 0.5 for
      efficient simulation.
	Returns:
	  accept/(nmax**2) (float) = acceptance ratio for current MCS.
    """
    #
    # Pre-compute some random numbers.  This is faster than
    # using lots of individual calls.  "scale" sets the width
    # of the distribution for the angle changes - increases
    # with temperature.
    scale=0.1+Ts
    accept_local = 0

    # set the seed so they are different for each MPI task
    rng = default_rng(seed=taskid)

    if taskid != 0:
        # Define which rows are even, and within the rows assigned to the MPI task,
        # this will be used to determine the pool of random points that can be chosen from.
        even_rows = np.arange(row_start, row_end)[np.arange(row_start, row_end) % 2 == 0]

        # How many points does each task want to select
        num_samples = len(even_rows) * nmax

        # Generate random numbers based on which rows are available (above)
        xran = np.random.choice(even_rows, size=num_samples)
        yran = np.random.randint(0, nmax, size=num_samples)
        aran = np.random.normal(scale=scale, size=num_samples)

        # This loop selects points from the random tables above, and conducts the energy calculation
        # and boltzmann check, changing the point according to "ang" if it satisfies either condition.
        for point in range(num_samples):
            ix = xran[point]
            iy = yran[point]
            ang = aran[point]
            en0 = one_energy(arr,ix,iy,nmax)
            arr[ix,iy] += ang
            en1 = one_energy(arr,ix,iy,nmax)
            if en1<=en0:
                accept_local += 1
            else:
            # Now apply the Monte Carlo test - compare
            # exp( -(E_new - E_old) / T* ) >= rand(0,1)
                boltz = np.exp( -(en1 - en0) / Ts )

                if boltz >= rng.uniform(0.0,1.0):
                    accept_local += 1
                else:
                    arr[ix,iy] -= ang

    # Now we have to synchronise the lattice for future steps on the MASTER node then bcast back out
    local_rows = arr[row_start:row_end].copy()
    gathered = comm.gather(local_rows, root=0) # List of rows
    if taskid == 0:
        arr[:,:] = np.vstack(gathered)
    comm.Bcast(arr, root=0)

    # Now the same again for odd rows, EXCLUDING THE LAST ROW if nmax is odd to account for wraparound
    if taskid != 0:
        odd_rows = np.arange(row_start, row_end)[np.arange(row_start, row_end) % 2 == 1]
        if nmax % 2 == 1:
            odd_rows = odd_rows[:-1]

        num_samples = len(odd_rows) * nmax

        xran = np.random.choice(odd_rows, size=num_samples)
        yran = np.random.randint(0, nmax, size=num_samples)
        aran = np.random.normal(scale=scale, size=num_samples)

        for point in range(num_samples):
            ix = xran[point]
            iy = yran[point]
            ang = aran[point]
            en0 = one_energy(arr,ix,iy,nmax)
            arr[ix,iy] += ang
            en1 = one_energy(arr,ix,iy,nmax)
            if en1<=en0:
                accept_local += 1
            else:
            # Now apply the Monte Carlo test - compare
            # exp( -(E_new - E_old) / T* ) >= rand(0,1)
                boltz = np.exp( -(en1 - en0) / Ts )

                if boltz >= rng.uniform(0.0,1.0):
                    accept_local += 1
                else:
                    arr[ix,iy] -= ang

    # Synchronise again
    local_rows = arr[row_start:row_end].copy()
    gathered = comm.gather(local_rows, root=0) # List of rows
    if taskid == 0:
        arr[:,:] = np.vstack(gathered)
    comm.Bcast(arr, root=0)

    # Sample from just that last row in a similar way
    if nmax % 2 != 0 and row_end == nmax:
        yran = np.random.randint(0, nmax, size=nmax)
        aran = np.random.normal(scale=scale, size=nmax)

        for point in range(nmax):
            ix = row_end-1
            iy = yran[point]
            ang = aran[point]
            en0 = one_energy(arr,ix,iy,nmax)
            arr[ix,iy] += ang
            en1 = one_energy(arr,ix,iy,nmax)
            if en1<=en0:
                accept_local += 1
            else:
            # Now apply the Monte Carlo test - compare
            # exp( -(E_new - E_old) / T* ) >= rand(0,1)
                boltz = np.exp( -(en1 - en0) / Ts )

                if boltz >= rng.uniform(0.0,1.0):
                    accept_local += 1
                else:
                    arr[ix,iy] -= ang

    # A final synchronisation
    local_rows = arr[row_start:row_end].copy()
    gathered = comm.gather(local_rows, root=0) # List of rows
    if taskid == 0:
        arr[:,:] = np.vstack(gathered)
    comm.Bcast(arr, root=0)

    # Gather all local accept totals into one global total on MASTER node
    accept_global = comm.reduce(accept_local, op=MPI.SUM, root=0)
    if taskid == 0:
        return accept_global / (nmax * nmax)
    else:
        return None
#=======================================================================
def split_rows(nrows, numworkers, taskid):
    """
    Arguments:
        nrows (int): number of rows
        numworkers (int): number of workers
        taskid (int): task id of the MPI process.
    Description:
        Divide the rows among the workers
        Returns (start, end) for workers.
    Returns:
        start (int): index of the first row
        end (int): index of the last row
    """
    worker = taskid-1
    # Integer division and then pick up the remainder.
    base = nrows // numworkers
    remainder = nrows % numworkers

    # If the taskid is less than the remainder, it's going to get some extra rows.
    if worker < remainder:
        start = worker * (base + 1) # Have to base the start based on how many tasks got extra before it.
        end = start + (base + 1) # Change the end, based on the amount of rows remaining.

    else:
        start = remainder * (base + 1) + (worker - remainder) * base # No extra rows, but start and end point still
        end = start + base                                          # need to be based on extra rows on previous taskids

    return start, end
#=======================================================================
def main(program, nsteps, nmax, temp, pflag):
    """
    Arguments:
	  program (string) = the name of the program;
	  nsteps (int) = number of Monte Carlo steps (MCS) to perform;
      nmax (int) = side length of square lattice to simulate;
	  temp (float) = reduced temperature (range 0 to 2);
	  pflag (int) = a flag to control plotting.
    Description:
      This is the main function running the Lebwohl-Lasher simulation.
    Returns:
      NULL
    """
    # Locate taskid and task count
    comm = MPI.COMM_WORLD

    # Initalisations for MPI
    taskid = comm.Get_rank()
    numtasks = comm.Get_size()
    numworkers = numtasks - 1

    # Define the master task
    MASTER = 0

    # on the MASTER task:
    if taskid == MASTER:
        # Check if numworkers is within range - quit if not
        if numworkers < 1:
            print("ERROR: the number of workers must be greater than 1.")
            comm.Abort()

        # Create and initialise lattice
        lattice = initdat(nmax)
        # Plot initial frame of lattice
        plotdat(lattice,pflag,nmax)

        # Create arrays to store energy, acceptance ratio and order parameter
        energy = np.zeros(nsteps + 1, dtype=np.dtype)
        ratio = np.zeros(nsteps + 1, dtype=np.dtype)
        order = np.zeros(nsteps + 1, dtype=np.dtype)

        # Set initial values in arrays
        energy[0] = all_energy(lattice,nmax)
        ratio[0] = 0.5 # ideal value
        order[0] = get_order(lattice,nmax)

        # Begin doing and timing some MC steps.
        initial = MPI.Wtime()

        for i in range(1, numworkers+1):
            comm.send(temp, dest=i, tag=1)
            comm.send(nmax, dest=i, tag=2)
            comm.Send(lattice, dest=i, tag=3)

    # on the other tasks
    elif taskid != MASTER:
        # get the starting values and lattice from the master task
        temp = comm.recv(source=MASTER, tag=1)
        nmax = comm.recv(source=MASTER, tag=2)
        lattice = np.empty((nmax, nmax), dtype=np.double)
        comm.Recv([lattice, MPI.DOUBLE], source=MASTER, tag=3)

        # split the lattice amongst workers
        rows_start, rows_end = split_rows(nmax, numworkers, taskid)

    # This will be the only section run in parallel.
    for it in range(1, nsteps + 1):
        if taskid == MASTER:
            ratio[it] = MC_step(lattice, temp, nmax, taskid, comm, 0, 0) # Save the results of the master task, as outlined in each function
            energy[it] = all_energy_MPI(lattice, nmax, taskid, comm, 0, 0)
            order[it] = get_order_MPI(lattice, nmax, taskid, comm, 0, 0)
        else:
            MC_step(lattice, temp, nmax, taskid, comm, rows_start, rows_end)  # discard results from the worker tasks
            all_energy_MPI(lattice, nmax, taskid, comm, rows_start, rows_end)
            get_order_MPI(lattice, nmax, taskid, comm, rows_start, rows_end)

    # finalise timings and output on the master task
    if taskid == MASTER:
        final = MPI.Wtime()

        runtime = final-initial

        # Final outputs
        print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order[nsteps-1],runtime))
        # Plot final frame of lattice and generate output file
        savedat(lattice,nsteps,temp,runtime,ratio,energy,order,nmax)
        plotdat(lattice,pflag,nmax)


#=======================================================================
# Main part of program, getting command line arguments and calling
# main simulation function.
#
if __name__ == '__main__':
    if int(len(sys.argv)) == 5:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
#=======================================================================
