from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

dim = 120
rows_per_process = dim // size

if rank < dim % size:
    rows_per_process += 1

if rank == 0:
    A = np.array([[(i + j) % dim + 1 for i in range(dim)] for j in range(dim)])
    u = np.array([i + 1 for i in range(dim)])
else:
    A = np.empty((rows_per_process, dim))
    u = np.empty(dim)

comm.Bcast(u, root=0)

if rank == 0:
    Asplit = np.array_split(A, size, axis=0)
else:
    Asplit = None

local_A = comm.scatter(Asplit, root=0)
local_v = np.dot(local_A, u)

v = None
if rank == 0:
    v = np.empty(dim)

v = comm.gather(local_v, root=0)

if rank == 0:
    v = np.concatenate(v)

if rank == 0:
    print(f"v = {v}")
