from mpi4py import MPI
import numpy as np
import time
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

lower_bound = 1
upper_bound = 100
arr_size = 100
begin = time.time()
if rank == 0:
    data = np.random.randint(lower_bound, upper_bound, arr_size).astype('d')  
else:
    data = None  

if rank == 0:
    cnt = [arr_size // size for _ in range(size)]
    for i in range(arr_size % size):
        cnt[i] += 1
    displs = [sum(cnt[:i]) for i in range(size)]
else:
    cnt = None
    displs = None

cnt = comm.bcast(cnt, root=0)
displs = comm.bcast(displs, root=0)

buf = np.zeros(cnt[rank], dtype='d')

comm.Scatterv([data, cnt, displs, MPI.DOUBLE], buf, root=0)
buf.sort()
sorted_data = None
if rank == 0:
    sorted_data = np.zeros(arr_size, dtype='d')  

comm.Gatherv(buf, [sorted_data, cnt, displs, MPI.DOUBLE], root=0)
fin = time.time()
if rank == 0:
    print("Sorted data:", sorted_data)
print("We use:",-begin+fin,"second to sort")
