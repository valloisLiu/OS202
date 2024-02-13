from mpi4py import MPI
import numpy as np

# 初始化MPI环境
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# 定义问题的维度
dim = 120

# 根进程初始化矩阵和向量
if rank == 0:
    A = np.array([[(i+j) % dim + 1. for i in range(dim)] for j in range(dim)])
    u = np.array([i + 1. for i in range(dim)])
else:
    A = np.empty((dim, dim))  # 分配空间用于接收广播的子矩阵
    u = np.empty(dim)  # 分配空间用于接收广播的向量

# 根进程广播向量u
u = comm.bcast(u, root=0)

# 分发矩阵的列
cols_per_proc = dim // size + (dim % size > rank)
start_col = rank * cols_per_proc
end_col = start_col + cols_per_proc

if rank == 0:
    # 根进程发送其他进程的子矩阵部分
    for i in range(1, size):
        start_col_i = i * cols_per_proc
        end_col_i = start_col_i + cols_per_proc
        comm.Send(np.ascontiguousarray(A[:, start_col_i:end_col_i]), dest=i)
    # 根进程保留第一部分
    A_local = A[:, start_col:end_col]
else:
    # 其他进程接收它们的子矩阵
    A_local = np.empty((dim, end_col - start_col))
    comm.Recv(A_local, source=0)

# 局部计算矩阵-向量乘积
v_local = np.dot(A_local, u[start_col:end_col])

# 初始化全局结果向量
if rank == 0:
    v = np.zeros(dim)
else:
    v = None

# 收集局部结果
v_gathered = comm.gather(v_local, root=0)

# 根进程拼接结果
if rank == 0:
    v = np.concatenate(v_gathered)

# 根进程打印结果
if rank == 0:
    print(f"v = {v}")
