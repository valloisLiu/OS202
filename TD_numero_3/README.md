# MPI Parallel Bucket Sort

This repository contains a Python implementation of the bucket sort algorithm using MPI for parallel processing. The core idea is to have multiple processes work together to sort a large array of numbers. 

## Algorithm

The sorting process is as follows:

1. Process 0 generates an array of random numbers within a specified range.
2. This array is then divided into roughly equal parts and sent to all participating processes.
3. Each process sorts its received chunk of the array.
4. Finally, Process 0 collects all the sorted chunks from the other processes and concatenates them to form the sorted array.

## Running the Code

To execute the sorting, use the following command:

mpiexec -np 2 python bucket_sort.py
