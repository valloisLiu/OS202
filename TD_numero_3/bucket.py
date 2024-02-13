import numpy as np
import multiprocessing

def parallel_bucket_sort(data, bucket_size):
    # Determine the interval for each bucket
    min_value = min(data)
    max_value = max(data)
    interval = (max_value - min_value) / bucket_size

    # Create buckets and distribute data
    buckets = [[] for _ in range(bucket_size)]
    for number in data:
        index = int((number - min_value) / interval)
        if index == bucket_size:
            index -= 1
        buckets[index].append(number)

    # Function to sort each bucket
    def sort_bucket(bucket_data):
        return sorted(bucket_data)

    # Use a pool of workers to sort each bucket in parallel
    with multiprocessing.Pool(processes=bucket_size) as pool:
        sorted_buckets = pool.map(sort_bucket, buckets)

    # Merge the sorted buckets
    sorted_data = [element for bucket in sorted_buckets for element in bucket]
    return sorted_data

# Generate random data
np.random.seed(0)  # For reproducibility
random_data = np.random.rand(1000)

# Number of buckets (can be equal to the number of available processors)
num_buckets = multiprocessing.cpu_count()

# Perform parallel bucket sort
sorted_data = parallel_bucket_sort(random_data, num_buckets)

print("Sorted Data:", sorted_data)
