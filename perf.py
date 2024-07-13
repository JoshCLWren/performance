import json
import platform
import socket
import time

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from save_results import save_results


def gather_system_info():
    info = {
        "processor": platform.processor(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_logical_processors": psutil.cpu_count(logical=True),
        "ram": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = torch.cuda.get_device_capability(0)
    return info


def cpu_matrix_multiplication(size, iterations=5):
    times = []
    for _ in range(iterations):
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)
        start_time = time.time()
        C = np.dot(A, B)
        end_time = time.time()
        times.append(end_time - start_time)
    return sum(times) / len(times)


def gpu_matrix_multiplication(size, iterations=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    times = []
    A = torch.rand(size, size, device=device)
    B = torch.rand(size, size, device=device)
    for _ in range(10):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()
    for _ in range(iterations):
        start_time = time.time()
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    return sum(times) / len(times)


def memory_intensive_operations(size, iterations=5):
    times = []
    for _ in range(iterations):
        large_array = np.random.rand(size, size)
        start_time = time.time()
        modified_array = large_array * 3.14159
        np.sum(modified_array)
        end_time = time.time()
        times.append(end_time - start_time)
    return sum(times) / len(times)


def kmeans_clustering(n_samples=100000, n_features=10, n_clusters=5, iterations=5):
    data, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters)
    times = []
    for _ in range(iterations):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        start_time = time.time()
        kmeans.fit(data)
        end_time = time.time()
        times.append(end_time - start_time)
    return sum(times) / len(times)


def pandas_data_manipulation(n_rows=1000000, iterations=5):
    times = []
    for _ in range(iterations):
        df = pd.DataFrame({"A": np.random.randn(n_rows), "B": np.random.rand(n_rows)})
        start_time = time.time()
        df["C"] = df["A"] + df["B"]
        result = df.groupby(pd.cut(df["C"], bins=10)).size()
        end_time = time.time()
        times.append(end_time - start_time)
    return sum(times) / len(times)


if __name__ == "__main__":
    system_name = socket.gethostname()
    matrix_size = 2000
    array_size = 20000
    iterations = 5

    # Measure performance
    cpu_duration = cpu_matrix_multiplication(matrix_size, iterations)
    gpu_duration = gpu_matrix_multiplication(matrix_size, iterations)
    memory_duration = memory_intensive_operations(array_size, iterations)
    clustering_time = kmeans_clustering()
    manipulation_time = pandas_data_manipulation()

    # Print results
    print(f"Average CPU Matrix Multiplication Time: {cpu_duration:.6f} seconds")
    print(f"Average GPU Matrix Multiplication Time: {gpu_duration:.6f} seconds")
    print(f"Average Memory Intensive Operation Time: {memory_duration:.6f} seconds")
    print(f"Average K-Means Clustering Time: {clustering_time:.6f} seconds")
    print(f"Average Pandas Data Manipulation Time: {manipulation_time:.6f} seconds")

    # System info and results
    system_info = gather_system_info()
    results = {
        system_name: {
            "system_info": system_info,
            "performance": {
                "cpu_matrix_multiplication": cpu_duration,
                "gpu_matrix_multiplication": gpu_duration,
                "memory_operations": memory_duration,
                "kmeans_clustering": clustering_time,
                "pandas_data_manipulation": manipulation_time,
            },
        }
    }

    save_results(results)
    print("Results saved to system_performance.json")
