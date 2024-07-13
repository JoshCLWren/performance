import tensorflow as tf
import time
import numpy as np


def gpu_matrix_multiplication(size, iterations=5):
    # Define the dtype based on what your hardware can support
    dtype = tf.float32

    # Initialize random matrices
    A = tf.random.normal([size, size], dtype=dtype)
    B = tf.random.normal([size, size], dtype=dtype)

    # Warm-up run to ensure GPU is initialized
    _ = tf.linalg.matmul(A, B)

    # Time the matrix multiplication
    start_time = time.time()
    for _ in range(iterations):
        _ = tf.linalg.matmul(A, B)
    duration = time.time() - start_time
    avg_time = duration / iterations
    return avg_time


if __name__ == "__main__":
    matrix_size = 1000
    avg_time = gpu_matrix_multiplication(matrix_size)
    print(f"Average time to multiply two {matrix_size}x{matrix_size} matrices: {avg_time:.6f} seconds")
