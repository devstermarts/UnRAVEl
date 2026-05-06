#### UnRAVEl
#### https://github.com/devstermarts/UnRAVEl
#### Author: Martin Heinze
#### Year: 2026
#### ----------

import numpy as np


def d_gen_correlation_based(mean, covar, dp_length, rng):
    """Generates array based on covariance matrix."""
    eigvals = np.linalg.eigvals(covar)
    if np.any(eigvals < -1e-12):
        raise ValueError("Covariance matrix is not positive semi-definite.")
        exit()
    data_array = rng.multivariate_normal(mean, covar, size=(dp_length)).astype(
        np.float32
    )
    return data_array


def d_gen_normal(mean, std, dp_length, lat_dims, rng):
    """Generates array based on normal distribution."""
    data_array = rng.normal(mean, std, size=(dp_length, lat_dims)).astype(np.float32)
    return data_array


def d_gen_uniform(mean, std, dp_length, lat_dims, rng):
    """Generates array based on uniform distribution with 3 sigma coverage."""
    min = mean - 3 * std  # = ca. 97% coverage
    max = mean + 3 * std  # = ca. 97% coverage
    data_array = rng.uniform(min, max, size=(dp_length, lat_dims)).astype(np.float32)
    return data_array


def d_gen_iqr_uniform(lower_bound, upper_bound, dp_length, lat_dims, rng):
    """Generates array based on uniform distribution, uses IQR bounds calculation as source."""
    min = lower_bound
    max = upper_bound
    data_array = rng.uniform(min, max, size=(dp_length, lat_dims)).astype(np.float32)
    return data_array


def p_gen_fibo(data_array, num_loops):
    """Generates array based on fibonacci sequence."""
    fibo = np.empty((0, data_array.shape[1]))
    h = 1
    i = 1
    while h <= data_array.shape[0]:  # To do: clip length to a fibonacci number
        h = h + i
        i = h - i
        fibo = np.append(fibo, data_array[0:i,], axis=0).astype(np.float32)
    fibo = np.tile(fibo, (num_loops, 1))  # Repeat the pattern num_loops times.
    print(f"Total pattern length: {fibo.shape[0]}")
    return fibo


def p_gen_orale(data_array, num_loops, std):
    """Builds a longer sequence based on a given array and value variations in a 3:1 pattern."""
    orale = np.empty((0, data_array.shape[1]))
    rng = np.random.default_rng()
    data_array_edit = data_array - rng.uniform(-std, std, size=(1, data_array.shape[1]))
    seq1 = np.append(np.tile(data_array, (3, 1)), data_array_edit, axis=0)
    seq2 = seq1 + rng.uniform(-std, std, size=(1, data_array.shape[1]))
    f_seq1 = np.append(np.tile(seq1, (3, 1)), seq2, axis=0)
    f_seq2 = f_seq1 - rng.uniform(-std, std, size=(1, data_array.shape[1]))
    orale = np.append(np.tile(f_seq1, (3, 1)), f_seq2, axis=0).astype(np.float32)
    orale = np.tile(orale, (num_loops, 1))
    print(f"Total pattern length: {orale.shape[0]}")
    return orale


def p_gen_blender(data_array1, data_array2, num_loops):
    """Blends two arrays into each other by replacing single data points of one array with the corresponding data points of the other array.
    Repeat blended part 'num_loops' times."""
    if data_array1.shape != data_array2.shape:
        raise ValueError("Arrays must have the same shape.")
    blend = [data_array1.copy() for _ in range(num_loops)]
    for i in range(data_array1.shape[0]):
        for j in range(data_array1.shape[1]):
            data_array1[i, j] = data_array2[i, j]
            for _ in range(num_loops):
                blend.append(data_array1.copy())
    blend = np.concatenate(blend, axis=0).astype(np.float32)

    # Add these as output options later:
    blend_rev = blend[::-1]
    blend_mirror = blend[:, ::-1]
    blend_rev_mirror = blend[::-1, ::-1]

    print(f"Total pattern length: {blend.shape[0]}")
    return blend


def p_gen_swapper(data_array1, data_array2, num_loops):
    """Swap values of randomly picked data points in two arrays of the same size.
    Repeat altered array 'num_loops' times."""
    if data_array1.shape != data_array2.shape:
        raise ValueError("Arrays must have the same shape.")
    swap = np.empty((0, data_array1.shape[1]))
    rng = np.random.default_rng()
    for _ in range(data_array1.shape[0]):
        row = rng.integers(0, data_array1.shape[0])
        col = rng.integers(0, data_array1.shape[1])
        data_array1[row, col], data_array2[row, col] = (
            data_array2[row, col],
            data_array1[row, col],
        )
        for _ in range(num_loops):
            swap = np.append(swap, data_array1.copy(), axis=0).astype(np.float32)
    print(f"Total pattern length: {swap.shape[0]}")
    return swap


def p_gen_default(data_array, num_loops):
    """Repeat array 'num_loops' times."""
    data_array_rep = np.tile(data_array, (num_loops, 1))
    print(f"Total pattern length: {data_array_rep.shape[0]}")
    return data_array_rep
