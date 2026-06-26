#### RAVE-Latent-Composition
#### https://github.com/devstermarts/RAVE-Latent-Composition
#### Author: Martin Heinze
#### Year: 2026
#### ----------

import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def scan_dirs(input, extensions):
    """Scans a folder and its subfolders for files of a specific type and outputs them as list."""

    all_files = []
    for root, dirs, files in os.walk(input):
        for name in files:
            _, ext = os.path.splitext(name)
            if name.startswith("."):
                continue  # Skip hidden files like .DS_Store. More elegant solution later?
            if ext.lower() in extensions:
                all_files.append(os.path.join(root, name))

    if len(all_files) != 0:
        print(f"\n{len(all_files)} files found in '{input}' and subdirectories.")
        return all_files
    else:
        print(
            f"\nNo files with extension(s) '{extensions}' found in '{input}' and subdirectories."
        )
        return []


def device():
    """Checks for accelerators."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Found device '{device}'")
    return device


def eval_encodings(files, iqr, output, timestamp):
    """Evaluates .npy files for value distribution and correlation.
    Plots distribution to file.

    """

    all_data_list = []

    # Todo: check for homogeneity of all files -> shape must be the same for the first 2 dims.

    for file in files:
        array = np.load(file)
        all_data_list.append(array)

    all_data = np.concatenate(all_data_list, axis=2)

    lat_dims = all_data.shape[1]
    print(f"Number of lat. dims in source files: {lat_dims}\n")

    val_dict = {}

    if iqr:
        all_data_iqr1 = np.percentile(all_data, 25, axis=2)
        all_data_iqr3 = np.percentile(all_data, 75, axis=2)
        all_data_iqr = all_data_iqr3 - all_data_iqr1
        all_data_iqr_lower_bound = all_data_iqr1 - 1.5 * all_data_iqr
        all_data_iqr_upper_bound = all_data_iqr3 + 1.5 * all_data_iqr
        all_data_iqr_median = (all_data_iqr1 + all_data_iqr3) / 2
        all_data_iqr_deviation = (
            all_data_iqr / 1.35
        )  # Only valid on normal distributed data!

        val_dict = {
            "mean": all_data_iqr_median,
            "std": all_data_iqr_deviation,
            "lower_bound": all_data_iqr_lower_bound,
            "upper_bound": all_data_iqr_upper_bound,
        }
        print(
            f"\n===================="
            f"\nData analysis:"
            f"\n--------------------"
            f"\nMedian (as 'mean')):\n{all_data_iqr_median}"
            f"\nDeviation (as 'std'):\n{all_data_iqr_deviation}"
            f"\nLower bound:\n{all_data_iqr_lower_bound}"
            f"\nUpper bound:\n{all_data_iqr_upper_bound}"
            f"\n--------------------"
        )

    else:
        all_data_mean = np.mean(all_data, axis=2)
        all_data_std = np.std(all_data, axis=2)
        all_data_corr_mat = np.corrcoef(np.squeeze(all_data, axis=0))
        all_data_covar_mat = np.outer(all_data_std, all_data_std) * all_data_corr_mat
        val_dict = {
            "mean": all_data_mean,
            "std": all_data_std,
            "corr": all_data_corr_mat,
            "covar": all_data_covar_mat,
        }
        print(
            f"\n===================="
            f"\nData analysis:"
            f"\n--------------------"
            f"\nMean:\n{all_data_mean}"
            f"\nStandard deviation:\n{all_data_std}"
            f"\nCorrelation matrix:\n{all_data_corr_mat}"
            f"\nCovariance matrix:\n{all_data_covar_mat}"
            f"\n--------------------"
        )

    all_data_show = np.squeeze(all_data, axis=0).T

    file_name = "distribution_analysis.png"
    file_path = os.path.join(output, timestamp, file_name)

    plt.figure(figsize=(12, 6))
    plt.tight_layout()

    plt.subplot(1, 2, 1)
    plt.title("Distribution violin plot")
    plt.violinplot(all_data_show, showmedians=True, showmeans=True)
    plt.xlabel("Latent Dimension")
    plt.ylabel("Values")
    plt.xticks(range(1, lat_dims + 1))
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.title("Distribution boxplot w/o extreme outliers")
    plt.boxplot(all_data_show, showfliers=False, showmeans=True)
    plt.xlabel("Latent Dimension")
    plt.ylabel("Values")
    plt.xticks(range(1, lat_dims + 1))
    plt.grid(True)

    plt.savefig(file_path)
    print(f"Saved value distribution plots to '{file_path}'")

    return val_dict
