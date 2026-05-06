#### UnRAVEl
#### https://github.com/devstermarts/UnRAVEl
#### Author: Martin Heinze
#### Year: 2026
#### ----------

import argparse
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from utils.gen_functions import (  # Various pattern generators
    d_gen_correlation_based,
    d_gen_iqr_uniform,
    d_gen_normal,
    d_gen_uniform,
    p_gen_blender,
    p_gen_default,
    p_gen_fibo,
    p_gen_orale,
    p_gen_swapper,
)
from utils.utils import eval_encodings, scan_dirs

# To do/ ideas:
#   - more patterns
#   - create dedicated modulation patterns (also for use in PD)


def arg_parser():
    """Parses arguments from CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to either a .npy file or a folder with .npy files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random generator.",
    )
    parser.add_argument(
        "--iqr",
        action="store_true",
        help="Use IQR to calculate value distribution.",
    )
    parser.add_argument(
        "--num_latents",
        type=int,
        default=10,
        help="Number of latents to be generated.",
    )
    parser.add_argument(
        "--num_loops",
        type=int,
        default=1,
        help="Number of loops/ repetitions to apply on latents.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=44100,
        help="Sample rate your model has been trained on.",
    )
    parser.add_argument(
        "--num_files",
        type=int,
        default=1,
        help="Number of files to be generated.",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default="normal",
        choices=["normal", "uniform", "iqr-uniform", "correlation-based"],
        help="Distribution of randomized values in latent files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="default",
        choices=["fibo", "orale", "blender", "swapper", "default"],
        help="Basic compositional patterns.",
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="If set, the script will generate embeddings instead of 'latent audio' files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./_generator-output",
        help="Path to where output files should be stored.",
    )
    return parser.parse_args()


def generate_data_array(
    args, mean, std, lower_bound, upper_bound, covar, lat_dims, rng
):
    if args.distribution == "correlation-based":
        return d_gen_correlation_based(mean, covar, args.num_latents, rng)
    elif args.distribution == "normal":
        return d_gen_normal(mean, std, args.num_latents, lat_dims, rng)
    elif args.distribution == "uniform":
        return d_gen_uniform(mean, std, args.num_latents, lat_dims, rng)
    elif args.distribution == "iqr-uniform" and args.iqr:
        return d_gen_iqr_uniform(
            lower_bound, upper_bound, args.num_latents, lat_dims, rng
        )
    else:
        print("Distribution type not supported")
        exit()


def generate(eval, args):
    """Generates an n channel 'latent audio' file in .wav format or an n channel .npy array with '--embeddings' flag."""
    mean = np.squeeze(eval["mean"], axis=0)
    std = np.squeeze(eval["std"], axis=0)
    if args.iqr:
        lower_bound = np.squeeze(eval["lower_bound"], axis=0)
        upper_bound = np.squeeze(eval["upper_bound"], axis=0)
        covar = None
    else:
        lower_bound = None
        upper_bound = None
        covar = np.squeeze(eval["covar"])

    lat_dims = mean.shape[0]  # To do: get no. of lat. dims. more elegantly?
    print(f"Number of lat. dims in generated file(s): {lat_dims}")

    pattern_functions = {
        "fibo": lambda data_array: p_gen_fibo(data_array, args.num_loops),
        "orale": lambda data_array: p_gen_orale(data_array, args.num_loops, std),
        "blender": lambda data_array1, data_array2: p_gen_blender(
            data_array1, data_array2, args.num_loops
        ),
        "swapper": lambda data_array1, data_array2: p_gen_swapper(
            data_array1, data_array2, args.num_loops
        ),
        "default": lambda data_array: p_gen_default(data_array, args.num_loops),
    }

    rng = np.random.default_rng(args.seed)

    for file in range(args.num_files):
        if args.pattern in ["blender", "swapper"]:
            data_array1 = generate_data_array(
                args, mean, std, lower_bound, upper_bound, covar, lat_dims, rng
            )
            data_array2 = generate_data_array(
                args, mean, std, lower_bound, upper_bound, covar, lat_dims, rng
            )
            data_array = pattern_functions[args.pattern](data_array1, data_array2)
        elif args.pattern in ["orale", "fibo"]:
            data_array = generate_data_array(
                args, mean, std, lower_bound, upper_bound, covar, lat_dims, rng
            )
            data_array = pattern_functions[args.pattern](data_array)
        elif args.pattern == "default":
            print(
                f"No specific pattern selected, generating random array of length {args.num_latents} and repeat {args.num_loops} times."
            )
            data_array = generate_data_array(
                args, mean, std, lower_bound, upper_bound, covar, lat_dims, rng
            )
            data_array = pattern_functions[args.pattern](data_array)
        else:
            print("Not a pattern. Exiting.")
            exit()

        file_name = f"file_{file}_plot--seed_{args.seed}--IQR-{args.iqr}--DIST-{args.distribution}--LATENTS-{args.num_latents}.png"
        file_path = os.path.join(args.output, timestamp, file_name)

        # Plot definitions ->
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title(f"Distribution plot for file {file} ({args.distribution})")
        plt.boxplot(data_array)
        plt.xlabel("Latent Dimension")
        plt.ylabel("Values")
        plt.xticks(range(1, lat_dims + 1))
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.title(f"Time series for file {file} ({args.distribution})")
        plt.plot(data_array)
        plt.xlabel("Samples")
        plt.ylabel("Values")
        plt.grid(True)

        plt.savefig(file_path)
        print(f"Saved value distribution plot to '{file_path}'")

        # File output ->
        new_file = f"file_{file}_signal--seed_{args.seed}--IQR-{args.iqr}--DIST-{args.distribution}--PATTERN-{args.pattern}--LATENTS-{args.num_latents}.wav"
        new_file_path = os.path.join(args.output, timestamp, new_file)
        latent_sample_rate = args.sample_rate // 2048
        sf.write(
            new_file_path,
            data_array,
            latent_sample_rate,
            subtype="FLOAT",  # Generate a .wav file with 32-bit floating-point samples
        )
        print(f"Saved file to '{new_file_path}'")
        if args.embeddings:
            data_array = np.expand_dims(data_array.T, axis=0)
            new_embeddings_file = f"file_{file}_embeddings--seed_{args.seed}--IQR-{args.iqr}--DIST-{args.distribution}--PATTERN-{args.pattern}--LATENTS-{args.num_latents}.npy"
            new_embeddings_file_path = os.path.join(
                args.output, timestamp, new_embeddings_file
            )
            np.save(new_embeddings_file_path, data_array)
            print(f"Saved embeddings to '{new_embeddings_file_path}'")


if __name__ == "__main__":
    args = arg_parser()

    if args.num_latents < 1:
        raise ValueError("Latents must be greater than 0.")
    if args.seed < 1:
        raise ValueError("Seed must be greater than 0.")
    if args.num_loops < 1:
        raise ValueError("Number of loops must be greater than 0.")
    if args.sample_rate < 1:
        raise ValueError("Sample rate must be greater than 0.")
    if args.num_files < 1:
        raise ValueError("Number of files must be greater than 0.")
    if args.iqr and args.distribution in ["correlation-based", "uniform"]:
        print(
            "'--iqr' can only be used in conjunction with 'normal' and 'iqr-uniform' distributions."
        )
        exit()
    if not args.iqr and args.distribution == "iqr-uniform":
        print("'iqr-uniform' distribution only in conjunction with '--iqr'.")
        exit()
    if args.pattern == "fibo" and args.num_latents < 55:
        print(
            f"Latent length should be at least 55 for fibonacci pattern. Got {args.num_latents}"
        )
        exit()

    if os.path.isdir(args.input):
        files = scan_dirs(args.input, ".npy")  # Scan input directory for .npy files
    elif os.path.isfile(args.input):
        files = [args.input]  # Use input file
    else:
        raise ValueError(f"Input {args.input} is neither a file nor a directory.")

    timestamp = str(datetime.datetime.now().timestamp())

    os.makedirs(os.path.join(args.output, timestamp), exist_ok=True)

    # Evaluate encodings .npy files ->
    eval = eval_encodings(files, args.iqr, args.output, timestamp)
    print(f"Evaluation done. Generating {args.num_files} file(s) now.")

    # Generate synthetic data ->
    generate(eval, args)
    print(
        f"----------"
        f"\nDone. {args.num_files} file(s) saved to '{args.output}'"
        f"\nPattern '{args.pattern}' used."
    )
    if args.pattern != "fibo":
        print(
            f"Base array length: {args.num_latents} data points"
            f"\nQuantized tempo: {(60 * (args.sample_rate / 2048) / args.num_latents):.2f} BPM"
        )
