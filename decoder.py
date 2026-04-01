#### UnRAVEl
#### https://github.com/devstermarts/UnRAVEl
#### Author: Martin Heinze
#### Year: 2026
#### ----------

import argparse
import hashlib
import os

import numpy as np
import soundfile as sf
import torch

from utils.model_inspect import eval_sample_rate
from utils.utils import device, scan_dirs

# To do/ ideas:
#   - add support for RAVE > v2.2.2
#   - decoder determinism check


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to either a .npy file or a folder with .npy files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model (same as used with encoder.py)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./_decoder-output",
        help="Path to store decoded files to.",
    )
    return parser.parse_args()


def decode_latents_to_audio(file, args):
    """Decodes latent embeddings into audio files using a model."""

    os.makedirs(args.output, exist_ok=True)

    model = torch.jit.load(args.model).to(device)

    with torch.no_grad():
        sample_rate = eval_sample_rate(model)
        if sample_rate == "Unknown":
            sample_rate = 44100
            print("Could not determine sample rate from model. Using 44100 Hz.")
        else:
            print(f"Retrieved sample rate from model: {sample_rate} Hz")

        z = np.load(file)
        z = torch.from_numpy(z).to(device)
        x = model.decode(z).detach().cpu().numpy()
        x = np.squeeze(x, axis=0).T

        # Hashing path for files w/ same name ->
        hash_path = hashlib.md5(file.encode()).hexdigest()
        file_name = f"decoded--{os.path.splitext(os.path.basename(file))[0].strip().replace(' ', '_')}--{hash_path}.wav"
        file_path = os.path.join(args.output, file_name)

        if os.path.exists(file_path):
            print("File already exists. Overwriting...")

        sf.write(
            file_path,
            x,
            sample_rate,
        )
        print(
            f"Decoded '{os.path.basename(file)}' from array with shape '{z.shape}' to audio and stored to '{file_path}'"
        )


if __name__ == "__main__":
    args = arg_parser()

    device = device()

    if os.path.isdir(args.input):
        # Scan input directory for .npy files ->
        files = scan_dirs(args.input, ".npy")
        for file in files:
            decode_latents_to_audio(file, args)
    elif os.path.isfile(args.input):
        # Use single input file ->
        decode_latents_to_audio(args.input, args)
    else:
        raise ValueError(f"Input {args.input} is neither a .npy file nor a directory.")
