#### UnRAVEl
#### https://github.com/devstermarts/UnRAVEl
#### Author: Martin Heinze
#### Year: 2026
#### ----------

import argparse
import hashlib
import os

import numpy as np
import torch
import torchaudio

from utils.model_inspect import eval_sample_rate
from utils.utils import device, scan_dirs

# To do/ ideas:
#   - add support for RAVE > v2.2.2
#   - smoke test other encoder architectures


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to either an audio file or a folder with audio files to encode through the model model.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=[".wav", ".mp3", ".ogg", ".aac", ".opus", ".aif", ".aiff", ".flac"],
        nargs="+",
        help="File extension to scan for, e.g. '.wav + .opus'",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./_encoder-output",
        help="Path to store .npy files to.",
    )
    return parser.parse_args()


def encode_audio_to_latents(file, args):
    """Encodes audio files into latent embeddings using a model model."""

    os.makedirs(args.output, exist_ok=True)

    model = torch.jit.load(args.model).to(device)

    with torch.no_grad():
        sample_rate = eval_sample_rate(model)
        if sample_rate == "Unknown":
            sample_rate = 44100
            print("Could not determine sample rate from model. Using 44100 Hz.")
        else:
            print(f"Retrieved sample rate from model: {sample_rate} Hz")

        y, sr = torchaudio.load(file)
        if sr != sample_rate:
            transform = torchaudio.transforms.Resample(sr, sample_rate)
            y = transform(y)  # Not tested yet, requires torch <= 2.8.*
            print(f"Resampling {file} to {sample_rate} Hz.")
        x = y.to(device)
        if x.shape[0] > 1:
            x = torch.mean(
                x, dim=0, keepdim=True
            )  # Calculating mean across channels -> mono
        x = x.reshape(1, 1, -1)
        z = model.encode(x)
        z = z.detach().cpu().numpy()

        # Hashing path for files w/ same name ->
        hash_path = hashlib.md5(file.encode()).hexdigest()
        file_name = f"encoded--{os.path.splitext(os.path.basename(file))[0].strip().replace(' ', '_')}--{hash_path}.npy"
        file_path = os.path.join(args.output, file_name)

        if os.path.exists(file_path):
            print("File already exists. Overwriting...")

        np.save(file_path, z)
        print(
            f"Encoded {len(y[0])} samples from '{os.path.basename(file)}' into latent of shape '{z.shape}' and stored to '{file_path}'"
        )
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.memory.empty_cache()
        else:
            pass


if __name__ == "__main__":
    args = arg_parser()

    device = device()

    if os.path.isdir(args.input):
        # Scan input directory for audio files ->
        files = scan_dirs(args.input, args.extensions)
        for file in files:
            encode_audio_to_latents(file, args)
    elif (
        os.path.isfile(args.input)
        and os.path.splitext(args.input)[1] in args.extensions
    ):
        # Use single input file ->
        file = args.input
        encode_audio_to_latents(file, args)
    else:
        raise ValueError(
            f"Input {args.input} is neither an audio file nor a directory."
        )
