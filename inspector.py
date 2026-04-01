#### UnRAVEl
#### https://github.com/devstermarts/UnRAVEl
#### Author: Martin Heinze
#### Year: 2026
#### ----------

import argparse
import csv
import hashlib
import inspect
import json
import os

import torch

from utils.model_inspect import (
    eval_channels,  # TBD.
    eval_dec,
    eval_enc,
    eval_lat_size,
    eval_sample_rate,
    eval_stereo,
)
from utils.utils import device, scan_dirs

# To do/ ideas:
#   - add support for RAVE > v2.2.2
#   - add support for num_channels in SAT checkpoints: "['model_config']['audio_channels']",
#   - evaluate mean, std state_dict entries in RAVE models (-> for use in generator.py w/o encoded embeddings)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a model or a folder containing models.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=[".ts", ".ckpt"],
        nargs="+",
        help="Model file extension to scan for.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Store model info to .csv file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./_inspector-output",
        required=False,
        help="Path to the folder where the state_dict .csv files should be stored.",
    )
    parser.add_argument(
        "--dict",
        action="store_true",
        help="Return model state_dict to CLI and enter detailed request mode.",
    )
    parser.add_argument(
        "--attr",
        action="store_true",
        help="Return model attributes to CLI and enter detailed request mode.",
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Return model config to CLI (only SAT checkpoints).",
    )
    return parser.parse_args()


def model_eval(file, args):
    """Evaluates model files."""

    with torch.no_grad():
        print(f"\n====================\nInspecting model '{os.path.basename(file)}'")
        try:
            model = torch.jit.load(file).to(device)  # load .ts model to device
            state_dict = model.state_dict()
        except:
            model = torch.load(
                file, map_location=torch.device(device)
            )  # load .ckpt model to device
            state_dict = model["state_dict"]

        enc, enc_type, enc_type_version = eval_enc(model)
        dec, dec_type = eval_dec(model)

        sample_rate = eval_sample_rate(model)
        lat_size = eval_lat_size(model)

        if lat_size != dec[0] or lat_size != enc[2]:
            print("Latent dimensions to encoder/decoder shape mismatch (or unknown).")

        is_stereo = eval_stereo(model)

        print(
            f"\nModel facts:"
            f"\n--------------------"
            f"\nName: {os.path.splitext(os.path.basename(file))[0]}"
            f"\nEncoder type: {enc_type} ({enc_type_version})"
            f"\nEncoder shape: {enc}"
            f"\nDecoder type: {dec_type}"
            f"\nDecoder shape: {dec}"
            f"\nLatent dimensions: {lat_size}"
            f"\nStereo: {is_stereo}"
            f"\nSample rate: {sample_rate}"
            f"\n--------------------"
        )

        if args.save:
            model_facts = {
                "name": os.path.splitext(os.path.basename(file))[0],
                "encoder_type": enc_type,
                "encoder_type_version": enc_type_version,
                "encoder_shape": enc,
                "decoder_type": dec_type,
                "decoder_shape": dec,
                "latent_dimensions": lat_size,
                "stereo": is_stereo,
                "sample_rate": sample_rate,
            }

            os.makedirs(args.output, exist_ok=True)
            # Hashing path for files w/ same name.
            hash_path = hashlib.md5(file.encode()).hexdigest()

            # Save state_dict as csv
            file_name = f"state_dict--{os.path.splitext(os.path.basename(file))[0].strip().replace(' ', '_')}--{hash_path}.csv"
            file_path = os.path.join(args.output, file_name)

            if os.path.exists(file_path):
                print("File already exists. Overwriting...")

            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["key", "shape"])
                for key, value in model_facts.items():
                    writer.writerow([key, value])
                writer.writerow(["END MODEL FACTS", ""])
                writer.writerow(["", ""])
                writer.writerow(["BEGIN STATE_DICT", ""])
                for key, value in state_dict.items():
                    writer.writerow([key, list(value.shape)])

            print(f"'state_dict' saved to {file_path}")

        if args.dict:  # Enter detailed request mode
            print("\nModel parameters:\n--------------------")

            for key, _ in state_dict.items():
                print(key)

            while True:
                print("\n====================")
                user_input = input(
                    f"Enter parameter name for model '{os.path.basename(file)}' (or 'exit' to quit): "
                )

                if user_input.lower() == "exit":
                    print("Exiting...")
                    break

                try:
                    print(f"\n{state_dict[user_input]}")
                except Exception as e:
                    print(f"\nError: {e} not found in model parameters.")

        if args.attr:  # Enter detailed request mode
            print("\nModel attributes:\n--------------------")

            for attr in dir(model):
                print(attr)

            while True:
                print("\n====================")
                user_input = input(
                    f"Enter attribute for '{os.path.basename(file)}' (or 'exit' to quit): "
                )

                if user_input.lower() == "exit":
                    print("Exiting...")
                    break

                try:
                    req_attr = getattr(model, user_input)
                    if callable(req_attr):
                        print(f"\n'{user_input}' is a method")
                        try:
                            print(f"\n{inspect.signature(req_attr)}")
                        except:
                            print("Signature not available")
                        q_input = input(f"\nTry to call method '{user_input}'? (y): ")
                        if q_input.lower() == "y":
                            print(f"{req_attr()}")

                    else:
                        print(f"\n'{user_input}' is a property or data attribute")
                        print(f"\n{req_attr}")
                        print(f"\n{type(req_attr)}")
                except AttributeError:
                    print(f"\nError: '{user_input}' not found in model attributes.")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")

        if args.config:
            try:
                model_config = json.dumps(model["model_config"], indent=4)
            except:
                print(f"No model config found in model {file}.")
            print(f"\nFull model config:\n--------------------\n{model_config}")


if __name__ == "__main__":
    args = arg_parser()

    device = device()

    if os.path.isdir(args.input):
        # Scan input directory for model files ->
        files = scan_dirs(args.input, args.extensions)
        for file in files:
            model_eval(file, args)
    elif (
        os.path.isfile(args.input)
        and os.path.splitext(args.input)[1] in args.extensions
    ):
        # Use single input file ->
        model_eval(args.input, args)
    else:
        raise ValueError(f"Input {args.input} is neither a model file nor a directory.")
