#### UnRAVEl
#### https://github.com/devstermarts/UnRAVEl
#### Author: Martin Heinze
#### Year: 2026
#### ----------


def eval_enc(model):
    """Check for encoder type name, version and shape."""
    enc_type = "Unknown"
    enc_type_version = "n/a"
    encoder_attr = [
        ".encoder.original_name",  # RAVE V2/V3
        "._rave.encoder.original_name",  # legacy RAVE V1
        "['model_config']['model']['encoder']['type']",  # SAT checkpoints only
    ]
    for attr in encoder_attr:
        try:
            enc_type = eval(f"model{attr}")
            enc_type_version = eval(f"model.encoder{attr}")  # Only works with RAVE
            break
        except AttributeError, KeyError, NotImplementedError:
            continue

    if enc_type == "Unknown" and enc_type_version == "n/a":
        print("Unknown encoder type.")

    try:
        enc = model.encode_params.cpu().numpy()
    except:
        print("No encoder parameters found in model.")
        enc = "Unknown"

    return enc, enc_type, enc_type_version


def eval_dec(model):
    """Check for decoder type name and shape."""
    dec_type = "Unknown"
    decoder_attr = [
        ".decoder.original_name",  # RAVE V2/V3
        "._rave.decoder.original_name",  # legacy RAVE V1
        "['model_config']['model']['decoder']['type']",  # SAT checkpoints only
    ]
    for attr in decoder_attr:
        try:
            dec_type = eval(f"model{attr}")
            break
        except AttributeError, KeyError, NotImplementedError:
            continue
    if dec_type == "Unknown":
        print("Unknown decoder type.")

    try:
        dec = model.decode_params.cpu().numpy()
    except:
        print("No decoder parameters found in model.")
        dec = "Unknown"

    return dec, dec_type


def eval_sample_rate(model):
    """Check for model sample rate."""
    sample_rate = "Unknown"
    sample_rate_attr = [
        ".sr",  # RAVE V2
        ".sampling_rate",  # RAVE V3 / forks
        "._rave.sampling_rate",  # legacy RAVE V1
        "['model_config']['sample_rate']",  # SAT checkpoints only
    ]
    for attr in sample_rate_attr:
        try:
            sample_rate = eval(f"model{attr}")
            break
        except AttributeError, KeyError, NotImplementedError:
            continue
    if sample_rate == "Unknown":
        print("No sample rate found in model.")
    return sample_rate


def eval_lat_size(model):
    """Check for model latent size."""
    lat_size = "Unknown"
    lat_size_attr = [
        ".latent_size",  # RAVE V2
        "._rave.latent_size",  # legacy RAVE V1
        "['model_config']['model']['latent_dim']",  # SAT checkpoints only
    ]  # add "lat_dims" ?
    for attr in lat_size_attr:
        try:
            lat_size = eval(f"model{attr}")
            break
        except AttributeError, KeyError, NotImplementedError:
            continue
    if lat_size == "Unknown":
        print("No latent size found in model.")
    return lat_size


def eval_stereo(model):
    """Check if model is pseudo stereo (RAVE <=v2.2.2 instantiates two decoders)."""
    try:
        is_stereo = model.stereo
    except:
        print("No 'stereo' information found in model.")
        is_stereo = "Unknown"
    return is_stereo


def eval_channels(model):
    """Check number of channels."""
    pass  # To do
