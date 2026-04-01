
# UnRAVEl

Speculative composition in latent space using RAVE models.
---
RAVE models encode data from the audio domain into highly compressed latent representations. The scripts in this repository build on the idea, that based on statistical information retrieved from these encodings, a speculative compositional practice can be established inside the latent space of the models. It is derived from the improvisation tactics empirically proven in [Latent Jamming](https://martstil.de/concepts/latent-jamming). 

The process consists of three phases:
* __audio data encoding:__ 1-n audio files are encoded into arrays in the shape of a model's latent space.  
* __generation of synthetic data:__ 1-n encodings are evaluated for their data distribution. Based on the results, arrays of synthetic data are generated which are used to populate preset patterns and apply alterations to these patterns.
* __decoding of synthetic data into audio data:__ 1-n generated/ synthetic data arrays are being decoded and up sampled back to the audio domain using the same model as in the encoding process.

Author: Martin Heinze | [`marts~`](https://martstil.de). Year: 2026

---

## Install

Create and activate a new conda environment e.g.:

```bash
conda create -n unravel python=3.14
conda activate unravel
```

Clone this repository, run pip install on requirements.txt

```bash
git clone https://github.com/devstermarts/UnRAVEl.git
cd UnRAVEl
pip install -r requirements.txt
```

FFMPEG is required. Install with: 

```bash
conda install ffmpeg -y
```


## Encoding to latent embeddings

__encoder.py__ takes one or many audio files to encode to .npy arrays using a .ts model. For best results, data should be similar to (or the same as) what the model has seen during training since we want to generate synthetic data in a value range the decoder can handle well.

```bash
python encoder.py \
--input /path/to/your/audio/files \ # or path to a single audio file
--extensions .opus \ # pick audio file extension
--model /path/to/your/pretrained/model.ts \
--output # default is './_encoder-output'
```

## Generation of synthetic data

__generator.py__ analyzes one or many .npy arrays for value distribution and correlation and creates arrays of synthetic data to populate patterns (see below), which can be exported as a 'latent audio' file - basically a multi channel (= latent dimensions), double precision (= for values outside -1/+1 boundary) .wav file at e.g. 21Hz resolution (if data source was 44.1KHz) for use with [ch4ns0n/ch8ns0n components inside Pure Data](https://github.com/devstermarts/pd-latent-jamming). Alternatively, the export writes a .npy file that can be transformed using the __decoder.py__ script. 

```bash
python generator.py \
--input /path/to/your/numpy/files \ # or path to a single .npy file
--seed 123 \ # seed for random generator, required for reproducible patterns
--iqr \ # use interquartile range to calculate value distribution
--num_latents 21 \ # length of synthetic latent pattern
--num_loops 4 \ # number of loops/repetitions used in pattern logic
--sample_rate 44100 \ # must be sample rate the model has been trained on
--num_files 3 \ # number of files to be generated, default is 1
--distribution correlation-based \ # distribution method to sample from
--pattern swapper \ # pattern generator, select from multiple patterns (or add your own)
--embeddings \ # create .npy files instead of 'latent audio' files.
--output /path/to/generated/files # default is './_generator-output'
```

### Distribution

Encodings are analyzed for their mean, standard deviation (both per latent dimension) and correlation between latent dimensions. When `--iqr` flag is used, interquartile range lower and upper bound is determined along with median and deviation instead. When generating synthetic data, these values inform distribution types (normal, uniform, iqr-uniform, correlation-based). Synthetic data in normal distribution (with and without `--iqr`) as well as correlation-based distribution should be the most truthful in relation to the model's capabilities.

### Tempo quantizing

Latent embeddings created with RAVE are highly compressed representations of audio domain data. A sample rate of 44.1KHz corresponds to 21 data points times the number of latent dimensions in the model's configuration. The audio domain resolution is high enough to be more or less irrelevant for tempo considerations, however, with the low resolution in latent space, limitations to achievable tempi are expressed by:  

`60 * (model sample rate / model compression) / latent data points`

For example:  

`60 * (44100 / 2048) / 11 = 117.45 BPM`

This leads to the following quantized tempi (in 4ths, 8ths for double time) achievable by looping `k` amount of latent data points (in the script: `--num_latents` and `--num_loops`): 

| k | Tempo BPM | Double time |
| ---: | ---: | ---: |
| 1  | 1291.99  | - |
| 2  | 645.99   | 1291.98 |
| 3  | 430.66   | - |
| 4  | 322.99   | 645.98 |
| 5  | 258.39   | - |
| 6  | 215.33   | 430.66 |
| 7  | 184.57   | - |
| 8  | 161.49   | 322.98 |
| 9  | 143.55   | - |
| 10 | 129.19   | 258.38 |
| 11 | 117.45   | - |
| 12 | 107.66   | 215.32 |
| 13 | 99.38    | - |
| 14 | 92.28    | 184.56 |
| 15 | 86.13    | - |
| 16 | 80.74    | 161.48 |
| 17 | 75.99    | - |
| 18 | 71.77    | 143.54 |
| 19 | 67.99    | - |
| 20 | 64.59    | 129.18 |
| ... | ... | ... |

### Patterns 

The script comes with the following POC patterns:  

| Pattern | Details | Audio example |
| :--- | :--- | :--- |
| fibo | Repeats a given array of shape `(data points, latent dimensions)` along the fibonacci series of integers where `1` corresponds to the first row in the array, `2` corresponds to the first two rows in the array, ..., `8` corresponds to rows 0-7 and so on. Note that above tempo considerations do not apply to this pattern type. | [1](https://github.com/user-attachments/files/26414512/fibo_bll.mp3) [2](https://github.com/user-attachments/files/26414519/fibo_nsp.mp3) | 
| orale | Approximation to a standard sequence in electronic music building an array using a 3:1 scheme where the original array is repeated three times and a fourth time with subtle changes applied to its values. This sequence is then repeated and altered again in the same scheme of 3:1.  | [1](https://github.com/user-attachments/files/26414524/orale_bll.mp3) [2](https://github.com/user-attachments/files/26414523/orale_nsp.mp3) |
| blender | Blends two arrays into one another by replacing single data points sequentially after `--num_loops` repetitions, starting with the first value in the first dimension, followed by the first value in the second dimension and so on until the last value in the last dimensions has been reached. The pattern can be inverted, mirrored or mirror-inverted (implemented, not active). | [1](https://github.com/user-attachments/files/26414579/blender_nsp.mp3) [2](https://github.com/user-attachments/files/26414588/blender_bll.mp3) |
| swapper | Swaps values of randomly picked data points in two arrays of the same size and repeats the altered array `--num_loops` times. | [1](https://github.com/user-attachments/files/26414697/swapper_bll.mp3) [2](https://github.com/user-attachments/files/26414695/swapper_nsp.mp3) |

    
## Decoding to audio 

__decoder.py__ decodes one or many .npy arrays back to the audio domain using the same .ts model that has been used in the encoding step. 

```bash
python decoder.py \
--input /path/to/your/numpy/files \ # or path to a single .npy file
--model /path/to/your/pretrained/model.ts \ # array shape must match model's expected shape
--output /path/to/decoded/audio/files # default is './_decoder-output'
```

---

## Model inspection

__inspector.py__ inspects one or many models' `state_dict`checks for other parameters like sample rate, number of latent dimensions and architecture. Details can be stored to a .csv file using the `--save` and `--output` flags. With `--dict` and `--attr`, contents of dedicated items from the `state_dict` and attributes (methods, properties, data attributes) can be retrieved through CLI. The script generally works with other architectures, e.g. vschaos2, MSPrior or AFTER but it's not explicitly written for these model types; it partly works with SAT autoencoders for quick inspection. The script has informative value only, it does not create any particular data for use in the other scripts but can be handy for debugging

```bash
python inspector.py \
--input /path/to/your/pretrained/model.ts \ # or path containing multiple models
--extensions .ts \ # model file name extension
--save \ # stores model state_dict entries and shape to .csv
--output /path/to/output \ # default is'./_inspector-output'
--dict \ # display all params in state_dict and enters detail request mode in CLI
--attr \ # displays all attributes in model and enters detail request mode in CLI
--config # displays full model config as used in training (only SAT checkpoint files)
```

---

## Compatibility

- Python: 3.14
- Torch/ Torchaudio: 2.10

Tested with models self trained with RAVE <= v2.2.2. Inspection more or less works with vschaos2, MSPrior, AFTER and Stable Audio autoencoders. 

---

## Credits

- Built on [RAVE](https://github.com/acids-ircam/RAVE)

---

## License

UnRAVEl © 2026 by Martin Heinze is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
