# Magpie TTS GGML port

This is a rewrite of Text-to-Speech from NVidia's NeMo framework.

The goal of this project is to have a fast-loading, low-dependency text-to-speech software. This program only uses [ggml-org/ggml](https://github.com/ggml-org/ggml) library to work with neural networks.

## Usage

```bash
# Basic usage
./magpie-tts -t "Hello, world!" -o output.wav

# With custom temperature and top-k
./magpie-tts -t "Hello, world!" -o output.wav --temp 0.5 --top-k 50

# Quiet mode (only outputs filename)
./magpie-tts -t "Hello" -o hello.wav -q
```

Run `./magpie-tts --help` for all options.

## Model Weights

Notice: the weights are [Licensed by NVIDIA Corporation under the NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)

### Pre-converted GGUF Weights

Download pre-converted weights from HuggingFace:
https://huggingface.co/m1el/magpie-tts-multilingual-357m-gguf

Place the downloaded files in the `weights/` directory.

### Convert from NeMo (alternative)

Convert from NeMo format using the conversion scripts:

```bash
# Convert TTS model
python scripts/convert_magpie_to_gguf.py \
    path/to/magpie_tts_multilingual_357m.nemo \
    weights/magpie-357m-f32.gguf

# Convert audio codec
python scripts/convert_codec_to_gguf.py \
    path/to/nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo \
    weights/nano-codec-f32.gguf

# Optional: Quantize to Q8 for smaller size (679 MB vs 858 MB)
python scripts/convert_magpie_to_gguf.py \
    path/to/magpie_tts_multilingual_357m.nemo \
    weights/magpie-357m-q8.gguf -q q8
```

## Development

To develop this project you need to clone [ggml-org/ggml](https://github.com/ggml-org/ggml) to this directory, then build ggml with all your favorite settings.

```bash
git clone https://github.com/ggml-org/ggml.git
mkdir ggml/build
cd ggml/build
cmake .. -DGGML_CUDA=ON  # optional: enable CUDA
make -j8
cd ../..
```

Once you have built ggml, you can build this binary:
```bash
make magpie-tts
```

## Performance

Benchmarked on RTX 4080:

| Version | Speed | Notes |
|---------|-------|-------|
| Graph-Reuse | 154 fps | Batched context + allocator reuse |
| GPU-Optimized | 134 fps | GPU-resident KV cache |
| Standard | 64 fps | Full decoder each step |

Output: 22050 Hz mono WAV audio.

## License

MIT
