#!/usr/bin/env python3
"""
Convert NeMo Nano Codec decoder weights to GGUF format.

The codec uses FSQ (Finite Scalar Quantization) which doesn't need learned embeddings.
We only need the audio decoder for TTS inference (codes -> waveform).

Usage:
    uv run scripts/convert_codec_to_gguf.py ../nemo-nano-codec-22khz-1.89kbps-21.5fps/nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo weights/nano-codec.gguf
"""

import argparse
import struct
import tarfile
from typing import Dict, Tuple
import torch
import yaml
import numpy as np
from pathlib import Path
import re

# GGUF constants
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# GGUF metadata types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8

# GGML tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


def write_string(f, s: str | bytes):
    """Write a GGUF string (length + data, no null terminator)."""
    if isinstance(s, str):
        data = s.encode('utf-8')
    else:
        data = s
    f.write(struct.pack('<Q', len(data)))
    f.write(data)


def write_kv_string(f, key: str, value: str):
    write_string(f, key)
    f.write(struct.pack('<i', GGUF_TYPE_STRING))
    write_string(f, value)


def write_kv_uint32(f, key: str, value: int):
    write_string(f, key)
    f.write(struct.pack('<i', GGUF_TYPE_UINT32))
    f.write(struct.pack('<I', value))


def write_kv_float32(f, key: str, value: float):
    write_string(f, key)
    f.write(struct.pack('<i', GGUF_TYPE_FLOAT32))
    f.write(struct.pack('<f', value))


def load_nemo_codec(path: str) -> Tuple[Dict[str, np.ndarray], dict]:
    """Load NeMo codec model and return weights and config."""
    with tarfile.open(path) as tar:
        # Extract config
        model_config = tar.extractfile("./model_config.yaml")
        config = yaml.safe_load(model_config)

        # Extract weights - only audio_decoder and vector_quantizer
        weights = tar.extractfile("./model_weights.ckpt")
        torch_weights = torch.load(weights, weights_only=True, map_location='cpu')

        # Filter to only decoder and FSQ params
        numpy_weights = {}
        for name, tensor in torch_weights.items():
            if name.startswith(('audio_decoder', 'vector_quantizer')):
                numpy_weights[name] = tensor.numpy()

    return numpy_weights, config


def convert_codec_to_gguf(
    input_path: str,
    output_path: str,
    use_f16: bool = False,
):
    """Convert codec decoder weights to GGUF format."""
    print(f"Loading codec from {input_path}...")
    tensors, config = load_nemo_codec(input_path)
    print(f"Loaded {len(tensors)} tensors (decoder + vector_quantizer only)")

    # Model hyperparameters
    hparams = {
        "codec.sample_rate": 22050,
        "codec.num_codebooks": 8,
        "codec.codebook_size": 2016,
        "codec.hop_length": 1024,  # 22050 / 21.5 fps ≈ 1024
        "codec.latent_dim": 32,    # Input dim to decoder
    }

    # Prepare tensor info
    tensor_infos = []
    current_offset = 0
    total_bytes = 0

    def shorten_name(name):
        """Shorten tensor names to fit GGML's 64 char limit."""
        # Abbreviations for common patterns
        replacements = [
            ('audio_decoder.', 'dec.'),
            ('vector_quantizer.', 'vq.'),
            ('.res_layers.', '.rl.'),
            ('.res_blocks.', '.rb.'),
            ('.up_sample_conv_layers.', '.up.'),
            ('.activations.', '.act.'),
            ('.input_activation.activation.snake_act.', '.in_act.'),
            ('.skip_activation.activation.snake_act.', '.sk_act.'),
            ('.input_conv.conv.', '.in_conv.'),
            ('.skip_conv.conv.', '.sk_conv.'),
            ('.post_activation.activation.snake_act.', '.post_act.'),
            ('.pre_conv.conv.', '.pre.'),
            ('.post_conv.conv.', '.post.'),
            ('.conv.', '.c.'),
        ]
        result = name
        for old, new in replacements:
            result = result.replace(old, new)
        return result

    for name, data in sorted(tensors.items()):
        # Convert weight normalization params to combined weights
        # PyTorch stores: original0 (g), original1 (v)
        # Combined: g * v / ||v||
        if '.parametrizations.weight.original' in name:
            continue  # Skip, we'll handle these specially

        # GGUF uses row-major with dimensions in reverse order
        shape_gguf = list(reversed(data.shape))
        while len(shape_gguf) < 4:
            shape_gguf.append(1)

        tensor_type = GGML_TYPE_F16 if use_f16 else GGML_TYPE_F32
        if use_f16:
            tensor_data = data.astype(np.float16).tobytes()
        else:
            tensor_data = data.astype(np.float32).tobytes()

        short_name = shorten_name(name)
        print(f"  {short_name}: {data.shape} {data.dtype}")

        aligned_offset = (current_offset + GGUF_DEFAULT_ALIGNMENT - 1) // GGUF_DEFAULT_ALIGNMENT * GGUF_DEFAULT_ALIGNMENT

        tensor_infos.append({
            'name': short_name,
            'shape': shape_gguf[:4],
            'n_dims': len(data.shape),
            'type': tensor_type,
            'offset': aligned_offset,
            'data': tensor_data,
        })

        current_offset = aligned_offset + len(tensor_data)
        total_bytes += len(tensor_data)

    # Handle weight normalization: combine g and v into single weight
    # Find pairs of original0 (g) and original1 (v)
    wn_pairs = {}
    for name, data in tensors.items():
        if '.parametrizations.weight.original0' in name:
            base = name.replace('.parametrizations.weight.original0', '')
            if base not in wn_pairs:
                wn_pairs[base] = {}
            wn_pairs[base]['g'] = data
        elif '.parametrizations.weight.original1' in name:
            base = name.replace('.parametrizations.weight.original1', '')
            if base not in wn_pairs:
                wn_pairs[base] = {}
            wn_pairs[base]['v'] = data

    # Combine weight normalization pairs
    for base, pair in wn_pairs.items():
        if 'g' in pair and 'v' in pair:
            g = pair['g']  # (out_ch, 1, 1)
            v = pair['v']  # (out_ch, in_ch, kernel)

            # Compute normalized weight: g * v / ||v||
            v_norm = np.sqrt(np.sum(v ** 2, axis=(1, 2), keepdims=True) + 1e-12)
            weight = g * v / v_norm

            name = base + '.weight'
            short_name = shorten_name(name)

            shape_gguf = list(reversed(weight.shape))
            while len(shape_gguf) < 4:
                shape_gguf.append(1)

            tensor_type = GGML_TYPE_F16 if use_f16 else GGML_TYPE_F32
            if use_f16:
                tensor_data = weight.astype(np.float16).tobytes()
            else:
                tensor_data = weight.astype(np.float32).tobytes()

            print(f"  {short_name}: {weight.shape} (combined from weight_norm)")

            aligned_offset = (current_offset + GGUF_DEFAULT_ALIGNMENT - 1) // GGUF_DEFAULT_ALIGNMENT * GGUF_DEFAULT_ALIGNMENT

            tensor_infos.append({
                'name': short_name,
                'shape': shape_gguf[:4],
                'n_dims': len(weight.shape),
                'type': tensor_type,
                'offset': aligned_offset,
                'data': tensor_data,
            })

            current_offset = aligned_offset + len(tensor_data)
            total_bytes += len(tensor_data)

    print(f"\nTotal: {total_bytes / 1e6:.1f} MB")

    # Write GGUF file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting GGUF to {output_path}...")

    with open(output_path, 'wb') as f:
        # Write header
        f.write(GGUF_MAGIC)
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<q', len(tensor_infos)))
        f.write(struct.pack('<q', len(hparams) + 2))

        # Write KV pairs
        write_kv_string(f, "general.architecture", "nano-codec")
        write_kv_string(f, "general.name", "nemo-nano-codec-22khz")

        for key, value in hparams.items():
            if isinstance(value, int):
                write_kv_uint32(f, key, value)
            elif isinstance(value, float):
                write_kv_float32(f, key, value)

        # Write tensor infos
        for info in tensor_infos:
            write_string(f, info['name'])
            f.write(struct.pack('<I', info['n_dims']))
            for dim in info['shape'][:info['n_dims']]:
                f.write(struct.pack('<q', dim))
            f.write(struct.pack('<i', info['type']))
            f.write(struct.pack('<Q', info['offset']))

        # Align before tensor data
        current_pos = f.tell()
        aligned_pos = (current_pos + GGUF_DEFAULT_ALIGNMENT - 1) // GGUF_DEFAULT_ALIGNMENT * GGUF_DEFAULT_ALIGNMENT
        f.write(b'\x00' * (aligned_pos - current_pos))

        data_start = f.tell()

        # Write tensor data
        for info in tensor_infos:
            target_pos = data_start + info['offset']
            current_pos = f.tell()
            if target_pos > current_pos:
                f.write(b'\x00' * (target_pos - current_pos))
            f.write(info['data'])

        file_size = f.tell()

    print(f"Written {file_size / 1024 / 1024:.2f} MB")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert NeMo Nano Codec to GGUF format"
    )
    parser.add_argument("input", help="Input NeMo codec file (.nemo)")
    parser.add_argument("output", help="Output GGUF file (.gguf)")
    parser.add_argument(
        "--f16",
        action="store_true",
        help="Use FP16 precision"
    )
    args = parser.parse_args()

    convert_codec_to_gguf(
        args.input,
        args.output,
        use_f16=args.f16,
    )


if __name__ == "__main__":
    main()
