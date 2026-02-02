#!/usr/bin/env python3
"""
Convert Magpie TTS model weights to GGUF format.

Usage:
    uv run scripts/convert_magpie_to_gguf.py ../magpie_tts_multilingual_357m/magpie_tts_multilingual_357m.nemo weights/magpie-357m.gguf
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
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8

# Block sizes for quantized types
QK4_0 = 32
QK8_0 = 32


def write_string(f, s: str | bytes):
    """Write a GGUF string (length + data, no null terminator)."""
    if isinstance(s, str):
        data = s.encode('utf-8')
    else:
        data = s
    f.write(struct.pack('<Q', len(data)))
    f.write(data)


def write_kv_string(f, key: str, value: str):
    """Write a string key-value pair."""
    write_string(f, key)
    f.write(struct.pack('<i', GGUF_TYPE_STRING))
    write_string(f, value)


def write_kv_uint32(f, key: str, value: int):
    """Write a uint32 key-value pair."""
    write_string(f, key)
    f.write(struct.pack('<i', GGUF_TYPE_UINT32))
    f.write(struct.pack('<I', value))


def write_kv_int32(f, key: str, value: int):
    """Write an int32 key-value pair."""
    write_string(f, key)
    f.write(struct.pack('<i', GGUF_TYPE_INT32))
    f.write(struct.pack('<i', value))


def write_kv_float32(f, key: str, value: float):
    """Write a float32 key-value pair."""
    write_string(f, key)
    f.write(struct.pack('<i', GGUF_TYPE_FLOAT32))
    f.write(struct.pack('<f', value))


def quantize_q8_0(data: np.ndarray) -> bytes:
    """Quantize float32 array to Q8_0 format."""
    data = data.astype(np.float32).flatten()
    n_elements = len(data)

    if n_elements % QK8_0 != 0:
        pad_size = QK8_0 - (n_elements % QK8_0)
        data = np.pad(data, (0, pad_size), mode='constant', constant_values=0)

    n_blocks = len(data) // QK8_0
    blocks = data.reshape(n_blocks, QK8_0)

    amax = np.max(np.abs(blocks), axis=1)
    scales = np.where(amax != 0, amax / 127.0, 0.0).astype(np.float16)

    scales_expanded = scales[:, np.newaxis].astype(np.float32)
    scales_safe = np.where(scales_expanded != 0, scales_expanded, 1.0)
    quantized = np.round(blocks / scales_safe).astype(np.int8)
    quantized = np.where(scales_expanded != 0, quantized, 0).astype(np.int8)

    block_dtype = np.dtype([('scale', np.float16), ('quants', np.int8, QK8_0)])
    output_arr = np.empty(n_blocks, dtype=block_dtype)
    output_arr['scale'] = scales
    output_arr['quants'] = quantized

    return output_arr.tobytes()


def quantize_q4_0(data: np.ndarray) -> bytes:
    """Quantize float32 array to Q4_0 format."""
    data = data.astype(np.float32).flatten()
    n_elements = len(data)

    if n_elements % QK4_0 != 0:
        pad_size = QK4_0 - (n_elements % QK4_0)
        data = np.pad(data, (0, pad_size), mode='constant', constant_values=0)

    n_blocks = len(data) // QK4_0
    blocks = data.reshape(n_blocks, QK4_0)

    amax = np.max(np.abs(blocks), axis=1)
    scales = np.where(amax != 0, amax / 7.0, 0.0).astype(np.float16)

    scales_expanded = scales[:, np.newaxis].astype(np.float32)
    scales_safe = np.where(scales_expanded != 0, scales_expanded, 1.0)
    quantized = np.round(blocks / scales_safe).astype(np.int8)
    quantized = np.clip(quantized, -8, 7)
    quantized = np.where(scales_expanded != 0, quantized, 0)

    quantized_u = (quantized + 8).astype(np.uint8)
    low = quantized_u[:, :QK4_0//2] & 0x0F
    high = quantized_u[:, QK4_0//2:] & 0x0F
    packed = (low | (high << 4)).astype(np.uint8)

    block_dtype = np.dtype([('scale', np.float16), ('quants', np.uint8, QK4_0 // 2)])
    output_arr = np.empty(n_blocks, dtype=block_dtype)
    output_arr['scale'] = scales
    output_arr['quants'] = packed

    return output_arr.tobytes()


def load_nemo_model(path: str) -> Tuple[Dict[str, np.ndarray], dict]:
    """Load NeMo model and return weights and config."""
    with tarfile.open(path) as tar:
        # Extract config
        model_config = tar.extractfile("./model_config.yaml")
        config = yaml.safe_load(model_config)

        # Extract weights
        weights = tar.extractfile("./model_weights.ckpt")
        torch_weights = torch.load(weights, weights_only=True, map_location='cpu')
        numpy_weights = {name: tensor.numpy() for name, tensor in torch_weights.items()}

    return numpy_weights, config


def should_quantize(name: str, patterns: list[str]) -> bool:
    """Determine if a tensor should be quantized."""
    if not patterns:
        # Default: quantize encoder/decoder layer weights (not biases, not norms, not embeddings)
        patterns = [
            # Self-attention: qkv_net.weight, o_net.weight
            r"\.layers\.\d+\.self_attention\.(qkv_net|o_net)\.weight$",
            # Cross-attention: q_net.weight, kv_net.weight, o_net.weight
            r"\.layers\.\d+\.cross_attention\.(q_net|kv_net|o_net)\.weight$",
            # FFN conv weights: pos_ff.proj.conv.weight, pos_ff.o_net.conv.weight
            r"\.layers\.\d+\.pos_ff\.(proj|o_net)\.conv\.weight$",
            # Final projection
            r"^final_proj\.weight$",
            # Local transformer out projections
            r"^local_transformer_out_projections\.\d+\.weight$",
            # Local transformer in projection
            r"^local_transformer_in_projection\.weight$",
        ]

    for pattern in patterns:
        if re.search(pattern, name):
            return True
    return False


def convert_magpie_to_gguf(
    input_path: str,
    output_path: str,
    quant_type: str = None,
    quant_patterns: list[str] = None,
):
    """Convert Magpie TTS weights to GGUF format."""
    print(f"Loading model from {input_path}...")
    tensors, config = load_nemo_model(input_path)
    print(f"Loaded {len(tensors)} tensors")

    # Parse quantization type
    ggml_quant_type = GGML_TYPE_F32
    quant_name = "F32"
    if quant_type:
        quant_type = quant_type.lower()
        if quant_type in ("q8_0", "q8"):
            ggml_quant_type = GGML_TYPE_Q8_0
            quant_name = "Q8_0"
        elif quant_type in ("q4_0", "q4"):
            ggml_quant_type = GGML_TYPE_Q4_0
            quant_name = "Q4_0"
        elif quant_type in ("f16", "fp16"):
            ggml_quant_type = GGML_TYPE_F16
            quant_name = "F16"

    if quant_patterns is None:
        quant_patterns = []

    # Model hyperparameters from architecture analysis
    hparams = {
        "magpie.sample_rate": 22050,
        "magpie.num_codebooks": 8,
        "magpie.codebook_size": 2016,
        "magpie.vocab_size_per_codebook": 2024,
        "magpie.text_vocab_size": 2380,
        "magpie.d_model": 768,
        "magpie.d_ffn": 3072,
        "magpie.encoder_layers": 6,
        "magpie.decoder_layers": 12,
        "magpie.encoder_heads": 12,
        "magpie.decoder_sa_heads": 12,
        "magpie.decoder_xa_heads": 1,
        "magpie.local_transformer_dim": 256,
        "magpie.local_transformer_layers": 1,
        "magpie.num_baked_speakers": 5,
        "magpie.baked_context_frames": 110,
        "magpie.text_bos_id": 2378,
        "magpie.text_eos_id": 2379,
        "magpie.audio_bos_id": 2016,
        "magpie.audio_eos_id": 2017,
        "magpie.context_audio_bos_id": 2018,
        "magpie.context_audio_eos_id": 2019,
        "magpie.mask_token_id": 2020,
    }

    # String hparams (vocabulary and dictionary)
    string_hparams = {}

    # Try to extract tokenizer vocabulary from model
    tokenizer_data_dir = Path(__file__).parent.parent / "tokenizer_data"
    vocab_file = tokenizer_data_dir / "vocab.txt"
    dict_file = tokenizer_data_dir / "dict.txt"

    if vocab_file.exists():
        print(f"Loading vocabulary from {vocab_file}...")
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab_tokens = [line.rstrip('\n') for line in f]
        # Store as newline-separated string (tokens don't contain newlines)
        string_hparams["magpie.tokenizer.vocab"] = "\n".join(vocab_tokens)
        hparams["magpie.tokenizer.vocab_size"] = len(vocab_tokens)
        print(f"  Vocabulary size: {len(vocab_tokens)}")

    if dict_file.exists():
        print(f"Loading pronunciation dictionary from {dict_file}...")
        # Store as word\tpron\n format (standard TSV)
        with open(dict_file, "r", encoding="utf-8") as f:
            dict_content = f.read()
        string_hparams["magpie.tokenizer.dict"] = dict_content
        dict_count = dict_content.count('\n')
        hparams["magpie.tokenizer.dict_size"] = dict_count
        print(f"  Dictionary entries: {dict_count}")

    # Load special token info
    special_file = tokenizer_data_dir / "special_tokens.txt"
    if special_file.exists():
        print(f"Loading special tokens from {special_file}...")
        with open(special_file, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    if value != "None" and value.isdigit():
                        hparams[f"magpie.tokenizer.{key}"] = int(value)

    # Prepare tensor info
    tensor_infos = []
    current_offset = 0

    stats = {
        "f32_tensors": 0,
        "f32_bytes": 0,
        "quantized_tensors": 0,
        "quantized_bytes_before": 0,
        "quantized_bytes_after": 0,
    }

    skipped_bytes = 0
    for name, data in sorted(tensors.items()):
        # Skip tensors we don't need for inference
        skip_patterns = [
            '_codec_model', 'speaker_encoder', '_speaker_encoder',
            'causal_mask',  # Computed at runtime
            '_baked_embedding_D', '_baked_embedding_T',  # Metadata scalars
            'baked_context_embedding_len',  # All same length (110)
        ]
        if any(skip in name for skip in skip_patterns):
            skipped_bytes += data.nbytes
            print(f"Skipping: {name} ({data.shape})")
            continue

        # GGUF uses row-major with dimensions in reverse order
        shape_gguf = list(reversed(data.shape))
        while len(shape_gguf) < 4:
            shape_gguf.append(1)

        n_elements = int(np.prod(data.shape))

        # Decide whether to quantize
        # Block-based quantization (Q4_0, Q8_0) requires inner dimension >= block_size (32)
        # F16 has no such constraint
        inner_dim = data.shape[-1] if len(data.shape) >= 1 else 0
        block_size = 32  # QK8_0 and QK4_0 block sizes
        is_block_quant = ggml_quant_type in (GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
        do_quantize = (
            ggml_quant_type != GGML_TYPE_F32
            and should_quantize(name, quant_patterns)
            and n_elements >= 256
            and len(data.shape) >= 2
            and (not is_block_quant or inner_dim >= block_size)  # Block size only for Q4/Q8
        )

        if do_quantize:
            tensor_type = ggml_quant_type
            if ggml_quant_type == GGML_TYPE_Q8_0:
                tensor_data = quantize_q8_0(data)
            elif ggml_quant_type == GGML_TYPE_Q4_0:
                tensor_data = quantize_q4_0(data)
            elif ggml_quant_type == GGML_TYPE_F16:
                tensor_data = data.astype(np.float16).tobytes()
            else:
                tensor_data = data.astype(np.float32).tobytes()
                tensor_type = GGML_TYPE_F32

            stats["quantized_tensors"] += 1
            stats["quantized_bytes_before"] += n_elements * 4
            stats["quantized_bytes_after"] += len(tensor_data)
            quant_str = f"-> {quant_name}"
        else:
            tensor_type = GGML_TYPE_F32
            tensor_data = data.astype(np.float32).tobytes()
            stats["f32_tensors"] += 1
            stats["f32_bytes"] += len(tensor_data)
            quant_str = ""

        print(f"  {name}: {data.shape} {data.dtype} {quant_str}")

        # Calculate aligned offset
        aligned_offset = (current_offset + GGUF_DEFAULT_ALIGNMENT - 1) // GGUF_DEFAULT_ALIGNMENT * GGUF_DEFAULT_ALIGNMENT

        tensor_infos.append({
            'name': name,
            'shape': shape_gguf[:4],
            'n_dims': len(data.shape),
            'type': tensor_type,
            'offset': aligned_offset,
            'data': tensor_data,
        })

        current_offset = aligned_offset + len(tensor_data)

    # Print stats
    print(f"\nSkipped: {skipped_bytes / 1e6:.1f} MB (causal masks, scalars)")
    print(f"Quantization: {quant_name}")
    print(f"  F32 tensors: {stats['f32_tensors']} ({stats['f32_bytes'] / 1e6:.1f} MB)")
    if stats["quantized_tensors"] > 0:
        ratio = stats["quantized_bytes_before"] / stats["quantized_bytes_after"]
        print(f"  Quantized tensors: {stats['quantized_tensors']}")
        print(f"    Before: {stats['quantized_bytes_before'] / 1e6:.1f} MB")
        print(f"    After:  {stats['quantized_bytes_after'] / 1e6:.1f} MB")
        print(f"    Ratio:  {ratio:.2f}x compression")

    total_before = stats['f32_bytes'] + stats['quantized_bytes_before']
    total_after = stats['f32_bytes'] + stats['quantized_bytes_after']
    print(f"\nTotal: {total_before / 1e6:.1f} MB -> {total_after / 1e6:.1f} MB ({total_after / total_before * 100:.1f}%)")

    # Write GGUF file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting GGUF to {output_path}...")

    with open(output_path, 'wb') as f:
        # Write header
        f.write(GGUF_MAGIC)
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<q', len(tensor_infos)))  # n_tensors
        f.write(struct.pack('<q', len(hparams) + len(string_hparams) + 2))   # n_kv

        # Write KV pairs
        write_kv_string(f, "general.architecture", "magpie")
        write_kv_string(f, "general.name", "magpie-tts-multilingual-357m")

        for key, value in hparams.items():
            if isinstance(value, int):
                write_kv_uint32(f, key, value)
            elif isinstance(value, float):
                write_kv_float32(f, key, value)

        # Write string hparams (vocabulary, dictionary)
        for key, value in string_hparams.items():
            write_kv_string(f, key, value)

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
        description="Convert Magpie TTS model to GGUF format"
    )
    parser.add_argument("input", help="Input NeMo model file (.nemo)")
    parser.add_argument("output", help="Output GGUF file (.gguf)")
    parser.add_argument(
        "-q", "--quantize",
        choices=["q8_0", "q8", "q4_0", "q4", "f16"],
        help="Quantization type"
    )
    parser.add_argument(
        "-p", "--pattern",
        action="append",
        dest="patterns",
        default=[],
        help="Regex pattern for tensors to quantize"
    )
    args = parser.parse_args()

    convert_magpie_to_gguf(
        args.input,
        args.output,
        quant_type=args.quantize,
        quant_patterns=args.patterns,
    )


if __name__ == "__main__":
    main()
