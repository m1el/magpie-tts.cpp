#!/usr/bin/env python3
"""
Audio Codec Inspection Script

This script loads the NeMo nano-codec and traces the forward pass of all modules
during decoding (codes -> waveform). Useful for understanding the architecture
and porting to GGML.

Usage:
    cd magpie.cpp && uv run scripts/inspect_codec.py
"""

from collections import OrderedDict
import inspect
import struct
import os
import torch
import numpy as np

# ==============================================================================
# Instrumentation utilities (same pattern as inspect_inference.py)
# ==============================================================================

def mk_hook(module_name):
    """Create a forward hook that logs module info and tensor shapes."""
    def hook(module, args, kwargs, output):
        if kwargs:
            args = (*args, kwargs)
        ty = get_ty((args, output))
        append_name(module_name, module, ty)
    return hook


visited = set()
def instrument_everything(model, prefix=''):
    """Recursively instrument all modules in a model."""
    append_name(prefix, model)
    if prefix != '':
        prefix = prefix + '.'

    if model in visited:
        return
    visited.add(model)
    for name, module in model.named_modules():
        if module in visited:
            continue
        visited.add(module)
        if name == '':
            continue
        module.register_forward_hook(mk_hook(prefix + name), with_kwargs=True)
        instrument_everything(module, prefix=prefix + name)


all_names = OrderedDict()
def append_name(name, mod, inout=None):
    """Record module information in a hierarchical dictionary."""
    chunks = name.split('.')
    cur = all_names
    for c in chunks:
        if c not in cur:
            cur[c] = {}
        cur = cur[c]
    cur['##cls'] = mod.__class__.__name__
    fn = mod.forward if hasattr(mod, 'forward') else mod
    try:
        cur['##sig'] = inspect.signature(fn)
    except (ValueError, TypeError):
        cur['##sig'] = ''
    if inout is not None:
        cur['##inout'] = inout


class TensorShape:
    """Helper class to pretty print tensor shapes."""
    def __init__(self, tensor):
        self.shape = list(tensor.shape)
        self.dtype = str(tensor.dtype).replace('torch.', '')
    def __str__(self):
        return f'Tensor[{self.dtype}]{self.shape}'
    def __repr__(self):
        return self.__str__()


def get_ty(obj):
    """Recursively get type information for nested structures."""
    if isinstance(obj, tuple):
        return tuple(get_ty(o) for o in obj)
    elif isinstance(obj, list):
        return [get_ty(o) for o in obj]
    elif isinstance(obj, dict):
        return {k: get_ty(v) for k, v in obj.items()}
    elif torch.is_tensor(obj):
        return TensorShape(obj)
    else:
        return type(obj).__name__


def pprint_names_flat(d, prefix='', file=None):
    """Print module names in flat format."""
    if prefix != '':
        prefix = prefix + '.'
    for k, v in d.items():
        if k.startswith('##'):
            continue
        if k != '':
            inout = v.get('##inout', '')
            if inout != '':
                i, o = inout
                inout = f' | {i} -> {o}'
            sig = v.get('##sig', '')
            if sig != '':
                inout = str(sig) + inout
            line = f'{prefix}{k}: {v.get("##cls", "unknown")}{inout}'
            print(line, file=file)
        pprint_names_flat(v, prefix=prefix + k, file=file)


# ==============================================================================
# Reference data dumping
# ==============================================================================

output_files = {}

def dump_tensor(tensor, filename, output_dir):
    """Save tensor data to a binary file for offline testing.

    GGML uses reversed dimension order: ne[0] = last PyTorch dim.
    We write data in C order (row-major) where last dim varies fastest.
    This matches GGML's expectation that ne[0] varies fastest.
    """
    filepath = os.path.join(output_dir, filename)
    shape = list(tensor.shape)
    shape.reverse()  # GGML stores dimensions in reverse order
    while len(shape) < 4:
        shape.append(1)

    data = tensor.detach().cpu().float().numpy()
    # Save in C (row-major) order - last PyTorch dim varies fastest
    # This matches GGML where ne[0] (= last PyTorch dim) varies fastest
    data_bytes = data.flatten('C').astype(np.float32).tobytes()

    with open(filepath, 'wb') as f:
        # Write 4 x int64 shape header
        for dim in shape:
            f.write(struct.pack('<q', dim))
        # Write float32 data
        f.write(data_bytes)

    print(f"  Saved {filename}: {list(tensor.shape)} -> {len(data_bytes)} bytes")
    return filepath


# ==============================================================================
# Codec loading and inspection
# ==============================================================================

def load_codec(codec_path: str, device: str = "cuda"):
    """Load the NeMo nano-codec from a .nemo file."""
    from nemo.collections.tts.models import AudioCodecModel

    print(f"Loading audio codec from {codec_path}...")
    codec = AudioCodecModel.restore_from(codec_path)
    codec.to(device)
    codec.eval()

    print(f"Codec loaded successfully on {device}")
    print(f"  - Sample rate: {codec.sample_rate}")
    print(f"  - Num codebooks: {codec.num_codebooks}")

    return codec


def print_codec_summary(codec):
    """Print a summary of the codec architecture."""
    print("\n" + "=" * 80)
    print("AUDIO CODEC SUMMARY")
    print("=" * 80)

    # Print main components
    print("\n### Main Components ###")
    for name, module in codec.named_children():
        if name.startswith('_'):
            continue
        param_count = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {module.__class__.__name__} ({param_count:,} params)")

    # Print audio decoder structure
    print("\n### Audio Decoder Structure ###")
    if hasattr(codec, 'audio_decoder'):
        dec = codec.audio_decoder
        print(f"  Type: {dec.__class__.__name__}")

        # Print pre_conv
        if hasattr(dec, 'pre_conv'):
            conv = dec.pre_conv.conv
            print(f"  pre_conv: in={conv.in_channels}, out={conv.out_channels}, kernel={conv.kernel_size}")

        # Print upsample layers
        if hasattr(dec, 'up_sample_conv_layers'):
            print(f"  Upsample layers: {len(dec.up_sample_conv_layers)}")
            for i, layer in enumerate(dec.up_sample_conv_layers):
                conv = layer.conv
                print(f"    [{i}]: in={conv.in_channels}, out={conv.out_channels}, "
                      f"kernel={conv.kernel_size}, stride={conv.stride}")

        # Print res layers
        if hasattr(dec, 'res_layers'):
            print(f"  Residual layers: {len(dec.res_layers)}")
            for i, layer in enumerate(dec.res_layers):
                n_blocks = len(layer.res_blocks)
                if n_blocks > 0:
                    inner_blocks = len(layer.res_blocks[0].res_blocks) if hasattr(layer.res_blocks[0], 'res_blocks') else 0
                    print(f"    [{i}]: {n_blocks} blocks x {inner_blocks} inner blocks each")

        # Print activations
        if hasattr(dec, 'activations'):
            print(f"  Activations: {len(dec.activations)}")
            for i, act in enumerate(dec.activations):
                act_type = act.activation.__class__.__name__
                if hasattr(act.activation, 'alpha'):
                    alpha_shape = list(act.activation.alpha.shape)
                    print(f"    [{i}]: {act_type} alpha={alpha_shape}")
                else:
                    print(f"    [{i}]: {act_type}")

        # Print post_conv
        if hasattr(dec, 'post_conv'):
            conv = dec.post_conv.conv
            print(f"  post_conv: in={conv.in_channels}, out={conv.out_channels}, kernel={conv.kernel_size}")

    # Print vector quantizer (FSQ)
    print("\n### Vector Quantizer ###")
    if hasattr(codec, 'vector_quantizer'):
        vq = codec.vector_quantizer
        print(f"  Type: {vq.__class__.__name__}")
        if hasattr(vq, 'fsqs'):
            print(f"  Num FSQ codebooks: {len(vq.fsqs)}")
            for i, fsq in enumerate(vq.fsqs):
                if hasattr(fsq, 'num_levels'):
                    levels = fsq.num_levels.flatten().tolist()
                    print(f"    [{i}]: levels={levels}")

    print("\n" + "=" * 80)


def decode_codes(codec, codes, dump_intermediates=False, output_dir=None):
    """
    Decode audio codes to waveform, optionally dumping intermediate tensors.

    Args:
        codec: Loaded AudioCodecModel
        codes: Tensor of shape [B, num_codebooks, T] with code indices
        dump_intermediates: Whether to dump intermediate tensors
        output_dir: Directory to save intermediate tensors

    Returns:
        Audio waveform tensor
    """
    device = next(codec.parameters()).device

    if dump_intermediates and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nDumping intermediate tensors to {output_dir}/")

    with torch.no_grad():
        # Step 1: Dequantize codes to latent
        # The vector quantizer converts codes to continuous latent representations
        codes = codes.to(device)
        codes_len = torch.tensor([codes.shape[-1]], device=device)

        print(f"\nInput codes shape: {list(codes.shape)}")

        if dump_intermediates and output_dir:
            # Save input codes
            dump_tensor(codes.float(), "codec_input_codes.bin", output_dir)

        # Dequantize using FSQ
        vq = codec.vector_quantizer

        # FSQ dequantization: convert indices to continuous values
        # GroupFiniteScalarQuantizer.decode expects [D, B, T] where D=num_codebooks
        # and returns [B, 32, T] (8 groups * 4 dims per group)
        indices = codes.transpose(0, 1)  # [B, 8, T] -> [8, B, T]
        latent = vq.decode(indices=indices, input_len=codes_len)  # [B, 32, T]

        # Also dump individual codebook dequantizations for debugging
        if dump_intermediates and output_dir:
            for i, fsq in enumerate(vq.fsqs):
                indices_i = indices[i:i+1, :, :]  # [1, B, T]
                dequant_i = fsq.decode(indices=indices_i, input_len=codes_len)  # [B, 4, T]
                dump_tensor(dequant_i, f"codec_fsq_dequant_{i}.bin", output_dir)

        print(f"Latent after FSQ dequant: {list(latent.shape)}")

        if dump_intermediates and output_dir:
            dump_tensor(latent, "codec_latent.bin", output_dir)

        # Step 2: Run audio decoder
        dec = codec.audio_decoder

        # pre_conv
        audio_len = codes_len.clone()
        out = dec.pre_conv(inputs=latent, input_len=audio_len)
        print(f"After pre_conv: {list(out.shape)}")

        if dump_intermediates and output_dir:
            dump_tensor(out, "codec_pre_conv.bin", output_dir)

        # Upsample + residual layers
        for i, (act, res_layer, up_conv, up_rate) in enumerate(zip(
            dec.activations, dec.res_layers, dec.up_sample_conv_layers, dec.up_sample_rates
        )):
            audio_len = (audio_len * up_rate).long()

            # Activation (Snake)
            out = act(out)
            if dump_intermediates and output_dir:
                dump_tensor(out, f"codec_act_{i}.bin", output_dir)

            # Upsample conv transpose
            out = up_conv(inputs=out, input_len=audio_len)
            print(f"After upsample[{i}]: {list(out.shape)}")
            if dump_intermediates and output_dir:
                dump_tensor(out, f"codec_upsample_{i}.bin", output_dir)

            # Residual layer
            out = res_layer(inputs=out, input_len=audio_len)
            print(f"After res_layer[{i}]: {list(out.shape)}")
            if dump_intermediates and output_dir:
                dump_tensor(out, f"codec_res_{i}.bin", output_dir)

        # post_activation
        out = dec.post_activation(out)
        if dump_intermediates and output_dir:
            dump_tensor(out, "codec_post_act.bin", output_dir)

        # post_conv
        out = dec.post_conv(inputs=out, input_len=audio_len)
        print(f"After post_conv: {list(out.shape)}")
        if dump_intermediates and output_dir:
            dump_tensor(out, "codec_post_conv.bin", output_dir)

        # Output activation (tanh)
        audio = dec.out_activation(out)
        audio = audio.squeeze(1)  # [B, T]

        print(f"Final audio shape: {list(audio.shape)}")
        if dump_intermediates and output_dir:
            dump_tensor(audio, "codec_output.bin", output_dir)

        return audio, audio_len


def main():
    """Main entry point for codec inspection."""
    import argparse

    parser = argparse.ArgumentParser(description="Inspect NeMo Audio Codec")
    parser.add_argument(
        "--codec",
        type=str,
        default="../nemo-nano-codec-22khz-1.89kbps-21.5fps/nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo",
        help="Path to NeMo codec file (.nemo)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_data/reference/codec",
        help="Directory to save reference tensors"
    )
    parser.add_argument(
        "--no-dump",
        action="store_true",
        help="Skip dumping intermediate tensors"
    )
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Save generated audio to file"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=10,
        help="Number of audio frames to test with"
    )
    args = parser.parse_args()

    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, using CPU (may be slow)")

    # Load codec
    try:
        codec = load_codec(args.codec, device)
    except Exception as e:
        print(f"Failed to load codec: {e}")
        print("\nMake sure you have the NeMo package installed with codec support.")
        return

    # Print summary
    print_codec_summary(codec)

    # Create test codes
    num_codebooks = codec.num_codebooks
    num_frames = args.num_frames

    # Use fixed seed for reproducibility
    torch.manual_seed(42)

    # Generate random codes (within valid range)
    # FSQ uses levels [8, 8, 8, 6] so max index is 8*8*8*6 - 1 = 3071
    # But actual codebook size is typically 2016
    codebook_size = 2016  # From the model
    codes = torch.randint(0, codebook_size, (1, num_codebooks, num_frames), device=device)

    print(f"\nTest codes: {list(codes.shape)}, range [{codes.min()}, {codes.max()}]")

    # Instrument model for tracing
    global visited, all_names
    visited = set()
    all_names = OrderedDict()
    instrument_everything(codec)

    # Decode codes
    dump_intermediates = not args.no_dump
    audio, audio_len = decode_codes(
        codec, codes,
        dump_intermediates=dump_intermediates,
        output_dir=args.output_dir
    )

    print(f"\nGenerated audio: {list(audio.shape)}, length={audio_len[0].item()} samples")
    print(f"Audio range: [{audio.min():.4f}, {audio.max():.4f}]")

    # Print module trace
    if dump_intermediates:
        print("\n" + "=" * 80)
        print("MODULE TRACE")
        print("=" * 80)

        trace_file = os.path.join(args.output_dir, "codec_trace.txt")
        with open(trace_file, 'w') as f:
            pprint_names_flat(all_names, file=f)
        print(f"Module trace saved to: {trace_file}")

        # Print first 30 lines
        import io
        buf = io.StringIO()
        pprint_names_flat(all_names, file=buf)
        lines = buf.getvalue().split('\n')
        print("\n(First 30 lines)")
        for line in lines[:30]:
            print(line)
        if len(lines) > 30:
            print(f"... ({len(lines) - 30} more lines)")

    # Save audio
    if args.save_audio:
        import soundfile as sf
        audio_path = os.path.join(args.output_dir, "test_audio.wav")
        audio_np = audio[0].cpu().numpy()
        sf.write(audio_path, audio_np, codec.sample_rate)
        print(f"\nAudio saved to: {audio_path}")

    print("\nCodec inspection complete!")


if __name__ == "__main__":
    main()
