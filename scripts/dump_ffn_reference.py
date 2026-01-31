#!/usr/bin/env python3
"""
Dump FFN intermediate values for debugging.
"""

import struct
import torch
import numpy as np
from pathlib import Path


def write_tensor(tensor, path):
    """Write tensor to binary file."""
    data = tensor.detach().float().cpu().numpy()
    shape = list(data.shape)
    while len(shape) < 4:
        shape.append(1)

    with open(path, 'wb') as f:
        for dim in reversed(shape[:4]):
            f.write(struct.pack('<q', dim))
        f.write(data.tobytes())
    print(f"  Wrote {path}: shape={list(tensor.shape)}")


def read_reference(path: str):
    """Read reference tensor from binary file."""
    with open(path, 'rb') as f:
        shape = list(struct.unpack('<4q', f.read(32)))
        data = np.frombuffer(f.read(), dtype=np.float32)
        shape.reverse()
        while len(shape) > 1 and shape[-1] == 1:
            shape.pop()
        return torch.from_numpy(data.reshape(shape).copy())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "../magpie_tts_multilingual_357m/magpie_tts_multilingual_357m.nemo"
    ref_dir = Path("test_data/reference")

    print("=== Dumping FFN Reference Values ===\n")

    from nemo.collections.tts.models import MagpieTTSModel
    print(f"Loading model from {model_path}...")
    model = MagpieTTSModel.restore_from(model_path)
    model.to(device)
    model.eval()

    layer = model.encoder.layers[0]
    ffn = layer.pos_ff

    print(f"\nFFN module: {type(ffn)}")
    print(f"  proj.conv.weight shape: {ffn.proj.conv.weight.shape}")
    print(f"  o_net.conv.weight shape: {ffn.o_net.conv.weight.shape}")

    # Load the input to norm_ff (output of self-attention + residual)
    # We need to compute this from the encoder
    input_with_pos = read_reference(ref_dir / "manual_enc_with_pos.bin")
    input_with_pos = input_with_pos.squeeze()  # Remove size-1 dims
    if input_with_pos.dim() == 2:
        input_with_pos = input_with_pos.unsqueeze(0)  # Add batch dim: [1, seq, d_model]
    input_with_pos = input_with_pos.to(device)
    print(f"\nInput (enc_with_pos) shape: {input_with_pos.shape}")

    with torch.no_grad():
        # Run through norm_self and self_attention
        x = layer.norm_self(input_with_pos)
        seq_len = x.shape[1]
        mask = torch.ones(1, seq_len, device=device, dtype=x.dtype)

        sa_out, _ = layer.self_attention(x, mask)
        x_after_sa = sa_out + input_with_pos  # residual
        print(f"After self-attention + residual: {x_after_sa.shape}")
        print(f"  First 8: {x_after_sa.flatten()[:8].cpu().numpy()}")

        # Now norm_ff
        x_norm_ff = layer.norm_pos_ff(x_after_sa)
        print(f"\nAfter norm_ff: {x_norm_ff.shape}")
        print(f"  First 8: {x_norm_ff.flatten()[:8].cpu().numpy()}")

        # FFN forward
        ffn_out = layer.pos_ff(x_norm_ff, mask)
        print(f"\nFFN output: {ffn_out.shape}")
        print(f"  First 8: {ffn_out.flatten()[:8].cpu().numpy()}")

        # Full layer output
        layer_out = ffn_out + x_after_sa
        print(f"\nFull layer output (FFN + residual): {layer_out.shape}")
        print(f"  First 8: {layer_out.flatten()[:8].cpu().numpy()}")

        # Save intermediate values
        write_tensor(x_after_sa, ref_dir / "debug_after_sa_residual.bin")
        write_tensor(x_norm_ff, ref_dir / "debug_norm_ff_input.bin")
        write_tensor(ffn_out, ref_dir / "debug_ffn_output.bin")

        # Manual FFN trace
        print("\n=== Manual FFN Trace ===")

        # FFN is ConvFFN with proj.conv (kernel_size=3) and o_net.conv (kernel_size=3)
        # proj: d_model -> d_ffn with SiLU activation
        # o_net: d_ffn -> d_model

        # First, let's trace through the actual FFN
        x_t = x_norm_ff.transpose(1, 2)  # [B, T, C] -> [B, C, T] for conv
        print(f"Input transposed for conv: {x_t.shape}")

        proj_out = ffn.proj(x_t)  # includes conv + activation
        print(f"After proj (conv + SiLU): {proj_out.shape}")
        print(f"  First 8: {proj_out.flatten()[:8].cpu().numpy()}")

        o_out = ffn.o_net(proj_out)
        print(f"After o_net: {o_out.shape}")
        print(f"  First 8: {o_out.flatten()[:8].cpu().numpy()}")

        o_out_t = o_out.transpose(1, 2)  # back to [B, T, C]
        print(f"After transpose back: {o_out_t.shape}")
        print(f"  First 8: {o_out_t.flatten()[:8].cpu().numpy()}")


if __name__ == "__main__":
    main()
