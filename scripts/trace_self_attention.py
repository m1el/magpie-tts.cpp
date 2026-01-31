#!/usr/bin/env python3
"""
Trace NeMo's self-attention to capture exact intermediate values.
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

        # Reverse shape from GGML order to PyTorch order
        shape.reverse()
        # Remove size-1 dims from the end
        while len(shape) > 1 and shape[-1] == 1:
            shape.pop()

        return torch.from_numpy(data.reshape(shape).copy())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "../magpie_tts_multilingual_357m/magpie_tts_multilingual_357m.nemo"
    ref_dir = Path("test_data/reference")

    print("=== Tracing NeMo Self-Attention ===\n")

    # Load model
    from nemo.collections.tts.models import MagpieTTSModel
    print(f"Loading model from {model_path}...")
    model = MagpieTTSModel.restore_from(model_path)
    model.to(device)
    model.eval()

    # Get the self-attention module
    sa = model.encoder.layers[0].self_attention
    print(f"\nSelf-Attention module: {type(sa)}")
    print(f"  n_heads: {sa.n_heads}")
    print(f"  d_model: {sa.d_model}")
    print(f"  d_head: {sa.d_head}")

    # Check the weight shapes
    print(f"\n  qkv_net.weight shape: {sa.qkv_net.weight.shape}")
    print(f"  o_net.weight shape: {sa.o_net.weight.shape}")

    # Load input (norm output)
    input_tensor = read_reference(ref_dir / "hook_encoder_layers_0_norm_self.bin")
    # Squeeze to [seq, d_model] then add batch dim to get [1, seq, d_model]
    input_tensor = input_tensor.squeeze()
    if input_tensor.dim() == 1:
        input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.unsqueeze(0).to(device)  # Add batch dim: [1, seq, d_model]
    print(f"\nInput shape: {input_tensor.shape}")

    # Load expected output
    expected = read_reference(ref_dir / "hook_encoder_layers_0_self_attention.bin")
    expected = expected.squeeze()
    print(f"Expected output shape: {expected.shape}")

    # Create capture dict
    captured = {}

    def capture_hook(name):
        def hook(module, args, output):
            if isinstance(output, tuple):
                output = output[0]
            captured[name] = output.detach().clone()
        return hook

    # Hook qkv_net to capture its output
    hook1 = sa.qkv_net.register_forward_hook(capture_hook("qkv_output"))

    # Run self-attention
    print("\nRunning self-attention forward pass...")
    with torch.no_grad():
        # Create mask (all ones for full attention)
        seq_len = input_tensor.shape[1]
        mask = torch.ones(1, seq_len, device=device, dtype=input_tensor.dtype)

        # Forward pass
        output = sa(input_tensor, mask)

    hook1.remove()

    # Get output
    if isinstance(output, dict):
        output = output['output']
    elif isinstance(output, tuple):
        output = output[0]

    print(f"\nNeMo output shape: {output.shape}")
    output_np = output.squeeze(0).cpu().numpy()
    expected_np = expected.cpu().numpy()

    print(f"NeMo first 8: {output_np.flatten()[:8]}")
    print(f"Expected first 8: {expected_np.flatten()[:8]}")

    diff = np.abs(output_np - expected_np)
    print(f"\nNeMo vs Expected: max_diff={diff.max()}, mean_diff={diff.mean()}")

    # Print captured QKV
    if "qkv_output" in captured:
        qkv = captured["qkv_output"]
        print(f"\n=== Captured QKV ===")
        print(f"QKV shape: {qkv.shape}")
        print(f"QKV first 8: {qkv.flatten()[:8].cpu().numpy()}")

        # Save for comparison
        write_tensor(qkv, ref_dir / "debug_qkv_output.bin")

    # Now let's manually trace through the computation
    print("\n=== Manual Trace ===")

    with torch.no_grad():
        x = input_tensor

        # QKV projection
        qkv = sa.qkv_net(x)
        print(f"After qkv_net: {qkv.shape}")
        print(f"  First 8: {qkv.flatten()[:8].cpu().numpy()}")

        # The model reshapes QKV - let's see how
        B, T, _ = qkv.shape
        qkv_reshaped = qkv.reshape(B, T, 3, sa.n_heads, sa.d_head)
        print(f"After reshape to [B, T, 3, n_heads, d_head]: {qkv_reshaped.shape}")

        # Split Q, K, V
        q, k, v = qkv_reshaped.chunk(3, dim=2)
        print(f"Q shape after chunk: {q.shape}")

        q = q.squeeze(2)  # Remove the '1' dimension from chunk
        k = k.squeeze(2)
        v = v.squeeze(2)
        print(f"Q shape after squeeze: {q.shape}")

        # Permute for attention
        q = q.permute(0, 2, 1, 3)  # [B, n_heads, T, d_head]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        print(f"Q shape after permute: {q.shape}")

        # Attention
        scale = 1.0 / (sa.d_head ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        print(f"Attention scores shape: {scores.shape}")
        print(f"  Scores[0,0,0,:8]: {scores[0,0,0,:8].cpu().numpy()}")

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        print(f"  Weights[0,0,0,:8]: {attn_weights[0,0,0,:8].cpu().numpy()}")

        # Weighted values
        attn_out = torch.matmul(attn_weights, v)
        print(f"Attention output shape: {attn_out.shape}")

        # Permute back
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous()
        print(f"After permute back: {attn_out.shape}")

        # Reshape to [B, T, d_model]
        attn_out = attn_out.reshape(B, T, sa.d_model)
        print(f"After reshape: {attn_out.shape}")
        print(f"  First 8: {attn_out.flatten()[:8].cpu().numpy()}")

        # Output projection
        output_manual = sa.o_net(attn_out)
        print(f"After o_net: {output_manual.shape}")
        print(f"  First 8: {output_manual.flatten()[:8].cpu().numpy()}")

        # Compare with expected
        manual_np = output_manual.squeeze(0).cpu().numpy()
        diff = np.abs(manual_np - expected_np)
        print(f"\nManual trace vs Expected: max_diff={diff.max()}, mean_diff={diff.mean()}")


if __name__ == "__main__":
    main()
