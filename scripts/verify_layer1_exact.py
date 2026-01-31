#!/usr/bin/env python3
"""Exactly reproduce layer 1 output from reference input."""

import struct
import torch
import numpy as np

def read_tensor_pytorch(path, device='cuda'):
    """Read tensor directly into PyTorch format [B, T, D]."""
    with open(path, 'rb') as f:
        shape = [struct.unpack('<q', f.read(8))[0] for _ in range(4)]
        data = np.frombuffer(f.read(), dtype=np.float32).copy()

    # GGML stores data in column-major order for shape [d_model, seq] = [768, 14]
    # So data[k] corresponds to GGML element [k % 768, k // 768] = [d, t]
    # For PyTorch [T, D] = [seq, d_model] = [14, 768]:
    # We need arr[t, d] = data[t*768 + d]
    # This is exactly what numpy row-major reshape(14, 768) gives us
    seq_len = shape[2]  # 14
    d_model = shape[1]  # 768
    data = data.reshape(seq_len, d_model)  # [14, 768]
    return torch.tensor(data, device=device).unsqueeze(0)  # [1, 14, 768]


def main():
    from nemo.collections.tts.models import MagpieTTSModel

    model_path = "../magpie_tts_multilingual_357m/magpie_tts_multilingual_357m.nemo"
    print(f"Loading model...")
    model = MagpieTTSModel.restore_from(model_path)
    model.eval()
    model.cuda()

    # Load reference tensors
    layer0_out = read_tensor_pytorch("test_data/reference/manual_enc_layer0_out.bin")
    layer1_expected = read_tensor_pytorch("test_data/reference/manual_enc_layer1_out.bin")

    print(f"layer0_out shape: {layer0_out.shape}")
    print(f"layer1_expected shape: {layer1_expected.shape}")
    print(f"layer0_out[0, 1, 157] = {layer0_out[0, 1, 157].item():.6f}")
    print(f"layer1_expected[0, 1, 157] = {layer1_expected[0, 1, 157].item():.6f}")

    enc_layer = model.encoder.layers[1]
    seq_len = layer0_out.shape[1]
    x_mask = torch.ones(1, seq_len, device='cuda')

    with torch.no_grad():
        # Run layer 1
        x_out = enc_layer(layer0_out, x_mask)
        if isinstance(x_out, dict):
            layer1_computed = x_out['output']
        elif isinstance(x_out, tuple):
            layer1_computed = x_out[0]
        else:
            layer1_computed = x_out

    print(f"\nlayer1_computed shape: {layer1_computed.shape}")
    print(f"layer1_computed[0, 1, 157] = {layer1_computed[0, 1, 157].item():.6f}")

    diff = (layer1_computed - layer1_expected).abs()
    print(f"\nMax diff: {diff.max().item():.6f}")
    print(f"Mean diff: {diff.mean().item():.6f}")

    # Find max diff position
    max_idx = diff.argmax()
    t = max_idx.item() // 768
    d = max_idx.item() % 768
    print(f"Max diff at token {t}, dim {d}")
    print(f"  Computed: {layer1_computed[0, t, d].item():.6f}")
    print(f"  Expected: {layer1_expected[0, t, d].item():.6f}")


if __name__ == "__main__":
    main()
