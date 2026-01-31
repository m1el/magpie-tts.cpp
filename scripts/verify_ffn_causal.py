#!/usr/bin/env python3
"""Verify FFN causal padding behavior."""

import struct
import torch
import numpy as np
from pathlib import Path


def read_tensor(path):
    with open(path, 'rb') as f:
        shape = [struct.unpack('<q', f.read(8))[0] for _ in range(4)]
        data = np.frombuffer(f.read(), dtype=np.float32).copy()
        # GGML stores col-major for shape [d_model, seq] = [768, 14]
        # Data[k] = element[k % 768, k // 768] in GGML = element[t, d] where t=k//768, d=k%768
        # For PyTorch [seq, d_model] row-major, we need arr[t, d] = data[t*768 + d]
        # This equals data[k] when k = t*768 + d, matching GGML's k = d + t*768
        # So just reshape directly to [seq, d_model]
        seq_len = shape[2]  # 14
        d_model = shape[1]  # 768
        data = data.reshape(seq_len, d_model)  # [14, 768]
        return torch.tensor(data, dtype=torch.float32)


def write_tensor(tensor, path):
    """Write tensor matching the format of dump_reference.py.

    For a [seq, d_model] = [14, 768] tensor:
    - Add batch dim: [1, 14, 768]
    - Pad to 4D: [1, 14, 768, 1]
    - Write reversed: [1, 768, 14, 1]
    """
    data = tensor.detach().float().cpu().numpy()
    # If 2D [seq, d_model], add batch dim to get [1, seq, d_model]
    if data.ndim == 2:
        data = data[np.newaxis, ...]  # [1, seq, d_model]
    shape = list(data.shape)
    while len(shape) < 4:
        shape.append(1)
    with open(path, 'wb') as f:
        for dim in reversed(shape[:4]):
            f.write(struct.pack('<q', dim))
        f.write(data.tobytes())


def main():
    from nemo.collections.tts.models import MagpieTTSModel

    model_path = "../magpie_tts_multilingual_357m/magpie_tts_multilingual_357m.nemo"
    print(f"Loading model...")
    model = MagpieTTSModel.restore_from(model_path)
    model.eval()
    model.cuda()

    output_dir = Path("test_data/reference")

    # Load layer 0 output as input
    layer0_out = read_tensor(output_dir / "manual_enc_layer0_out.bin").cuda()
    print(f"layer0_out shape: {layer0_out.shape}")  # [14, 768]

    enc_layer = model.encoder.layers[1]
    pos_ff = enc_layer.pos_ff

    with torch.no_grad():
        x = layer0_out.unsqueeze(0)  # [1, 14, 768]
        seq_len = x.shape[1]

        # Create full mask (all ones)
        mask = torch.ones(1, seq_len, device='cuda')

        # Step by step with all intermediate saves
        # Step 1: norm before self-attention
        norm1_out = enc_layer.norm_self(x)
        write_tensor(norm1_out[0], str(output_dir / "debug_l1_norm1.bin"))
        print(f"norm1_out shape: {norm1_out.shape}")

        # Step 2: self-attention
        sa_out = enc_layer.self_attention(norm1_out)
        if isinstance(sa_out, tuple):
            sa_out = sa_out[0]
        write_tensor(sa_out[0], str(output_dir / "debug_l1_sa.bin"))
        print(f"sa_out shape: {sa_out.shape}")

        # Step 3: residual 1
        res1 = x + sa_out
        write_tensor(res1[0], str(output_dir / "debug_l1_res1.bin"))
        print(f"res1 shape: {res1.shape}")

        # Step 4: norm before FFN
        norm2_out = enc_layer.norm_pos_ff(res1)
        write_tensor(norm2_out[0], str(output_dir / "debug_l1_norm2.bin"))
        print(f"norm2_out shape: {norm2_out.shape}")

        # Step 5: FFN
        ffn_out = pos_ff(norm2_out, mask)
        write_tensor(ffn_out[0], str(output_dir / "debug_l1_ffn_correct.bin"))
        print(f"ffn_out shape: {ffn_out.shape}")

        # Step 6: residual 2 (final output)
        res2 = res1 + ffn_out

        print(f"\nFull layer 1 output shape: {res2.shape}")
        print(f"res2[:5, :5]:\n{res2[0, :5, :5]}")

        # Load expected layer1 output
        layer1_expected = read_tensor(output_dir / "manual_enc_layer1_out.bin").cuda()
        print(f"\nExpected layer1 shape: {layer1_expected.shape}")

        diff = (res2[0] - layer1_expected).abs()
        print(f"Max diff to expected: {diff.max().item():.6f}")
        print(f"Mean diff: {diff.mean().item():.6f}")

        # Check position 925 = token 1, dim 157
        print(f"\nPosition 925 (token 1, dim 157):")
        print(f"  Computed: {res2[0, 1, 157].item():.6f}")
        print(f"  Expected: {layer1_expected[1, 157].item():.6f}")


if __name__ == "__main__":
    main()
