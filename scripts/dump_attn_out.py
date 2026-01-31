#!/usr/bin/env python3
"""Dump attention output before o_net projection."""

import struct
import torch
import numpy as np
from pathlib import Path


def read_tensor(path):
    with open(path, 'rb') as f:
        shape = [struct.unpack('<q', f.read(8))[0] for _ in range(4)]
        data = np.frombuffer(f.read(), dtype=np.float32).copy()
        seq_len = shape[2]
        d_model = shape[1]
        data = data.reshape(seq_len, d_model)
        return torch.tensor(data, dtype=torch.float32)


def write_tensor(tensor, path):
    data = tensor.detach().float().cpu().numpy()
    if data.ndim == 2:
        data = data[np.newaxis, ...]
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
    layer0_out = read_tensor(output_dir / "manual_enc_layer0_out.bin").cuda()

    enc_layer = model.encoder.layers[1]
    sa = enc_layer.self_attention

    with torch.no_grad():
        x = layer0_out.unsqueeze(0)
        seq_len = x.shape[1]
        d_model = 768
        n_heads = 12
        d_head = 64

        # Norm
        norm1_out = enc_layer.norm_self(x)

        # QKV
        qkv = sa.qkv_net(norm1_out)
        q, k, v = qkv.split(d_model, dim=-1)

        # Reshape to [B, T, H, D_head] then [B, H, T, D_head]
        q = q.view(1, seq_len, n_heads, d_head).transpose(1, 2)
        k = k.view(1, seq_len, n_heads, d_head).transpose(1, 2)
        v = v.view(1, seq_len, n_heads, d_head).transpose(1, 2)

        # Compute attention
        scale = 1.0 / (d_head ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device='cuda'), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        # Reshape back: [B, H, T, D_head] -> [B, T, H, D_head] -> [B, T, D]
        attn_out_reshaped = attn_out.transpose(1, 2).contiguous().view(1, seq_len, d_model)

        print(f"attn_out_reshaped shape: {attn_out_reshaped.shape}")
        write_tensor(attn_out_reshaped[0], str(output_dir / "debug_l1_attn_out.bin"))

        # Output projection
        sa_out = sa.o_net(attn_out_reshaped)

        print(f"\nPosition 7322 (token 9, dim 410):")
        t, d = 9, 410
        print(f"  attn_out_reshaped[{t}, {d}] = {attn_out_reshaped[0, t, d].item():.6f}")
        print(f"  sa_out[{t}, {d}] = {sa_out[0, t, d].item():.6f}")

        # Check the attention patterns for token 9
        print(f"\nAttention weights for token 9, head 0:")
        print(f"  {attn_weights[0, 0, 9, :]}")


if __name__ == "__main__":
    main()
