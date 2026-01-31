#!/usr/bin/env python3
"""
Dump decoder-specific reference tensors for GGML validation.

Captures intermediate decoder tensors for layer-by-layer testing.
"""

import argparse
import os
import struct
import torch
import numpy as np
from pathlib import Path


def write_tensor_binary(tensor: torch.Tensor, path: str):
    """Write tensor to binary file in GGML-compatible format.

    Format:
    - 4 x int64: dimensions (GGML order - ne[0], ne[1], ne[2], ne[3])
    - data: float32 values in column-major (Fortran) order for GGML compatibility

    GGML uses column-major ordering where the first dimension varies fastest.
    For a tensor with shape [a, b], GGML stores elements as:
        offset = i + j * a  (where i is index in dim 0, j is index in dim 1)

    PyTorch/NumPy use row-major ordering where the last dimension varies fastest.
    For a tensor with shape [a, b], NumPy stores elements as:
        offset = i * b + j

    To convert, we flatten with Fortran order ('F') which outputs elements
    with the first index varying fastest.
    """
    data = tensor.detach().float().cpu().numpy()
    shape = list(data.shape)

    # Pad shape to 4 dimensions
    while len(shape) < 4:
        shape.append(1)

    with open(path, 'wb') as f:
        # Write dimensions in GGML order (ne[0], ne[1], ne[2], ne[3])
        for dim in shape[:4]:
            f.write(struct.pack('<q', dim))
        # Flatten in Fortran (column-major) order for GGML
        # This makes element [i,j] appear at file offset i + j*shape[0]
        f.write(data.flatten('F').tobytes())

    print(f"  Wrote {path}: shape={list(tensor.shape)}, size={os.path.getsize(path)} bytes")


def load_magpie_model(model_path: str, device: str = "cuda"):
    """Load Magpie TTS model."""
    from nemo.collections.tts.models import MagpieTTSModel

    print(f"Loading model from {model_path}...")
    model = MagpieTTSModel.restore_from(model_path)
    model.use_kv_cache_for_inference = False  # Disable KV cache for simpler testing
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return model


def dump_decoder_layer_reference(model, output_dir: str, device: str):
    """Dump reference data for decoder layer testing."""
    print("\n=== Dumping Decoder Layer Reference Data ===\n")

    # Use fixed test inputs
    text = "Hello world"
    speaker_id = 0

    # Tokenize text
    tokenizer = model.tokenizer
    if hasattr(tokenizer, 'tokenizers'):
        token_name = list(tokenizer.tokenizers.keys())[0]
        text_tokens = tokenizer.tokenizers[token_name](text)
    else:
        text_tokens = tokenizer(text)

    text_tokens = [model.bos_id] + text_tokens + [model.eos_id]
    tokens = torch.tensor([text_tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        # === 1. Get encoder output ===
        embedded = model.text_embedding(tokens)
        seq_len = embedded.shape[1]
        x_mask = torch.ones(1, seq_len, device=device, dtype=embedded.dtype)

        pos_emb = model.encoder.position_embeddings.weight[:seq_len].unsqueeze(0)
        x = embedded + pos_emb

        for layer in model.encoder.layers:
            x_out = layer(x, x_mask)
            x = x_out['output'] if isinstance(x_out, dict) else x_out[0] if isinstance(x_out, tuple) else x_out

        encoder_output = model.encoder.norm_out(x)
        write_tensor_binary(encoder_output.squeeze(0).T, os.path.join(output_dir, "dec_encoder_output.bin"))
        print(f"  Encoder output shape: {encoder_output.shape} -> saved as [d_model, seq]")

        # === 2. Create decoder input (baked context + BOS) ===
        baked_flat = model.baked_context_embedding.weight[speaker_id]
        T_ctx, D = 110, 768
        context = baked_flat.view(T_ctx, D).unsqueeze(0)

        # BOS audio embedding
        audio_codes = torch.full((1, 8, 1), model.audio_bos_id, dtype=torch.long, device=device)
        audio_emb = sum(model.audio_embeddings[cb](audio_codes[:, cb, :]) for cb in range(8))

        # Decoder input = [context; audio_emb]
        dec_input = torch.cat([context, audio_emb], dim=1)  # [1, 111, 768]
        dec_len = dec_input.shape[1]

        # Add position embeddings
        dec_pos = model.decoder.position_embeddings.weight[:dec_len].unsqueeze(0)
        dec_with_pos = dec_input + dec_pos

        # Save decoder input (after position embeddings)
        write_tensor_binary(dec_with_pos.squeeze(0).T, os.path.join(output_dir, "dec_input.bin"))
        print(f"  Decoder input shape: {dec_with_pos.shape} -> saved as [d_model, seq]")

        # === 3. Dump decoder layer 0 intermediate tensors ===
        layer0 = model.decoder.layers[0]
        x_dec = dec_with_pos

        # Pre-self-attention norm
        norm1_out = layer0.norm_self(x_dec)
        write_tensor_binary(norm1_out.squeeze(0).T, os.path.join(output_dir, "dec_l0_norm_self.bin"))

        # Self-attention (need to call the module directly with proper mask)
        # Create causal mask for decoder
        dec_mask = torch.ones(1, dec_len, device=device, dtype=x_dec.dtype)

        # Call self-attention layer
        sa_out = layer0.self_attention(norm1_out, dec_mask)
        if isinstance(sa_out, dict):
            sa_out = sa_out['output']
        elif isinstance(sa_out, tuple):
            sa_out = sa_out[0]
        write_tensor_binary(sa_out.squeeze(0).T, os.path.join(output_dir, "dec_l0_self_attn.bin"))

        # Residual
        x_after_sa = x_dec + sa_out
        write_tensor_binary(x_after_sa.squeeze(0).T, os.path.join(output_dir, "dec_l0_after_sa.bin"))

        # Pre-cross-attention norm (query)
        norm_xa_q = layer0.norm_xattn_query(x_after_sa)
        write_tensor_binary(norm_xa_q.squeeze(0).T, os.path.join(output_dir, "dec_l0_norm_xa_q.bin"))

        # Pre-cross-attention norm (memory/encoder output)
        enc_mask = torch.ones(1, seq_len, device=device, dtype=encoder_output.dtype)
        norm_xa_mem = layer0.norm_xattn_memory(encoder_output)
        write_tensor_binary(norm_xa_mem.squeeze(0).T, os.path.join(output_dir, "dec_l0_norm_xa_mem.bin"))

        # Cross-attention
        xa_out = layer0.cross_attention(norm_xa_q, enc_mask, norm_xa_mem, enc_mask)
        if isinstance(xa_out, dict):
            xa_out = xa_out['output']
        elif isinstance(xa_out, tuple):
            xa_out = xa_out[0]
        write_tensor_binary(xa_out.squeeze(0).T, os.path.join(output_dir, "dec_l0_cross_attn.bin"))

        # Residual
        x_after_xa = x_after_sa + xa_out
        write_tensor_binary(x_after_xa.squeeze(0).T, os.path.join(output_dir, "dec_l0_after_xa.bin"))

        # Pre-FFN norm
        norm_ff = layer0.norm_pos_ff(x_after_xa)
        write_tensor_binary(norm_ff.squeeze(0).T, os.path.join(output_dir, "dec_l0_norm_ff.bin"))

        # FFN (pos_ff)
        ff_out = layer0.pos_ff(norm_ff, dec_mask)
        if isinstance(ff_out, dict):
            ff_out = ff_out['output']
        elif isinstance(ff_out, tuple):
            ff_out = ff_out[0]
        write_tensor_binary(ff_out.squeeze(0).T, os.path.join(output_dir, "dec_l0_ff_out.bin"))

        # Full layer output
        layer0_out = x_after_xa + ff_out
        write_tensor_binary(layer0_out.squeeze(0).T, os.path.join(output_dir, "dec_l0_out.bin"))

        # === 4. Run through all decoder layers ===
        x = dec_with_pos
        for i, layer in enumerate(model.decoder.layers):
            layer_out = layer(x, dec_mask, encoder_output, enc_mask)
            if isinstance(layer_out, dict):
                x = layer_out['output']
            elif isinstance(layer_out, tuple):
                x = layer_out[0]
            else:
                x = layer_out
            if i < 3:  # Save first few layers
                write_tensor_binary(x.squeeze(0).T, os.path.join(output_dir, f"dec_l{i}_out_full.bin"))

        # Final norm
        dec_output = model.decoder.norm_out(x)
        write_tensor_binary(dec_output.squeeze(0).T, os.path.join(output_dir, "dec_output.bin"))
        print(f"  Decoder output shape: {dec_output.shape}")

        # === 5. Final projection (last frame only) ===
        last_frame = dec_output[:, -1:, :]  # [1, 1, 768]
        logits = model.final_proj(last_frame)  # [1, 1, 16192]
        write_tensor_binary(logits.squeeze(0).T, os.path.join(output_dir, "dec_logits.bin"))
        print(f"  Logits shape: {logits.shape}")

    print("\n=== Decoder Reference Data Complete ===")


def dump_cross_attention_reference(model, output_dir: str, device: str):
    """Dump detailed cross-attention reference data."""
    print("\n=== Dumping Cross-Attention Reference Data ===\n")

    # Use simple fixed inputs
    d_model = 768
    dec_seq = 5  # Small decoder sequence for testing
    enc_seq = 8  # Small encoder sequence
    d_xa_head = 128  # Cross-attention head dimension
    n_xa_heads = 1

    # Create test inputs
    query = torch.randn(1, dec_seq, d_model, device=device)
    memory = torch.randn(1, enc_seq, d_model, device=device)

    write_tensor_binary(query.squeeze(0).T, os.path.join(output_dir, "xa_query_input.bin"))
    write_tensor_binary(memory.squeeze(0).T, os.path.join(output_dir, "xa_memory_input.bin"))

    layer0 = model.decoder.layers[0]

    with torch.no_grad():
        # Get cross-attention weights
        q_w = layer0.cross_attention.q_net.weight  # [128, 768]
        kv_w = layer0.cross_attention.kv_net.weight  # [256, 768]
        o_w = layer0.cross_attention.o_net.weight  # [768, 128]

        write_tensor_binary(q_w.T, os.path.join(output_dir, "xa_q_weight.bin"))
        write_tensor_binary(kv_w.T, os.path.join(output_dir, "xa_kv_weight.bin"))
        write_tensor_binary(o_w.T, os.path.join(output_dir, "xa_o_weight.bin"))

        # Compute Q, K, V
        Q = torch.matmul(query, q_w.T)  # [1, dec_seq, 128]
        KV = torch.matmul(memory, kv_w.T)  # [1, enc_seq, 256]
        K = KV[:, :, :d_xa_head]  # [1, enc_seq, 128]
        V = KV[:, :, d_xa_head:]  # [1, enc_seq, 128]

        write_tensor_binary(Q.squeeze(0).T, os.path.join(output_dir, "xa_Q.bin"))
        write_tensor_binary(K.squeeze(0).T, os.path.join(output_dir, "xa_K.bin"))
        write_tensor_binary(V.squeeze(0).T, os.path.join(output_dir, "xa_V.bin"))

        # Attention: Q @ K.T / sqrt(d)
        scale = 1.0 / (d_xa_head ** 0.5)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [1, dec_seq, enc_seq]
        write_tensor_binary(scores.squeeze(0), os.path.join(output_dir, "xa_scores.bin"))

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        write_tensor_binary(attn_weights.squeeze(0), os.path.join(output_dir, "xa_weights.bin"))

        # Attend to values
        attn_out = torch.matmul(attn_weights, V)  # [1, dec_seq, 128]
        write_tensor_binary(attn_out.squeeze(0).T, os.path.join(output_dir, "xa_attn_out.bin"))

        # Output projection
        output = torch.matmul(attn_out, o_w.T)  # [1, dec_seq, 768]
        write_tensor_binary(output.squeeze(0).T, os.path.join(output_dir, "xa_output.bin"))

        print(f"  Q shape: {Q.shape}")
        print(f"  K shape: {K.shape}")
        print(f"  V shape: {V.shape}")
        print(f"  Output shape: {output.shape}")

    print("\n=== Cross-Attention Reference Data Complete ===")


def main():
    parser = argparse.ArgumentParser(description="Dump decoder reference tensors")
    parser.add_argument(
        "--model",
        type=str,
        default="../magpie_tts_multilingual_357m/magpie_tts_multilingual_357m.nemo",
        help="Path to Magpie TTS model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_data/reference",
        help="Output directory for reference tensors"
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, using CPU")

    model = load_magpie_model(args.model, device)

    dump_decoder_layer_reference(model, args.output_dir, device)
    dump_cross_attention_reference(model, args.output_dir, device)


if __name__ == "__main__":
    main()
