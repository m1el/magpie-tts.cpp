#!/usr/bin/env python3
"""
Dump local transformer reference tensors for GGML validation.

The local transformer autoregressively predicts codes for each codebook
within a single audio frame.
"""

import argparse
import os
import struct
import torch
import numpy as np
from pathlib import Path


def write_tensor_binary(tensor: torch.Tensor, path: str):
    """Write tensor to binary file in GGML-compatible format."""
    data = tensor.detach().float().cpu().numpy()
    shape = list(data.shape)
    while len(shape) < 4:
        shape.append(1)
    with open(path, 'wb') as f:
        for dim in shape[:4]:
            f.write(struct.pack('<q', dim))
        f.write(data.flatten('F').tobytes())
    print(f"  Wrote {path}: shape={list(tensor.shape)}, size={os.path.getsize(path)} bytes")


def load_magpie_model(model_path: str, device: str = "cuda"):
    """Load Magpie TTS model."""
    from nemo.collections.tts.models import MagpieTTSModel
    print(f"Loading model from {model_path}...")
    model = MagpieTTSModel.restore_from(model_path)
    model.use_kv_cache_for_inference = False
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return model


def dump_local_transformer_weights(model, output_dir: str, device: str):
    """Dump local transformer weights for verification."""
    print("\n=== Dumping Local Transformer Weights ===\n")

    with torch.no_grad():
        # Input projection weights
        in_proj = model.local_transformer_in_projection
        write_tensor_binary(in_proj.weight.T, os.path.join(output_dir, "lt_in_proj_w.bin"))
        write_tensor_binary(in_proj.bias, os.path.join(output_dir, "lt_in_proj_b.bin"))
        print(f"  in_projection: {in_proj.weight.shape}")

        # Position embeddings
        pos_emb = model.local_transformer.position_embeddings.weight
        write_tensor_binary(pos_emb.T, os.path.join(output_dir, "lt_pos_emb.bin"))
        print(f"  position_embeddings: {pos_emb.shape}")

        # Output projections (8 codebooks)
        for cb in range(8):
            out_proj = model.local_transformer_out_projections[cb]
            write_tensor_binary(out_proj.weight.T, os.path.join(output_dir, f"lt_out_proj_{cb}_w.bin"))
            write_tensor_binary(out_proj.bias, os.path.join(output_dir, f"lt_out_proj_{cb}_b.bin"))
        print(f"  out_projections: 8 x {out_proj.weight.shape}")


def dump_local_transformer_reference(model, output_dir: str, device: str):
    """Dump reference data for local transformer testing."""
    print("\n=== Dumping Local Transformer Reference Data ===\n")

    text = "Hello world"
    speaker_id = 0

    # Tokenize
    tokenizer = model.tokenizer
    if hasattr(tokenizer, 'tokenizers'):
        token_name = list(tokenizer.tokenizers.keys())[0]
        text_tokens = tokenizer.tokenizers[token_name](text)
    else:
        text_tokens = tokenizer(text)

    text_tokens = [model.bos_id] + text_tokens + [model.eos_id]
    tokens = torch.tensor([text_tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        # === Get encoder output ===
        embedded = model.text_embedding(tokens)
        seq_len = embedded.shape[1]
        x_mask = torch.ones(1, seq_len, device=device, dtype=embedded.dtype)
        pos_emb = model.encoder.position_embeddings.weight[:seq_len].unsqueeze(0)
        x = embedded + pos_emb

        for layer in model.encoder.layers:
            x_out = layer(x, x_mask)
            x = x_out['output'] if isinstance(x_out, dict) else x_out[0] if isinstance(x_out, tuple) else x_out
        encoder_output = model.encoder.norm_out(x)

        # === Prepare decoder input ===
        baked_flat = model.baked_context_embedding.weight[speaker_id]
        T_ctx, D = 110, 768
        context = baked_flat.view(T_ctx, D).unsqueeze(0)

        audio_codes = torch.full((1, 8, 1), model.audio_bos_id, dtype=torch.long, device=device)
        audio_emb = sum(model.audio_embeddings[cb](audio_codes[:, cb, :]) for cb in range(8))

        dec_input = torch.cat([context, audio_emb], dim=1)
        dec_len = dec_input.shape[1]
        dec_pos = model.decoder.position_embeddings.weight[:dec_len].unsqueeze(0)
        dec_with_pos = dec_input + dec_pos

        # === Run through decoder ===
        dec_mask = torch.ones(1, dec_len, device=device, dtype=dec_with_pos.dtype)
        enc_mask = torch.ones(1, seq_len, device=device, dtype=encoder_output.dtype)

        x = dec_with_pos
        for layer in model.decoder.layers:
            layer_out = layer(x, dec_mask, encoder_output, enc_mask)
            if isinstance(layer_out, dict):
                x = layer_out['output']
            elif isinstance(layer_out, tuple):
                x = layer_out[0]
            else:
                x = layer_out
        dec_output = model.decoder.norm_out(x)

        # === Get last frame for local transformer ===
        last_frame = dec_output[:, -1, :]  # [1, 768]
        write_tensor_binary(last_frame.squeeze(0), os.path.join(output_dir, "lt_dec_hidden.bin"))
        print(f"  Decoder hidden (last frame): {last_frame.shape}")

        # === Local transformer step-by-step ===
        # Step 1: Input projection
        lt_input = model.local_transformer_in_projection(last_frame)  # [1, 256]
        write_tensor_binary(lt_input.squeeze(0), os.path.join(output_dir, "lt_input_projected.bin"))
        print(f"  After input projection: {lt_input.shape}")

        # Step 2: Add position embedding for position 0
        lt_pos = model.local_transformer.position_embeddings.weight[0:1].unsqueeze(0)  # [1, 1, 256]
        lt_with_pos = lt_input.unsqueeze(1) + lt_pos  # [1, 1, 256]
        write_tensor_binary(lt_with_pos.squeeze(0).squeeze(0), os.path.join(output_dir, "lt_pos0_input.bin"))
        print(f"  After pos embedding: {lt_with_pos.shape}")

        # Step 3: Run through local transformer layer
        lt_mask = torch.ones(1, 1, device=device, dtype=lt_with_pos.dtype)
        layer = model.local_transformer.layers[0]

        # Pre-norm for self-attention
        norm_self = layer.norm_self(lt_with_pos)
        write_tensor_binary(norm_self.squeeze(0).T, os.path.join(output_dir, "lt_norm_self.bin"))

        # Self-attention (just 1 position, so trivial)
        sa_out = layer.self_attention(norm_self, lt_mask)
        if isinstance(sa_out, dict):
            sa_out = sa_out['output']
        elif isinstance(sa_out, tuple):
            sa_out = sa_out[0]
        write_tensor_binary(sa_out.squeeze(0).T, os.path.join(output_dir, "lt_self_attn.bin"))

        # Residual
        after_sa = lt_with_pos + sa_out
        write_tensor_binary(after_sa.squeeze(0).T, os.path.join(output_dir, "lt_after_sa.bin"))

        # Pre-norm for FFN
        norm_ff = layer.norm_pos_ff(after_sa)
        write_tensor_binary(norm_ff.squeeze(0).T, os.path.join(output_dir, "lt_norm_ff.bin"))

        # FFN
        ff_out = layer.pos_ff(norm_ff, lt_mask)
        if isinstance(ff_out, dict):
            ff_out = ff_out['output']
        elif isinstance(ff_out, tuple):
            ff_out = ff_out[0]
        write_tensor_binary(ff_out.squeeze(0).T, os.path.join(output_dir, "lt_ff_out.bin"))

        # Layer output
        lt_layer_out = after_sa + ff_out  # [1, 1, 256]
        write_tensor_binary(lt_layer_out.squeeze(0).T, os.path.join(output_dir, "lt_layer_out.bin"))
        print(f"  Layer output: {lt_layer_out.shape}")

        # Step 4: Output projection for codebook 0
        logits_cb0 = model.local_transformer_out_projections[0](lt_layer_out[:, -1, :])  # [1, 2024]
        write_tensor_binary(logits_cb0.squeeze(0), os.path.join(output_dir, "lt_logits_cb0.bin"))
        print(f"  Logits codebook 0: {logits_cb0.shape}")

        # === Full autoregressive sequence ===
        # Run full local transformer autoregressive inference for all 8 codebooks
        print("\n  Running full autoregressive local transformer...")

        # Start with projected decoder hidden
        lt_seq = lt_input.unsqueeze(1)  # [1, 1, 256]

        all_logits = []
        sampled_codes = []

        for cb in range(8):
            # Add position embeddings
            seq_len_lt = lt_seq.shape[1]
            pos_embs = model.local_transformer.position_embeddings.weight[:seq_len_lt].unsqueeze(0)
            lt_with_pos = lt_seq + pos_embs

            # Run transformer layer
            lt_mask = torch.ones(1, seq_len_lt, device=device, dtype=lt_with_pos.dtype)

            x = lt_with_pos
            for layer in model.local_transformer.layers:
                layer_out = layer(x, lt_mask)
                if isinstance(layer_out, dict):
                    x = layer_out['output']
                elif isinstance(layer_out, tuple):
                    x = layer_out[0]
                else:
                    x = layer_out

            # Get logits for this codebook from last position
            last_hidden = x[:, -1, :]  # [1, 256]
            logits = model.local_transformer_out_projections[cb](last_hidden)  # [1, 2024]
            all_logits.append(logits)

            # Sample (argmax for deterministic testing)
            sampled_code = logits.argmax(dim=-1)  # [1]
            sampled_codes.append(sampled_code.item())

            # If not last codebook, embed sampled code and append
            if cb < 7:
                # Use audio embedding for the sampled code
                code_emb = model.audio_embeddings[cb](sampled_code.unsqueeze(-1))  # [1, 1, 768]
                # Project to local transformer dimension
                code_lt = model.local_transformer_in_projection(code_emb)  # [1, 1, 256]
                lt_seq = torch.cat([lt_seq, code_lt], dim=1)  # [1, seq+1, 256]

        # Save all logits
        for cb in range(8):
            write_tensor_binary(all_logits[cb].squeeze(0), os.path.join(output_dir, f"lt_full_logits_cb{cb}.bin"))

        print(f"  Sampled codes (argmax): {sampled_codes}")

        # Save intermediate hidden states for testing
        # Run again with first 3 positions to save intermediate states
        lt_seq_test = lt_input.unsqueeze(1)  # [1, 1, 256]
        for cb in range(3):
            # Get logits and sample
            seq_len_lt = lt_seq_test.shape[1]
            pos_embs = model.local_transformer.position_embeddings.weight[:seq_len_lt].unsqueeze(0)
            lt_with_pos = lt_seq_test + pos_embs
            lt_mask = torch.ones(1, seq_len_lt, device=device, dtype=lt_with_pos.dtype)

            x = lt_with_pos
            for layer in model.local_transformer.layers:
                layer_out = layer(x, lt_mask)
                if isinstance(layer_out, dict):
                    x = layer_out['output']
                elif isinstance(layer_out, tuple):
                    x = layer_out[0]
                else:
                    x = layer_out

            # Save hidden state after transformer
            write_tensor_binary(x.squeeze(0).T, os.path.join(output_dir, f"lt_hidden_seq{seq_len_lt}.bin"))

            logits = model.local_transformer_out_projections[cb](x[:, -1, :])
            sampled_code = logits.argmax(dim=-1)

            if cb < 2:
                code_emb = model.audio_embeddings[cb](sampled_code.unsqueeze(-1))
                code_lt = model.local_transformer_in_projection(code_emb)
                lt_seq_test = torch.cat([lt_seq_test, code_lt], dim=1)

    print("\n=== Local Transformer Reference Data Complete ===")


def main():
    parser = argparse.ArgumentParser(description="Dump local transformer reference tensors")
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

    dump_local_transformer_weights(model, args.output_dir, device)
    dump_local_transformer_reference(model, args.output_dir, device)


if __name__ == "__main__":
    main()
