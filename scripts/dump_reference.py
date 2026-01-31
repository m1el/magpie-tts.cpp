#!/usr/bin/env python3
"""
Dump reference tensors from PyTorch Magpie TTS model for GGML validation.

Uses hooks to capture actual intermediate tensors during inference,
similar to inspect_inference.py.

Usage:
    uv run scripts/dump_reference.py --text "Hello world" --output-dir test_data/reference
"""

import argparse
import os
import struct
import torch
import numpy as np
from pathlib import Path
from collections import OrderedDict


def write_tensor_binary(tensor: torch.Tensor, path: str):
    """Write tensor to binary file in GGML-compatible format.

    Format:
    - 4 x int64: dimensions (padded to 4, GGML reversed order)
    - data: float32 values in row-major order
    """
    data = tensor.detach().float().cpu().numpy()
    shape = list(data.shape)

    # Pad shape to 4 dimensions
    while len(shape) < 4:
        shape.append(1)

    with open(path, 'wb') as f:
        # Write dimensions (GGML uses reversed order)
        for dim in reversed(shape[:4]):
            f.write(struct.pack('<q', dim))
        # Write data
        f.write(data.tobytes())

    print(f"  Wrote {path}: shape={list(tensor.shape)}, size={os.path.getsize(path)} bytes")


class TensorCapture:
    """Capture tensors from specified module paths during forward pass."""

    def __init__(self, model, capture_patterns: list[str]):
        """
        Args:
            model: The PyTorch model
            capture_patterns: List of module path patterns to capture
                e.g. ["text_embedding", "encoder.layers.0.self_attention"]
        """
        self.captures = OrderedDict()
        self.hooks = []
        self.capture_patterns = capture_patterns

        # Register hooks on matching modules
        for name, module in model.named_modules():
            for pattern in capture_patterns:
                if name == pattern or name.startswith(pattern + "."):
                    if name == pattern:  # Exact match - capture output
                        hook = module.register_forward_hook(self._make_hook(name))
                        self.hooks.append(hook)
                        break

    def _make_hook(self, name):
        def hook(module, args, output):
            # Handle different output types
            if isinstance(output, dict):
                # Transformer layers return dict with 'output' key
                if 'output' in output:
                    output = output['output']
                else:
                    return  # Skip if no 'output' key
            elif isinstance(output, tuple):
                output = output[0]

            if torch.is_tensor(output):
                self.captures[name] = output.clone()
        return hook

    def clear(self):
        self.captures.clear()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def load_magpie_model(model_path: str, device: str = "cuda"):
    """Load Magpie TTS model."""
    from nemo.collections.tts.models import MagpieTTSModel

    print(f"Loading model from {model_path}...")
    model = MagpieTTSModel.restore_from(model_path)
    model.use_kv_cache_for_inference = True
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return model


def dump_with_hooks(model, text: str, speaker_id: int, output_dir: str, device: str):
    """Run inference with hooks to capture intermediate tensors."""
    print(f"\n{'='*60}")
    print(f"Capturing tensors for: '{text}'")
    print(f"{'='*60}")

    # Define what to capture
    capture_patterns = [
        # Text embedding
        "text_embedding",

        # Encoder layers
        "encoder",
        "encoder.layers.0",
        "encoder.layers.0.norm_self",
        "encoder.layers.0.self_attention",
        "encoder.layers.0.norm_pos_ff",
        "encoder.layers.0.pos_ff",
        "encoder.layers.1",
        "encoder.layers.2",
        "encoder.layers.3",
        "encoder.layers.4",
        "encoder.layers.5",
        "encoder.norm_out",

        # Audio embeddings
        "audio_embeddings.0",
        "audio_embeddings.1",
        "audio_embeddings.2",
        "audio_embeddings.3",
        "audio_embeddings.4",
        "audio_embeddings.5",
        "audio_embeddings.6",
        "audio_embeddings.7",

        # Decoder layers
        "decoder",
        "decoder.layers.0",
        "decoder.layers.0.norm_self",
        "decoder.layers.0.self_attention",
        "decoder.layers.0.norm_xattn_query",
        "decoder.layers.0.cross_attention",
        "decoder.layers.0.norm_pos_ff",
        "decoder.layers.0.pos_ff",
        "decoder.norm_out",

        # Final projection
        "final_proj",

        # Local transformer
        "local_transformer_in_projection",
        "local_transformer",
        "local_transformer_out_projections.0",
        "local_transformer_out_projections.1",
        "local_transformer_out_projections.2",
        "local_transformer_out_projections.3",
        "local_transformer_out_projections.4",
        "local_transformer_out_projections.5",
        "local_transformer_out_projections.6",
        "local_transformer_out_projections.7",

        # Baked context
        "baked_context_embedding",
    ]

    # Setup capture hooks
    capture = TensorCapture(model, capture_patterns)

    # Create batch
    tokenizer = model.tokenizer
    if hasattr(tokenizer, 'tokenizers'):
        token_name = list(tokenizer.tokenizers.keys())[0]
        text_tokens = tokenizer.tokenizers[token_name](text)
    else:
        text_tokens = tokenizer(text)

    text_tokens = [model.bos_id] + text_tokens + [model.eos_id]
    tokens_tensor = torch.tensor([text_tokens], dtype=torch.long, device=device)
    text_lens = torch.tensor([len(text_tokens)], dtype=torch.long, device=device)

    # Save input tokens
    write_tensor_binary(tokens_tensor, os.path.join(output_dir, "input_text_tokens.bin"))

    batch = {
        'text': tokens_tensor,
        'text_lens': text_lens,
        'sample_rate': model.output_sample_rate,
        'context_sample_rate': model.output_sample_rate,
        'baked_context_speaker_ids': torch.tensor([speaker_id], dtype=torch.long, device=device),
    }

    # Limit inference steps via inference_parameters
    original_max_steps = model.inference_parameters.max_decoder_steps
    model.inference_parameters.max_decoder_steps = 10  # Limit for testing

    print("\nRunning inference (limited to 10 steps for reference generation)...")
    with torch.no_grad():
        output = model.infer_batch(
            batch,
            use_cfg=False,
            return_cross_attn_probs=False,
        )

    # Restore original
    model.inference_parameters.max_decoder_steps = original_max_steps

    # Save captured tensors
    print("\n=== Captured Tensors ===")
    for name, tensor in capture.captures.items():
        safe_name = name.replace(".", "_")
        write_tensor_binary(tensor, os.path.join(output_dir, f"hook_{safe_name}.bin"))

    # Save final outputs
    print("\n=== Final Outputs ===")
    write_tensor_binary(output.predicted_codes, os.path.join(output_dir, "predicted_codes.bin"))
    write_tensor_binary(output.predicted_audio, os.path.join(output_dir, "predicted_audio.bin"))

    # Cleanup
    capture.remove_hooks()

    print(f"\n{'='*60}")
    print(f"Reference data saved to {output_dir}")
    print(f"Total tensors captured: {len(capture.captures)}")
    print(f"{'='*60}")


def dump_manual_layers(model, text: str, speaker_id: int, output_dir: str, device: str):
    """Manually dump layer-by-layer outputs with explicit control."""
    print(f"\n{'='*60}")
    print(f"Manual layer dump for: '{text}'")
    print(f"{'='*60}")

    # Tokenize
    tokenizer = model.tokenizer
    if hasattr(tokenizer, 'tokenizers'):
        token_name = list(tokenizer.tokenizers.keys())[0]
        text_tokens = tokenizer.tokenizers[token_name](text)
    else:
        text_tokens = tokenizer(text)

    text_tokens = [model.bos_id] + text_tokens + [model.eos_id]
    tokens = torch.tensor([text_tokens], dtype=torch.long, device=device)
    write_tensor_binary(tokens, os.path.join(output_dir, "manual_text_tokens.bin"))

    with torch.no_grad():
        # 1. Text embedding
        embedded = model.text_embedding(tokens)
        write_tensor_binary(embedded, os.path.join(output_dir, "manual_text_embedded.bin"))

        # 2. Text encoder
        seq_len = embedded.shape[1]
        x_mask = torch.ones(1, seq_len, device=device, dtype=embedded.dtype)

        # Position embeddings
        pos_emb = model.encoder.position_embeddings.weight[:seq_len].unsqueeze(0)
        x = embedded + pos_emb
        write_tensor_binary(x, os.path.join(output_dir, "manual_enc_with_pos.bin"))

        # Encoder layers
        for i, layer in enumerate(model.encoder.layers):
            x_out = layer(x, x_mask)
            if isinstance(x_out, dict):
                x = x_out['output']
            elif isinstance(x_out, tuple):
                x = x_out[0]
            else:
                x = x_out
            write_tensor_binary(x, os.path.join(output_dir, f"manual_enc_layer{i}_out.bin"))

        # Final norm
        encoded = model.encoder.norm_out(x)
        write_tensor_binary(encoded, os.path.join(output_dir, "manual_enc_output.bin"))

        # 3. Baked context
        baked_flat = model.baked_context_embedding.weight[speaker_id]
        T, D = 110, 768
        context = baked_flat.view(T, D).unsqueeze(0)
        write_tensor_binary(context, os.path.join(output_dir, "manual_baked_context.bin"))

        # 4. Initial audio (BOS)
        audio_codes = torch.full((1, 8, 1), model.audio_bos_id, dtype=torch.long, device=device)
        write_tensor_binary(audio_codes, os.path.join(output_dir, "manual_audio_bos.bin"))

        # Sum audio embeddings
        audio_emb = sum(model.audio_embeddings[cb](audio_codes[:, cb, :]) for cb in range(8))
        write_tensor_binary(audio_emb, os.path.join(output_dir, "manual_audio_emb.bin"))

        # 5. Decoder input = [context; audio_emb]
        dec_input = torch.cat([context, audio_emb], dim=1)
        dec_len = dec_input.shape[1]
        dec_pos = model.decoder.position_embeddings.weight[:dec_len].unsqueeze(0)
        dec_input = dec_input + dec_pos
        write_tensor_binary(dec_input, os.path.join(output_dir, "manual_dec_input.bin"))

        # Decoder forward
        dec_mask = torch.ones(1, dec_len, device=device, dtype=dec_input.dtype)
        enc_mask = torch.ones(1, seq_len, device=device, dtype=encoded.dtype)
        dec_out = model.decoder(dec_input, dec_mask, encoded, enc_mask)
        if isinstance(dec_out, dict):
            dec_out = dec_out['output']
        elif isinstance(dec_out, tuple):
            dec_out = dec_out[0]
        write_tensor_binary(dec_out, os.path.join(output_dir, "manual_dec_output.bin"))

        # 6. Final projection (last frame only)
        logits = model.final_proj(dec_out[:, -1:, :])
        write_tensor_binary(logits, os.path.join(output_dir, "manual_logits.bin"))

        # 7. Local transformer
        lt_input = model.local_transformer_in_projection(dec_out[:, -1, :])
        write_tensor_binary(lt_input, os.path.join(output_dir, "manual_lt_input.bin"))

    print(f"\nManual layer dump complete")


def main():
    parser = argparse.ArgumentParser(description="Dump reference tensors for GGML validation")
    parser.add_argument(
        "--model",
        type=str,
        default="../magpie_tts_multilingual_357m/magpie_tts_multilingual_357m.nemo",
        help="Path to Magpie TTS model"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello world",
        help="Text to synthesize"
    )
    parser.add_argument(
        "--speaker-id",
        type=int,
        default=0,
        help="Baked speaker ID (0-4)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_data/reference",
        help="Output directory for reference tensors"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hooks", "manual", "both"],
        default="both",
        help="Capture mode: hooks (use forward hooks), manual (explicit layer calls), both"
    )
    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, using CPU")

    # Load model
    model = load_magpie_model(args.model, device)

    # Dump based on mode
    if args.mode in ["hooks", "both"]:
        dump_with_hooks(model, args.text, args.speaker_id, args.output_dir, device)

    if args.mode in ["manual", "both"]:
        dump_manual_layers(model, args.text, args.speaker_id, args.output_dir, device)


if __name__ == "__main__":
    main()
