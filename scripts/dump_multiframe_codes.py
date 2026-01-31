#!/usr/bin/env python3
"""
Generate reference audio codes for multiple frames to compare with GGML.
"""

import torch
import struct
import os

def main():
    from nemo.collections.tts.models import MagpieTTSModel

    model_path = "../magpie_tts_multilingual_357m/magpie_tts_multilingual_357m.nemo"
    print(f"Loading model from {model_path}...")
    model = MagpieTTSModel.restore_from(model_path)
    model.to("cuda")
    model.eval()

    # Tokenize "Hello world"
    text = "Hello world"
    tokenizer = model.tokenizer
    if hasattr(tokenizer, 'tokenizers'):
        token_name = list(tokenizer.tokenizers.keys())[0]
        text_tokens = tokenizer.tokenizers[token_name](text)
    else:
        text_tokens = tokenizer(text)

    text_tokens = [model.bos_id] + text_tokens + [model.eos_id]
    tokens = torch.tensor([text_tokens], dtype=torch.long, device="cuda")
    text_lens = torch.tensor([len(text_tokens)], dtype=torch.long, device="cuda")

    print(f"Tokens: {text_tokens}")
    print(f"Token count: {len(text_tokens)}")

    # Limit to 20 frames for comparison
    model.inference_parameters.max_decoder_steps = 20

    batch = {
        'text': tokens,
        'text_lens': text_lens,
        'sample_rate': model.output_sample_rate,
        'context_sample_rate': model.output_sample_rate,
        'baked_context_speaker_ids': torch.tensor([0], dtype=torch.long, device="cuda"),
    }

    print("\nRunning inference (20 frames)...")
    with torch.no_grad():
        output = model.infer_batch(
            batch,
            use_cfg=False,
            return_cross_attn_probs=False,
        )

    codes = output.predicted_codes.cpu().numpy()
    print(f"\nPredicted codes shape: {codes.shape}")  # Should be [1, 8, n_frames]

    # Print all codes
    n_frames = codes.shape[2]
    print(f"\nGenerated {n_frames} frames:")
    for t in range(n_frames):
        frame_codes = codes[0, :, t]
        print(f"  Frame {t}: {' '.join(map(str, frame_codes))}")

    # Save to binary file
    output_path = "test_data/reference/pytorch_codes_20frames.bin"
    with open(output_path, 'wb') as f:
        # Write shape header
        shape = [8, n_frames, 1, 1]
        for dim in reversed(shape):
            f.write(struct.pack('<q', dim))
        # Write data as int32 (flatten in column-major order for GGML)
        for cb in range(8):
            for t in range(n_frames):
                f.write(struct.pack('<i', int(codes[0, cb, t])))

    print(f"\nSaved codes to {output_path}")

    # Also save just the codes as text for easy comparison
    text_path = "test_data/reference/pytorch_codes_20frames.txt"
    with open(text_path, 'w') as f:
        for t in range(n_frames):
            f.write(' '.join(map(str, codes[0, :, t])) + '\n')
    print(f"Saved codes to {text_path}")

if __name__ == "__main__":
    main()
