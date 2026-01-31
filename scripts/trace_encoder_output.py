#!/usr/bin/env python3
"""Trace encoder output between manual and inference."""

import torch
import numpy as np

def main():
    from nemo.collections.tts.models import MagpieTTSModel

    model_path = "../magpie_tts_multilingual_357m/magpie_tts_multilingual_357m.nemo"
    print(f"Loading model...")
    model = MagpieTTSModel.restore_from(model_path)
    model.to("cuda")
    model.eval()

    text = "Hello world"
    tokenizer = model.tokenizer
    if hasattr(tokenizer, 'tokenizers'):
        token_name = list(tokenizer.tokenizers.keys())[0]
        text_tokens = tokenizer.tokenizers[token_name](text)
    else:
        text_tokens = tokenizer(text)

    text_tokens = [model.bos_id] + text_tokens + [model.eos_id]
    tokens = torch.tensor([text_tokens], dtype=torch.long, device="cuda")
    print(f"Tokens: {text_tokens}")

    # Manual encoder
    print("\n=== Manual Encoder ===")
    with torch.no_grad():
        embedded = model.text_embedding(tokens)
        seq_len = embedded.shape[1]
        x_mask = torch.ones(1, seq_len, device="cuda", dtype=embedded.dtype)
        pos_emb = model.encoder.position_embeddings.weight[:seq_len].unsqueeze(0)
        x = embedded + pos_emb

        # Check encoder layer settings
        layer = model.encoder.layers[0]
        print(f"Encoder layer 0 self_attention is_causal: {getattr(layer.self_attention, 'is_causal', 'N/A')}")

        for i, layer in enumerate(model.encoder.layers):
            x_out = layer(x, x_mask)
            x = x_out['output'] if isinstance(x_out, dict) else x_out[0] if isinstance(x_out, tuple) else x_out
        encoder_output = model.encoder.norm_out(x)

    print(f"Manual encoder output shape: {encoder_output.shape}")
    print(f"Manual encoder output first 5: {encoder_output[0, 0, :5].cpu().numpy()}")
    print(f"Manual encoder output last 5 at pos 0: {encoder_output[0, 0, -5:].cpu().numpy()}")

    # Now capture during inference
    print("\n=== Inference Encoder (via hook) ===")
    encoder_outputs = []

    def hook_fn(module, input, output):
        if isinstance(output, dict):
            encoder_outputs.append(output['output'].clone())
        else:
            encoder_outputs.append(output[0].clone() if isinstance(output, tuple) else output.clone())

    # Hook on the final encoder norm
    hook = model.encoder.norm_out.register_forward_hook(
        lambda m, i, o: encoder_outputs.append(o.clone())
    )

    text_lens = torch.tensor([len(text_tokens)], dtype=torch.long, device="cuda")
    model.inference_parameters.max_decoder_steps = 1
    model.inference_parameters.temperature = 0.0001

    batch = {
        'text': tokens,
        'text_lens': text_lens,
        'sample_rate': model.output_sample_rate,
        'context_sample_rate': model.output_sample_rate,
        'baked_context_speaker_ids': torch.tensor([0], dtype=torch.long, device="cuda"),
    }

    with torch.no_grad():
        output = model.infer_batch(batch, use_cfg=False)

    hook.remove()

    if encoder_outputs:
        inf_enc_out = encoder_outputs[-1]
        print(f"Inference encoder output shape: {inf_enc_out.shape}")
        print(f"Inference encoder output first 5: {inf_enc_out[0, 0, :5].cpu().numpy()}")
        print(f"Inference encoder output last 5 at pos 0: {inf_enc_out[0, 0, -5:].cpu().numpy()}")

        # Compare
        diff = (encoder_output - inf_enc_out).abs().max().item()
        print(f"\nMax diff between manual and inference encoder output: {diff}")

if __name__ == "__main__":
    main()
