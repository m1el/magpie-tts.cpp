#!/usr/bin/env python3
"""Trace decoder output between manual and inference."""

import torch

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
    text_lens = torch.tensor([len(text_tokens)], dtype=torch.long, device="cuda")
    seq_len = len(text_tokens)

    # Manual computation
    print("\n=== Manual Decoder (First Frame) ===")
    with torch.no_grad():
        # Encoder
        embedded = model.text_embedding(tokens)
        x_mask = torch.ones(1, seq_len, device="cuda", dtype=embedded.dtype)
        pos_emb = model.encoder.position_embeddings.weight[:seq_len].unsqueeze(0)
        x = embedded + pos_emb
        for layer in model.encoder.layers:
            x_out = layer(x, x_mask)
            x = x_out['output'] if isinstance(x_out, dict) else x_out[0] if isinstance(x_out, tuple) else x_out
        encoder_output = model.encoder.norm_out(x)

        # Decoder input
        baked_flat = model.baked_context_embedding.weight[0]
        context = baked_flat.view(110, 768).unsqueeze(0)
        audio_codes = torch.full((1, 8, 1), model.audio_bos_id, dtype=torch.long, device="cuda")
        audio_emb = sum(model.audio_embeddings[cb](audio_codes[:, cb, :]) for cb in range(8))

        dec_input = torch.cat([context, audio_emb], dim=1)  # [1, 111, 768]
        dec_len = dec_input.shape[1]
        dec_pos = model.decoder.position_embeddings.weight[:dec_len].unsqueeze(0)
        dec_with_pos = dec_input + dec_pos

        # Decoder
        dec_mask = torch.ones(1, dec_len, device="cuda", dtype=dec_with_pos.dtype)
        enc_mask = torch.ones(1, seq_len, device="cuda", dtype=encoder_output.dtype)
        x = dec_with_pos
        for layer in model.decoder.layers:
            layer_out = layer(x, dec_mask, encoder_output, enc_mask)
            x = layer_out['output'] if isinstance(layer_out, dict) else layer_out[0] if isinstance(layer_out, tuple) else layer_out
        dec_output = model.decoder.norm_out(x)

        manual_last_hidden = dec_output[:, -1, :]  # [1, 768]
        print(f"Manual decoder output shape: {dec_output.shape}")
        print(f"Manual last hidden first 5: {manual_last_hidden[0, :5].cpu().numpy()}")
        print(f"Manual last hidden last 5: {manual_last_hidden[0, -5:].cpu().numpy()}")

    # Inference with hook
    print("\n=== Inference Decoder (via hook) ===")
    decoder_outputs = []

    hook = model.decoder.norm_out.register_forward_hook(
        lambda m, i, o: decoder_outputs.append(o.clone())
    )

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

    if decoder_outputs:
        inf_dec_out = decoder_outputs[0]  # First call (for first frame)
        print(f"Inference decoder output shape: {inf_dec_out.shape}")
        print(f"Inference last hidden first 5: {inf_dec_out[0, -1, :5].cpu().numpy()}")
        print(f"Inference last hidden last 5: {inf_dec_out[0, -1, -5:].cpu().numpy()}")

        # Compare last hidden states
        if manual_last_hidden.shape == inf_dec_out[:, -1, :].shape:
            diff = (manual_last_hidden - inf_dec_out[:, -1, :]).abs().max().item()
            print(f"\nMax diff in last hidden: {diff}")
        else:
            print(f"\nShapes differ: manual {manual_last_hidden.shape} vs inference {inf_dec_out[:, -1, :].shape}")

        # Check decoder input dimensions
        print(f"\nManual decoder input seq len: {dec_len}")
        print(f"Inference decoder output seq len: {inf_dec_out.shape[1]}")

if __name__ == "__main__":
    main()
