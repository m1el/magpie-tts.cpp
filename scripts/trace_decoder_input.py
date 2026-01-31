#!/usr/bin/env python3
"""Trace decoder input between manual and inference."""

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

    # Manual context
    print("\n=== Manual Context ===")
    with torch.no_grad():
        baked_flat = model.baked_context_embedding.weight[0]
        context = baked_flat.view(110, 768).unsqueeze(0)
        print(f"Manual context shape: {context.shape}")
        print(f"Manual context first 5: {context[0, 0, :5].cpu().numpy()}")
        print(f"Manual context at [50, 0]: {context[0, 50, :5].cpu().numpy()}")

    # Trace what inference actually uses
    print("\n=== Tracing Inference ===")

    # Hook on decoder forward to see inputs
    decoder_inputs = []

    original_forward = model.decoder.forward

    def hooked_forward(inputs, *args, **kwargs):
        decoder_inputs.append({
            'inputs': inputs.clone(),
            'args': args,
            'kwargs': {k: v.clone() if torch.is_tensor(v) else v for k, v in kwargs.items()}
        })
        return original_forward(inputs, *args, **kwargs)

    model.decoder.forward = hooked_forward

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

    model.decoder.forward = original_forward

    if decoder_inputs:
        print(f"Number of decoder calls: {len(decoder_inputs)}")
        for i, di in enumerate(decoder_inputs[:1]):
            print(f"\nDecoder call {i}:")
            print(f"  Input shape: {di['inputs'].shape}")
            print(f"  Input first 5 at pos 0: {di['inputs'][0, 0, :5].cpu().numpy()}")
            print(f"  Input first 5 at pos 50: {di['inputs'][0, 50, :5].cpu().numpy()}")
            if 'attention_mask' in di['kwargs']:
                print(f"  Attention mask shape: {di['kwargs']['attention_mask'].shape}")

    # Check if context_encoder is used
    print("\n=== Checking Context Encoder ===")
    print(f"Has context_encoder: {hasattr(model, 'context_encoder')}")
    if hasattr(model, 'context_encoder'):
        print(f"Context encoder layers: {len(model.context_encoder.layers)}")

        # Run context through context_encoder
        with torch.no_grad():
            ctx_len = context.shape[1]
            ctx_mask = torch.ones(1, ctx_len, device="cuda", dtype=context.dtype)
            ctx_pos = model.context_encoder.position_embeddings.weight[:ctx_len].unsqueeze(0)
            ctx_with_pos = context + ctx_pos

            x = ctx_with_pos
            for layer in model.context_encoder.layers:
                layer_out = layer(x, ctx_mask)
                x = layer_out['output'] if isinstance(layer_out, dict) else layer_out[0] if isinstance(layer_out, tuple) else layer_out
            encoded_context = model.context_encoder.norm_out(x)

            print(f"\nEncoded context shape: {encoded_context.shape}")
            print(f"Encoded context first 5 at pos 0: {encoded_context[0, 0, :5].cpu().numpy()}")

            # Compare with decoder input
            if decoder_inputs:
                inf_ctx = decoder_inputs[0]['inputs'][0, :110, :5].cpu().numpy()
                man_ctx = encoded_context[0, :, :5].cpu().numpy()
                print(f"\nDecoder input (first 110 pos, first 5 dim):")
                print(f"  From inference: {inf_ctx[0]}")
                print(f"  From manual context_encoder: {man_ctx[0]}")

if __name__ == "__main__":
    main()
