# Magpie TTS Architecture (357M Multilingual Model)

This document describes the architecture of the NVIDIA Magpie TTS model as observed from the `nvidia/magpie_tts_multilingual_357m` checkpoint.

## Model Overview

| Property | Value |
|----------|-------|
| Model Type | `decoder_ce` (context encoder decoder) |
| Sample Rate | 22050 Hz |
| Num Codebooks | 8 |
| Codebook Size | 2016 |
| Vocab Size Per Codebook | 2024 (codebook + 8 special tokens) |
| Frame Stacking Factor | 1 |
| Total Parameters | ~357M |

## Architecture Diagram

```
                          Input Text
                              │
                              ▼
                    ┌─────────────────┐
                    │ Text Embedding  │  (2380 tokens -> 768 dim)
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Text Encoder   │  6 layers, 12 heads
                    │   (Transformer) │  d_model=768, d_ffn=3072
                    └────────┬────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │  Encoded Text (conditioning)  │
              └──────────────┬───────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   │                   │
┌─────────────────┐          │      ┌────────────────────┐
│ Baked Context   │          │      │   Audio Codes      │
│   Embedding     │ ◄────────┼──────│   (autoregressive) │
│  (5 speakers)   │          │      │   BOS -> ... -> EOS│
└────────┬────────┘          │      └──────────┬─────────┘
         │                   │                 │
         ▼                   │                 ▼
┌─────────────────┐          │      ┌─────────────────┐
│ Context Encoder │          │      │ Audio Embeddings│  8 embeddings
│  (2 layers)     │          │      │  (sum across    │  (2024 x 768 each)
└────────┬────────┘          │      │   codebooks)    │
         │                   │      └────────┬────────┘
         ▼                   │               │
    ┌─────────────┐          │               │
    │  Context    │          │               │
    │  Prepended  ├──────────┴───────────────┤
    └─────────────┘                          │
                                             ▼
                             ┌─────────────────────────┐
                             │      Main Decoder       │
                             │  12 layers, 12 SA heads │
                             │  1 XA head per layer    │
                             │  d_model=768, d_ffn=3072│
                             │  (causal self-attention)│
                             └───────────┬─────────────┘
                                         │
                                         ▼
                             ┌─────────────────────────┐
                             │    Final Projection     │
                             │  768 -> 16,192          │
                             │  (8 * 2024 logits)      │
                             └───────────┬─────────────┘
                                         │
                                         ▼
                             ┌─────────────────────────┐
                             │   Local Transformer     │
                             │   (autoregressive)      │
                             │   1 layer, d=256        │
                             │   Refines per-codebook  │
                             └───────────┬─────────────┘
                                         │
                                         ▼
                             ┌─────────────────────────┐
                             │   Predicted Codes       │
                             │   (8 codebooks x T)     │
                             └───────────┬─────────────┘
                                         │
                                         ▼
                             ┌─────────────────────────┐
                             │   Audio Codec Decoder   │
                             │   (HiFiGAN-based)       │
                             └───────────┬─────────────┘
                                         │
                                         ▼
                                  Output Waveform
```

## Component Details

### 1. Text Embedding
- **Module**: `text_embedding` (nn.Embedding)
- **Vocab Size**: 2380 (phonemes + BOS/EOS)
- **Embedding Dim**: 768
- **Example**: `(1, 70) -> (1, 70, 768)` (70 phonemes)

### 2. Text Encoder
- **Module**: `encoder` (Transformer)
- **Type**: Non-causal (bidirectional attention)
- **Layers**: 6
- **Hidden Dim**: 768
- **FFN Dim**: 3072
- **Self-Attention Heads**: 12
- **Head Dim**: 64
- **Kernel Size**: 3 (for Conv FFN)
- **Position Embeddings**: Learnable (`encoder.position_embeddings`)

**Layer Structure**:
```
for each layer:
    x = x + self_attention(layer_norm(x))
    x = x + pos_ff(layer_norm(x))
output = layer_norm(x)
```

### 3. Context Encoder
- **Module**: `context_encoder` (Transformer)
- **Layers**: 2
- **Hidden Dim**: 768
- **Used for**: Encoding speaker context (baked or from audio)

### 4. Baked Context Embedding
- **Module**: `baked_context_embedding` (nn.Embedding)
- **Num Speakers**: 5
- **Shape**: (5, 84480) = (5, 110 * 768) flattened
- **Each Speaker**: 110 frames of 768-dim context

### 5. Audio Embeddings
- **Module**: `audio_embeddings` (ModuleList of 8 Embeddings)
- **Vocab Size Per Codebook**: 2024
- **Embedding Dim**: 768
- **All 8 embeddings are summed together** to get frame embedding

### 6. Main Decoder
- **Module**: `decoder` (Transformer)
- **Type**: Causal (autoregressive)
- **Layers**: 12
- **Hidden Dim**: 768
- **FFN Dim**: 3072
- **Self-Attention Heads**: 12
- **Cross-Attention Heads**: 1 (per layer)
- **Cross-Attention Head Dim**: 128
- **Kernel Size**: 1 (causal convolution)
- **KV Cache**: Used for efficient inference

**Layer Structure**:
```
for each layer:
    x = x + self_attention(layer_norm(x))          # causal
    x = x + cross_attention(layer_norm(x), cond)   # to text encoder output
    x = x + pos_ff(layer_norm(x))                  # causal convolution
output = layer_norm(x)
```

### 7. Final Projection
- **Module**: `final_proj` (nn.Linear)
- **Input Dim**: 768
- **Output Dim**: 16,192 (= 8 codebooks × 2024 tokens)

### 8. Local Transformer
- **Module**: `local_transformer` (Transformer)
- **Type**: Autoregressive (causal)
- **Purpose**: Refine audio codes sequentially across codebooks within each frame
- **Layers**: 1
- **Hidden Dim**: 256
- **Input Projection**: `local_transformer_in_projection` (768 -> 256)
- **Output Projections**: `local_transformer_out_projections` (8 × Linear, 256 -> 2024 each)

**Inference Process**:
```python
# For each frame's decoder output (768-dim):
lt_input = in_projection(dec_out)  # 768 -> 256
for codebook in range(8):
    lt_out = local_transformer(lt_input)
    logits = out_projections[codebook](lt_out)
    sampled_code = sample(logits)
    lt_input = cat([lt_input, embed(sampled_code)])
```

## Audio Codec Model

The audio codec (NeMo Nano Codec) is a separate model for converting between audio and discrete tokens:

| Property | Value |
|----------|-------|
| Sample Rate | 22050 Hz |
| Frame Rate | ~21.5 fps (1024 samples/frame) |
| Codebooks | 8 |
| Bitrate | ~1.89 kbps |

### Codec Architecture

**Encoder** (HiFiGAN-based):
- Input: Raw audio waveform
- Downsample layers: 5 (Conv1d with stride)
- Residual blocks between downsampling
- Output: Continuous latent representation

**Vector Quantizer**:
- FSQ (Finite Scalar Quantization) or VQ
- 8 parallel codebooks
- Codebook size: 2016

**Decoder** (HiFiGAN-based):
- Input: Quantized codes
- Upsample layers: 5 (ConvTranspose1d)
- Residual blocks between upsampling
- Output: Reconstructed audio waveform

## Special Tokens

| Token | ID | Purpose |
|-------|-----|---------|
| Text BOS | 2378 | Beginning of text |
| Text EOS | 2379 | End of text |
| Audio BOS | 2016 | Beginning of audio sequence |
| Audio EOS | 2017 | End of audio sequence |
| Context Audio BOS | 2018 | Beginning of context |
| Context Audio EOS | 2019 | End of context |
| Mask Token | 2020 | For MaskGit training |

## Tensor Shapes (Example Inference)

For text "Hello, this is a test of the Magpie text to speech system." (70 phonemes):

| Stage | Shape | Notes |
|-------|-------|-------|
| Text tokens | (1, 70) | int64, includes BOS/EOS |
| Text embedded | (1, 70, 768) | float32 |
| Text encoded | (1, 70, 768) | float32, encoder output |
| Context (baked) | (1, 110, 768) | float32, speaker embedding |
| Context encoded | (1, 110, 768) | float32, after context_encoder |
| Decoder input | (1, 227, 768) | float32, context + audio frames |
| Decoder output | (1, 227, 768) | float32 |
| Logits | (1, 227, 16192) | float32, 8×2024 per frame |
| Predicted codes | (1, 8, 116) | int64, 116 audio frames |
| Output audio | (1, 118784) | float32, 22050 Hz |

## Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| max_decoder_steps | 500 | Maximum audio frames to generate |
| temperature | 0.7 | Sampling temperature |
| topk | 80 | Top-k for sampling |
| cfg_scale | 2.5 | Classifier-free guidance scale |
| apply_attention_prior | True | Use attention guidance |
| eos_detection_method | argmax_or_multinomial_any | When to stop |
| min_generated_frames | 4 | Minimum frames before EOS allowed |

## Weight File Layout

When porting to GGML, the following weight tensors need to be extracted:

```
text_embedding.weight                    # (2380, 768)
encoder.position_embeddings.weight       # (4096, 768)
encoder.layers.{0-5}.norm_self.weight   # (768,)
encoder.layers.{0-5}.self_attention.qkv_net.weight  # (2304, 768)
encoder.layers.{0-5}.self_attention.o_net.weight    # (768, 768)
encoder.layers.{0-5}.norm_pos_ff.weight # (768,)
encoder.layers.{0-5}.pos_ff.proj.conv.weight        # (3072, 768, 3)
encoder.layers.{0-5}.pos_ff.o_net.conv.weight       # (768, 3072, 3)
encoder.norm_out.weight                  # (768,)

decoder.layers.{0-11}.norm_self.weight  # (768,)
decoder.layers.{0-11}.self_attention.qkv_net.weight # (2304, 768)
decoder.layers.{0-11}.self_attention.o_net.weight   # (768, 768)
decoder.layers.{0-11}.norm_xattn_query.weight       # (768,)
decoder.layers.{0-11}.cross_attention.q_net.weight  # (128, 768)
decoder.layers.{0-11}.cross_attention.kv_net.weight # (256, 768)
decoder.layers.{0-11}.cross_attention.o_net.weight  # (768, 128)
decoder.layers.{0-11}.norm_xattn_memory.weight      # (768,)
decoder.layers.{0-11}.norm_pos_ff.weight            # (768,)
decoder.layers.{0-11}.pos_ff.proj.conv.weight       # (3072, 768, 1)
decoder.layers.{0-11}.pos_ff.o_net.conv.weight      # (768, 3072, 1)
decoder.norm_out.weight                  # (768,)

audio_embeddings.{0-7}.weight           # (2024, 768) each

final_proj.weight                        # (16192, 768)
final_proj.bias                          # (16192,)

local_transformer_in_projection.weight   # (256, 768)
local_transformer_in_projection.bias     # (256,)
local_transformer.position_embeddings.weight  # (10, 256)
local_transformer.layers.0.norm_self.weight   # (256,)
local_transformer.layers.0.self_attention.qkv_net.weight  # (768, 256)
local_transformer.layers.0.self_attention.o_net.weight    # (256, 256)
local_transformer.layers.0.norm_pos_ff.weight  # (256,)
local_transformer.layers.0.pos_ff.proj.conv.weight   # (1024, 256, 1)
local_transformer.layers.0.pos_ff.o_net.conv.weight  # (256, 1024, 1)
local_transformer_out_projections.{0-7}.weight  # (2024, 256) each
local_transformer_out_projections.{0-7}.bias    # (2024,) each

context_encoder.*                        # Similar structure to encoder (2 layers)
baked_context_embedding.weight           # (5, 84480)
```

## GGML Port Strategy

1. **Start with Text Encoder**: Implement transformer blocks, layernorm, self-attention
2. **Add Main Decoder**: Similar to encoder but with cross-attention and causal masking
3. **Implement Local Transformer**: Small autoregressive transformer for code refinement
4. **Add Embeddings**: Text and audio embeddings
5. **Port Audio Codec**: Separate model for audio decode
6. **Implement Sampling**: Top-k, temperature, EOS detection

Key operations needed:
- Matrix multiplication (GGML: `ggml_mul_mat`)
- LayerNorm (GGML: `ggml_norm`)
- GELU activation (GGML: `ggml_gelu`)
- Softmax (GGML: `ggml_soft_max`)
- Causal attention mask (GGML: `ggml_diag_mask_inf`)
- Embedding lookup (GGML: `ggml_get_rows`)
- 1D Convolution (GGML: `ggml_conv_1d`)
