# Magpie TTS GGML Port Implementation Plan

## Overview

This document describes the step-by-step plan for porting NVIDIA Magpie TTS from PyTorch (NeMo) to GGML. The plan follows the principle of **testing every layer before proceeding to the next**.

**Related documents:**
- [GGML_INIT_OUTLINE.md](GGML_INIT_OUTLINE.md) - Detailed architecture for GGML context, graph, and model initialization
- [MAGPIE_ARCHITECTURE.md](MAGPIE_ARCHITECTURE.md) - Model architecture documentation
- [MAGPIE_INFERENCE_FINDINGS.md](MAGPIE_INFERENCE_FINDINGS.md) - Inference flow analysis

## Model Components to Port

1. **Text Encoder** (6 layers, 12 heads, d=768)
2. **Context Encoder** (2 layers, d=768)
3. **Main Decoder** (12 layers, 12 SA heads, 1 XA head, d=768)
4. **Local Transformer** (1 layer, d=256)
5. **Audio Codec Decoder** (HiFiGAN-based)

## Phase 1: GGUF Conversion Scripts

### 1.1 Create `scripts/convert_magpie_to_gguf.py`

Convert the main Magpie TTS model weights to GGUF format.

**Key tensors to extract:**
```
# Text Embedding (2380 x 768)
text_embedding.weight

# Text Encoder (6 layers)
encoder.position_embeddings.weight              # (4096, 768)
encoder.layers.{0-5}.norm_self.weight           # (768,)
encoder.layers.{0-5}.self_attention.qkv_net.weight  # (2304, 768)
encoder.layers.{0-5}.self_attention.o_net.weight    # (768, 768)
encoder.layers.{0-5}.norm_pos_ff.weight         # (768,)
encoder.layers.{0-5}.pos_ff.proj.conv.weight    # (3072, 768, 3)
encoder.layers.{0-5}.pos_ff.o_net.conv.weight   # (768, 3072, 3)
encoder.norm_out.weight                          # (768,)

# Main Decoder (12 layers)
decoder.layers.{0-11}.norm_self.weight          # (768,)
decoder.layers.{0-11}.self_attention.qkv_net.weight # (2304, 768)
decoder.layers.{0-11}.self_attention.o_net.weight   # (768, 768)
decoder.layers.{0-11}.norm_xattn_query.weight   # (768,)
decoder.layers.{0-11}.cross_attention.q_net.weight  # (128, 768)
decoder.layers.{0-11}.cross_attention.kv_net.weight # (256, 768)
decoder.layers.{0-11}.cross_attention.o_net.weight  # (768, 128)
decoder.layers.{0-11}.norm_xattn_memory.weight  # (768,)
decoder.layers.{0-11}.norm_pos_ff.weight        # (768,)
decoder.layers.{0-11}.pos_ff.proj.conv.weight   # (3072, 768, 1)
decoder.layers.{0-11}.pos_ff.o_net.conv.weight  # (768, 3072, 1)
decoder.norm_out.weight                          # (768,)

# Audio Embeddings (8 codebooks)
audio_embeddings.{0-7}.weight                   # (2024, 768) each

# Final Projection
final_proj.weight                                # (16192, 768)
final_proj.bias                                  # (16192,)

# Local Transformer
local_transformer_in_projection.weight           # (256, 768)
local_transformer_in_projection.bias             # (256,)
local_transformer.position_embeddings.weight     # (10, 256)
local_transformer.layers.0.norm_self.weight      # (256,)
local_transformer.layers.0.self_attention.qkv_net.weight  # (768, 256)
local_transformer.layers.0.self_attention.o_net.weight    # (256, 256)
local_transformer.layers.0.norm_pos_ff.weight    # (256,)
local_transformer.layers.0.pos_ff.proj.conv.weight   # (1024, 256, 1)
local_transformer.layers.0.pos_ff.o_net.conv.weight  # (256, 1024, 1)
local_transformer_out_projections.{0-7}.weight  # (2024, 256) each
local_transformer_out_projections.{0-7}.bias    # (2024,) each

# Context Encoder (2 layers, same structure as encoder)
context_encoder.*

# Baked Context Embedding (5 speakers)
baked_context_embedding.weight                   # (5, 84480)
```

**Hyperparameters to store:**
```
magpie.sample_rate = 22050
magpie.num_codebooks = 8
magpie.codebook_size = 2016
magpie.vocab_size_per_codebook = 2024
magpie.text_vocab_size = 2380
magpie.d_model = 768
magpie.d_ffn = 3072
magpie.encoder_layers = 6
magpie.decoder_layers = 12
magpie.encoder_heads = 12
magpie.decoder_sa_heads = 12
magpie.decoder_xa_heads = 1
magpie.local_transformer_dim = 256
magpie.local_transformer_layers = 1
magpie.num_baked_speakers = 5
magpie.baked_context_frames = 110
magpie.text_bos_id = 2378
magpie.text_eos_id = 2379
magpie.audio_bos_id = 2016
magpie.audio_eos_id = 2017
magpie.context_audio_bos_id = 2018
magpie.context_audio_eos_id = 2019
magpie.mask_token_id = 2020
```

### 1.2 Create `scripts/convert_codec_to_gguf.py`

Convert the NeMo Nano Codec model weights to GGUF format.

**Key components:**
- HiFiGAN Encoder (for optional audio context encoding)
- Vector Quantizer (FSQ, 8 codebooks x 2016)
- HiFiGAN Decoder (for audio reconstruction)

---

## Phase 2: Layer-by-Layer Implementation

Each phase includes:
1. **Implement** the layer in GGML
2. **Dump** intermediate tensors from PyTorch reference
3. **Compare** GGML output vs PyTorch output (max abs diff < 1e-4)

### 2.1 Text Embedding Layer

**Files:**
- `src/magpie.cpp` - main implementation
- `src/magpie.h` - model structure and API
- `test/test_text_embedding.py` - dump reference tensors
- `test/test_text_embedding.cpp` - GGML test

**Implementation:**
```cpp
// Input: text tokens [batch, seq_len]
// Output: embedded [batch, seq_len, 768]
ggml_tensor* magpie_text_embedding(
    ggml_context* ctx,
    ggml_tensor* tokens,      // [seq_len] int32
    ggml_tensor* emb_weight   // [vocab_size, d_model]
);
```

**Test plan:**
1. Dump PyTorch: `text_embedding(text_tokens)` -> `test_data/text_emb_out.bin`
2. GGML: Load tokens, compute embedding, compare

### 2.2 Positional Embedding + LayerNorm

**Implementation:**
```cpp
// Add positional embeddings
ggml_tensor* magpie_add_position_embeddings(
    ggml_context* ctx,
    ggml_tensor* x,           // [seq_len, d_model]
    ggml_tensor* pos_weight,  // [max_seq, d_model]
    int offset
);

// RMSNorm or LayerNorm (check which Magpie uses)
ggml_tensor* magpie_layer_norm(
    ggml_context* ctx,
    ggml_tensor* x,
    ggml_tensor* weight,
    float eps
);
```

### 2.3 Self-Attention Block

**Implementation:**
```cpp
ggml_tensor* magpie_self_attention(
    ggml_context* ctx,
    ggml_tensor* x,           // [seq_len, d_model]
    ggml_tensor* qkv_weight,  // [3*d_model, d_model]
    ggml_tensor* o_weight,    // [d_model, d_model]
    int n_heads,
    bool is_causal,
    ggml_tensor* kv_cache_k,  // optional KV cache
    ggml_tensor* kv_cache_v
);
```

**Test plan:**
1. Test non-causal attention (encoder)
2. Test causal attention (decoder)
3. Test with KV cache

### 2.4 Cross-Attention Block (Decoder only)

**Implementation:**
```cpp
ggml_tensor* magpie_cross_attention(
    ggml_context* ctx,
    ggml_tensor* query,       // [seq_len, d_model]
    ggml_tensor* memory,      // [mem_len, d_model]
    ggml_tensor* q_weight,    // [n_heads*d_head, d_model]
    ggml_tensor* kv_weight,   // [2*n_heads*d_head, d_model]
    ggml_tensor* o_weight,    // [d_model, n_heads*d_head]
    int n_heads
);
```

### 2.5 Feed-Forward Network (Conv-based)

The encoder uses kernel_size=3 convolutions, decoder uses kernel_size=1 (pointwise).

**Implementation:**
```cpp
ggml_tensor* magpie_pos_ff(
    ggml_context* ctx,
    ggml_tensor* x,           // [seq_len, d_model]
    ggml_tensor* proj_weight, // [d_ffn, d_model, kernel]
    ggml_tensor* o_weight,    // [d_model, d_ffn, kernel]
    int kernel_size
);
```

### 2.6 Full Encoder Layer

Combine: LayerNorm -> Self-Attention -> Residual -> LayerNorm -> FFN -> Residual

**Test plan:**
1. Test single encoder layer
2. Test all 6 layers stacked

### 2.7 Full Decoder Layer

Combine: LayerNorm -> Self-Attention -> Residual -> LayerNorm -> Cross-Attention -> Residual -> LayerNorm -> FFN -> Residual

**Test plan:**
1. Test single decoder layer
2. Test all 12 layers stacked

### 2.8 Audio Embedding + Summation

**Implementation:**
```cpp
// Sum embeddings across 8 codebooks
ggml_tensor* magpie_audio_embedding(
    ggml_context* ctx,
    ggml_tensor* codes,       // [num_codebooks, seq_len] int32
    ggml_tensor* emb_weights[8]  // [vocab_size, d_model] each
);
```

### 2.9 Final Projection + Reshape

**Implementation:**
```cpp
// Project decoder output to logits for all codebooks
ggml_tensor* magpie_final_proj(
    ggml_context* ctx,
    ggml_tensor* x,           // [seq_len, d_model]
    ggml_tensor* weight,      // [num_codebooks * vocab_size, d_model]
    ggml_tensor* bias         // [num_codebooks * vocab_size]
);
// Output: [seq_len, num_codebooks, vocab_size]
```

### 2.10 Local Transformer

Autoregressive refinement across codebooks within each frame.

**Implementation:**
```cpp
// For each frame, autoregressively predict codes for codebooks 0-7
ggml_tensor* magpie_local_transformer(
    ggml_context* ctx,
    ggml_tensor* frame_hidden,    // [d_model] from decoder
    ggml_tensor* in_proj_weight,  // [lt_dim, d_model]
    ggml_tensor* in_proj_bias,    // [lt_dim]
    ggml_tensor* pos_weight,      // [10, lt_dim]
    // ... attention weights ...
    ggml_tensor* out_proj_weights[8],  // [vocab_size, lt_dim] each
    ggml_tensor* out_proj_biases[8]    // [vocab_size] each
);
```

### 2.11 Context Encoder

Similar to text encoder but with 2 layers. Used for encoding speaker context.

### 2.12 Baked Context Embedding

**Implementation:**
```cpp
// Get pre-computed speaker embedding
ggml_tensor* magpie_get_baked_context(
    ggml_context* ctx,
    ggml_tensor* baked_weight,  // [num_speakers, frames * d_model]
    int speaker_id
);
// Output: [frames, d_model]
```

---

## Phase 3: Audio Codec Decoder

The audio codec converts discrete codes to waveform using HiFiGAN-style decoder.

### 3.1 Vector Dequantization

Look up codes in codebook embeddings and sum/combine.

### 3.2 Upsampling Blocks

ConvTranspose1d with stride for upsampling.

### 3.3 Residual Blocks

Conv1d with residual connections.

### 3.4 Full Decoder Pipeline

Combine all blocks for codes -> waveform conversion.

---

## Phase 4: Full Inference Pipeline

### 4.1 Text-to-Codes Generation

```cpp
struct magpie_inference_result {
    ggml_tensor* codes;       // [num_codebooks, num_frames]
    int num_frames;
};

magpie_inference_result magpie_generate(
    magpie_model* model,
    const int* text_tokens,
    int text_len,
    int speaker_id,           // 0-4 for baked speakers
    float temperature,
    int top_k,
    int max_frames
);
```

### 4.2 Codes-to-Audio Synthesis

```cpp
ggml_tensor* magpie_decode_audio(
    magpie_codec_model* codec,
    ggml_tensor* codes        // [num_codebooks, num_frames]
);
// Output: [num_samples] at 22050 Hz
```

### 4.3 End-to-End Test

1. Load models
2. Tokenize text
3. Generate codes
4. Decode to audio
5. Compare with PyTorch reference

---

## Phase 5: Optimization

### 5.1 Quantization

- Q8_0 for encoder/decoder weights
- Q4_0 for more aggressive compression
- Keep embeddings in F16/F32

### 5.2 KV-Cache Optimization

Pre-allocate and reuse KV cache for autoregressive generation.

### 5.3 Metal/CUDA Backends

Enable GPU acceleration for supported operations.

---

## Testing Infrastructure

### Reference Data Generation

Create `scripts/dump_reference.py`:
```python
def dump_layer_outputs(model, text, output_dir):
    """Dump intermediate tensors for each layer."""
    # Hook into each layer and save outputs
    pass
```

### GGML Test Framework

Create `test/test_runner.cpp`:
```cpp
bool compare_tensors(
    ggml_tensor* ggml_out,
    const char* reference_path,
    float tolerance = 1e-4f
);
```

---

## File Structure

```
magpie.cpp/
├── src/
│   ├── magpie.h           # Model structures and API
│   ├── magpie.cpp         # Main implementation
│   ├── magpie-encoder.cpp # Encoder layers
│   ├── magpie-decoder.cpp # Decoder layers
│   ├── magpie-codec.cpp   # Audio codec
│   └── magpie-sample.cpp  # Sampling utilities
├── scripts/
│   ├── convert_magpie_to_gguf.py
│   ├── convert_codec_to_gguf.py
│   └── dump_reference.py
├── test/
│   ├── test_runner.cpp
│   ├── test_text_embedding.py
│   ├── test_encoder.py
│   ├── test_decoder.py
│   └── test_data/         # Reference tensors
├── weights/
│   ├── magpie-357m.gguf
│   └── nano-codec.gguf
└── examples/
    └── main.cpp           # Example usage
```

---

## Milestones

| Milestone | Description | Verification |
|-----------|-------------|--------------|
| M1 | GGUF conversion scripts working | Load tensors, verify shapes |
| M2 | Text embedding + positional | Compare output to reference |
| M3 | Full encoder working | Compare encoded text output |
| M4 | Decoder self-attention | Compare with causal mask |
| M5 | Decoder cross-attention | Compare with encoder output |
| M6 | Full decoder working | Compare decoder output |
| M7 | Local transformer | Compare code predictions |
| M8 | Audio codec decoder | Compare waveform |
| M9 | End-to-end inference | Listen to generated audio |
| M10 | Quantization | Size reduction + quality check |

---

## Current Status

- [x] Model architecture documented (MAGPIE_ARCHITECTURE.md)
- [x] Inference flow analyzed (MAGPIE_INFERENCE_FINDINGS.md)
- [x] Reference GGUF converter available (nemotron-asr.cpp)
- [x] GGUF conversion script for Magpie TTS (scripts/convert_magpie_to_gguf.py)
- [x] GGUF conversion script for Nano Codec (scripts/convert_codec_to_gguf.py)
- [x] Reference tensor dumping script (scripts/dump_reference.py)
- [x] Generated GGUF files:
  - weights/magpie-357m-f32.gguf (855 MB)
  - weights/magpie-357m-q8.gguf (277 MB, 67% reduction)
  - weights/nano-codec-f32.gguf (120 MB)
- [x] Generated reference tensors in test_data/reference/
- [ ] **NEXT: Implement text embedding layer in GGML**
- [ ] Test text embedding against reference
- [ ] Continue layer-by-layer...
