# Magpie TTS GGML Port - Status

## Goal
Port NVIDIA Magpie TTS (357M multilingual) from PyTorch/NeMo to GGML for efficient CPU/GPU inference.

## Model Location
- **Magpie TTS**: `../magpie_tts_multilingual_357m/magpie_tts_multilingual_357m.nemo`
- **Audio Codec**: `../nemo-nano-codec-22khz-1.89kbps-21.5fps/nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo`

## Completed Work

### Phase 1: Analysis & Conversion Scripts ✅
1. **Model inspection** (`inspect_inference.py`) - Traces all modules during inference
2. **Architecture documentation** (`MAGPIE_ARCHITECTURE.md`) - Complete model structure
3. **GGUF conversion** (`scripts/convert_magpie_to_gguf.py`) - Converts .nemo to .gguf
4. **Codec conversion** (`scripts/convert_codec_to_gguf.py`) - Converts audio codec
5. **Reference dumping** (`scripts/dump_reference.py`) - Dumps tensors for testing

### Generated Artifacts
```
weights/
├── magpie-357m-f32.gguf   (858 MB) - Full precision
├── magpie-357m-f16.gguf   (465 MB) - F16 precision (54% of F32)
├── magpie-357m-q8.gguf    (679 MB) - Q8 quantized (79% of F32)
└── nano-codec-f32.gguf    (121 MB) - Audio codec decoder

test_data/reference/       (77 files) - Reference tensors for layer testing
```

### Project Structure
```
src/
├── magpie-tts.cpp    # Main CLI binary
├── magpie.cpp        # TTS model implementation
├── magpie.h          # Header file
└── nano-codec.cpp    # Audio codec (HiFiGAN decoder)

docs/                 # Documentation (architecture, plans, status)
weights/              # GGUF model files + README
scripts/              # Conversion scripts
tests/                # Component and integration tests
```

### Documentation
- `README.md` - Project overview and usage
- `LICENSE` - MIT License
- `docs/MAGPIE_ARCHITECTURE.md` - Model architecture with tensor shapes
- `docs/MAGPIE_INFERENCE_FINDINGS.md` - Inference flow analysis
- `docs/GGML_PORT_PLAN.md` - Implementation plan with milestones
- `docs/GGML_INIT_OUTLINE.md` - GGML initialization architecture

## Current Phase: Implementation

### Completed Implementation Steps
1. ✅ **Create `src/magpie.h`** - All C++ structures defined from GGML_INIT_OUTLINE.md
2. ✅ **Create `src/magpie.cpp`** - Implemented `magpie_init()` to load GGUF, tensor mapping, KV cache init
3. ✅ **Test loading** - All 209 tensors load correctly to GPU, all critical tensors mapped to structs
4. ✅ **Implement text embedding** - `magpie_build_text_embedding()` using `ggml_get_rows()`
   - Tested against `test_data/reference/manual_text_embedded.bin` - PERFECT MATCH (max diff: 0.0)
   - Test: `make test_text_embedding && ./test_text_embedding`
5. ✅ **Implement audio embedding** - `magpie_build_audio_embedding()`, sum of 8 codebook embeddings
   - Tested against `test_data/reference/manual_audio_emb.bin` - PERFECT MATCH (max diff: 0.0)
   - Test: `make test_audio_embedding && ./test_audio_embedding`
6. ✅ **Implement Layer norm** - `magpie_build_layer_norm()` using ggml_norm + weight multiply
   - **FIX**: Model uses LayerNorm (with mean subtraction), NOT RMSNorm
   - Tested against `test_data/reference/enc_l0_norm1.bin` - PERFECT MATCH (max diff: 0.0)
   - Test: `./test_norm_step`
7. ✅ **Implement self-attention** - `magpie_build_self_attention()` using Flash Attention API
   - Key finding: NeMo encoder uses **causal** self-attention (is_causal=True)
   - CUDA requires F16 attention mask, CPU works with F32
   - Tested against `test_data/reference/hook_encoder_layers_0_self_attention.bin` - PASS (max diff: 0.006)
   - Test: `make test_self_attention && ./test_self_attention`
8. ✅ **Implement causal Conv1D FFN** - `magpie_build_conv_ffn()` for kernel_size > 1
   - Uses left-padding of (kernel_size-1) zeros for causal convolution
   - Implemented as sum of shifted matmuls for each kernel position
   - Uses GELU activation (not SiLU)
   - Key: ggml_permute(src, 2, 0, 1, 3) to rearrange weight layout for kernel slicing
   - Tested against `test_data/reference/debug_ffn_output.bin` - PASS (max diff: 0.007)
   - Test: `./test_conv_ffn_cpu` (manual test)
9. ✅ **Implement full encoder layer** - `magpie_build_encoder_layer()` combining all components
   - norm_self -> self_attention -> residual -> norm_ff -> conv_ffn -> residual
   - Tested against `test_data/reference/enc_l0_out.bin` - PASS (max diff: 0.094)
   - Test: `./test_layer_step`
10. ✅ **Implement full encoder** - `magpie_build_full_encoder()` with 6 layers + final norm
   - Tested against `test_data/reference/enc_output.bin` - **PASS (max diff: 0.008, avg diff: 0.0003)**
   - Test: `./test_full_encoder_v2`

### ENCODER COMPLETE ✅

The full 6-layer encoder now matches PyTorch with excellent accuracy:
- **Max diff: 0.008366**
- **Avg diff: 0.000266**

Key fix: Changed from RMSNorm to LayerNorm (the model uses `torch.nn.LayerNorm` with `bias=False`)

### Completed Implementation Steps (Decoder)
11. ✅ **Implement cross-attention** - `magpie_build_cross_attention()` for decoder-encoder attention
    - Query from decoder, Key/Value from encoder output
    - 1 head with d_head=128 (different from self-attention d_head=64)
    - No causal mask needed (full attention to encoder output)
    - Tested as part of decoder layer
12. ✅ **Implement decoder layer** - `magpie_build_decoder_layer()` combining all components
    - norm_self -> causal self_attention -> residual
    - norm_xa_query -> cross_attention (query to normed encoder) -> residual
    - norm_ff -> conv_ffn (kernel=1, pointwise) -> residual
    - Tested against `test_data/reference/dec_l0_out.bin` - PASS (max diff: 0.003502)
    - Test: `./test_decoder_layer`
13. ✅ **Implement full decoder** - 12 layers + final norm
    - Tested against `test_data/reference/dec_output.bin` - **PASS (max diff: 0.003, avg diff: 0.0003)**
    - Test: `make test_full_decoder && ./test_full_decoder`

### DECODER COMPLETE ✅

The full 12-layer decoder now matches PyTorch with excellent accuracy:
- **Max diff: 0.002658**
- **Avg diff: 0.000303**

Key implementation details:
- Cross-attention uses separate norm for query (decoder) and memory (encoder)
- Decoder FFN uses kernel_size=1 (pointwise, not causal conv)
- Reference data must be saved in column-major (Fortran) order for GGML

### Completed Implementation Steps (Final Projection + Local Transformer)
14. ✅ **Implement final projection** - `magpie_build_final_proj()` for logits
    - Simple linear layer with bias: W @ x + b
    - Tested against `test_data/reference/dec_logits.bin` - **PASS (max diff: 0.000001)**
    - Test: `./test_final_proj`
15. ✅ **Implement local transformer (single step)** - First step for codebook 0
    - Input projection (768 -> 256) + position embeddings + transformer layer + output projection
    - Tested against `test_data/reference/lt_logits_cb0.bin` - **PASS (max diff: 0.000004)**
    - Argmax matches PyTorch: token 293
    - Test: `./test_local_transformer` (test 1 & 2)
16. ✅ **Implement full local transformer** - All 8 codebook steps autoregressively
    - `magpie_local_transformer_sample_all()` - samples codes for all 8 codebooks
    - Key fix: lt_heads = 1 (not 4) for local transformer
    - All 8 sampled codes match PyTorch exactly: [293, 1454, 512, 1455, 476, 40, 1817, 1014]
    - Test: `./test_local_transformer` (test 3)

### LOCAL TRANSFORMER COMPLETE ✅

The full local transformer now matches PyTorch with exact accuracy for all 8 codebooks!

### Completed Implementation Steps (Audio Codec)
17. ✅ **Implement FSQ dequantization** - `fsq_dequantize_cpu()` for codebook indices to continuous values
    - 8 codebooks × 4 dims per codebook = 32 latent dimensions
    - levels = [8, 7, 6, 6], formula: code = (nonneg - L/2) / (L/2)
    - Tested against `test_data/reference/codec/codec_latent.bin` - **EXACT MATCH (max diff: 0.0)**
    - Test: `./test_codec_fsq`
18. ✅ **Implement CausalConv1d** - `magpie_codec_build_causal_conv1d()` with left padding
    - Left-pad by (kernel_size - 1) × dilation for causality
    - Tested pre-conv layer - PASS (max diff: 0.001252)
19. ✅ **Implement HalfSnake activation** - `magpie_codec_build_half_snake()`
    - First half: Snake activation x + (1/α) × sin²(αx)
    - Second half: LeakyReLU (slope=0.01)
    - Fixed handling of odd channel counts (e.g., 27 → 13+14 split)
    - Tested against reference - PASS (max diff: 0.001177)
20. ✅ **Implement grouped ConvTranspose1d** - `magpie_codec_build_conv_transpose1d()`
    - NeMo uses groups=out_ch with in_ch=2×out_ch (not true depthwise)
    - GGML doesn't support grouped conv_transpose, implemented per-group
    - Tested upsample layer 0 - PASS (max diff: 0.000000)
21. ✅ **Implement HiFiGAN residual blocks** - 3 blocks with kernels [3, 7, 11], averaged
    - Each block has 3 inner blocks with dilations [1, 3, 5]
    - Tested residual layer 0 - PASS (max diff: 0.006206)
22. ✅ **Full codec decoder** - 5 upsample stages, complete pipeline
    - Pre-conv → 5 × (HalfSnake → Upsample → ResLayer) → Post-act → Post-conv → Tanh
    - Tested end-to-end - **PASS (max diff: 0.004516)**
    - Test: `./test_codec_decode`

### AUDIO CODEC COMPLETE ✅

The full HiFiGAN audio codec decoder now matches PyTorch with excellent accuracy:
- **FSQ dequantization: EXACT MATCH (0.0)**
- **Full decoder: max diff 0.004516, within 0.05 tolerance**

Key implementation details:
- Grouped ConvTranspose requires per-group processing (432 groups for first upsample)
- Graph needs 131072 nodes capacity for many per-channel operations
- HalfSnake split must use alpha tensor size, not channels/2 (handles odd channels)

### Completed Implementation Steps (End-to-End Pipeline)
23. ✅ **Implement `magpie_encode_text()`** - Runs full encoder on text tokens
    - Builds encoder graph, computes, stores output in state
    - Tested as part of end-to-end pipeline
24. ✅ **Implement `magpie_synthesize_codes()`** - Full autoregressive generation
    - Extracts baked speaker context (110 frames, 768 dims)
    - Audio embedding via sum of 8 codebook embeddings
    - Runs decoder with cross-attention to encoder output
    - Local transformer samples all 8 codebooks per frame
    - EOS detection for stopping
    - Test: `./test_e2e_inference`
25. ✅ **End-to-end audio generation** - Full pipeline working
    - Text tokens → Encoder → Decoder (autoregressive) → Local transformer → Codes → Codec → WAV
    - Tested: 50 frames generated in ~10 seconds on RTX 4080
    - Output: 2.32 seconds of audio at 22050 Hz

### END-TO-END PIPELINE COMPLETE ✅

The full Magpie TTS pipeline is now functional:
- Text encoding, baked context, audio embedding all working
- Autoregressive decoder with cross-attention
- Local transformer for 8-codebook sampling
- HiFiGAN audio codec decoding
- WAV file output

First frame codes match PyTorch exactly: [293, 1454, 512, 1455, 476, 40, 1817, 1014]

### Bug Fix (2026-01-31): Audio Embedding Scaling ✅

**Problem**: Generated audio sounded like "tsss tsss" noise instead of speech.

**Root Cause Investigation**:
1. Encoder outputs matched PyTorch (max diff: 0.0) ✅
2. Decoder outputs differed significantly (max diff: 5.7) ❌
3. Created tracing scripts to debug:
   - `scripts/trace_encoder_output.py` - confirmed encoder matches
   - `scripts/trace_decoder_output.py` - found decoder mismatch
   - `scripts/trace_decoder_input.py` - traced what NeMo actually uses

**Key Findings**:
1. NeMo's `embed_audio_tokens()` divides the sum of codebook embeddings by `(num_codebooks * frame_stacking_factor) = 8`
2. Our GGML implementation was not scaling, resulting in 8x larger audio embeddings
3. This caused decoder hidden states to be completely wrong

**Fix Applied**:
- Added `ggml_scale(ctx, sum, 1.0f / 8.0f)` after summing audio embeddings
- Fixed in both `magpie_build_audio_embedding()` and inline code in `magpie_synthesize_codes()`

**Result**:
- Manual decoder output now matches NeMo inference exactly (max diff: 0.0)
- First 3 codes of frame 0 now match: `285 1455 512` (vs PyTorch `285 1455 512 ...`)

### Completed Implementation Steps (KV Cache Optimization)
26. ✅ **GPU-Resident KV Cache** - `magpie_synthesize_codes_optimized()` implemented
    - Uses persistent GPU-resident cache tensors (no CPU round-trips)
    - Applies `ggml_cpy` + `ggml_view` pattern for in-place cache updates
    - Cross-attention K/V pre-computed once and stored on GPU
    - **Performance: 2.3x faster than old cached version**
    - **52 frames/sec on RTX 4080** (comparable to uncached)
    - Test: `./test_e2e_optimized`

### KV Cache Performance Results

| Version | Speed (fps) | Speedup vs Uncached | Notes |
|---------|-------------|---------------------|-------|
| **Graph-Reuse** | **154.2** | **2.4x** | Batched context + allocator reuse |
| GPU-Optimized | 133.8 | 2.1x | GPU-resident KV cache |
| Uncached | 64.0 | 1.0x | Full decoder each step |

Key optimizations in graph-reuse version:
- **Batched context processing**: All 110 context frames in ONE graph (7.4ms total)
- **Persistent allocator**: Reserve once for max graph size, reuse across all steps
- **26% faster than GPU-Optimized**, builds on all previous optimizations

Key optimizations in GPU-optimized version:
- Pre-allocate KV cache as flat tensors on GPU: `ggml_backend_alloc_ctx_tensors()`
- Use views to write at current position: `ggml_view_1d(cache, d_model, offset)`
- Copy new K/V via graph operation: `ggml_cpy(k_new, k_slot)` runs on GPU
- No CPU round-trips during generation loop

**Known Issue**: Cached versions produce different outputs than uncached (even with deterministic sampling).
This is likely due to differences in how context frames are processed. The uncached version
processes the full sequence at once, while cached versions process frame-by-frame. Both produce
valid speech, but the specific codes diverge due to numerical differences in order of operations.

### Completed Implementation Steps (Graph Reuse Optimization)
29. ✅ **Graph-Reuse Optimization** - `magpie_synthesize_codes_graph_reuse()` implemented
    - **Batched context processing**: All 110 context frames in ONE graph pass
      - Uses `ggml_get_rows` for dynamic position embedding lookup
      - Batched causal self-attention with `ggml_diag_mask_inf`
      - 110 frames processed in ~7ms vs ~110ms before (15x faster for context)
    - **Persistent allocator reuse**: Reserve once, reuse across all autoregressive steps
      - Avoids allocator recreation overhead each step
    - **Performance: 154 fps on RTX 4080** (26% faster than GPU-Optimized)
    - Test: `./test_graph_reuse --text "Hello world" --compare`

### Completed Implementation Steps (Streaming API)
30. ✅ **Streaming TTS API** - `magpie_synthesize_streaming()` implemented
    - **Callback-based audio delivery**: Audio chunks delivered as they're generated
    - **Sentence chunking**: Splits text at sentence boundaries for lower latency
    - **Configurable chunk size**: Trade latency vs throughput (4 frames = 186ms default)
    - **Performance metrics**:
      - Time to first audio: ~165-193 ms (depending on chunk size)
      - Real-time factor: 1.4-3.3x (depending on chunk size)
    - Test: `./test_streaming --text "Hello! World." --chunk-size 4`

### Completed Implementation Steps (Quantization Fix)
31. ✅ **Q8 Quantization Fix** - Fixed block-size validation
    - Small tensors (inner dim < 32) now kept as F32
    - Q8 model: 679 MB (vs 858 MB F32) - 21% smaller
    - Reconvert with: `uv run scripts/convert_magpie_to_gguf.py ... -q q8`

### Completed Implementation Steps (EOS Detection & Tokenizer)
27. ✅ **Improved EOS Detection** - Full NeMo-compatible EOS handling
    - Forbidden token masking: BOS (2016), CONTEXT tokens (2018-2019), MASK (2020), RESERVED (2021-2023)
    - Temperature + top-k sampling implemented
    - Dual tracking: argmax AND sampled codes for argmax_or_multinomial_any detection
    - EOS forbidding during min_generated_frames (first 4 frames)
    - Test: EOS now detected reliably (step 31-34 for test text)
28. ✅ **Built-in Tokenizer** - Pure C++ text-to-tokens conversion
    - Vocabulary (96 IPA tokens) embedded in GGUF
    - Pronunciation dictionary (125k words, CMUdict) embedded in GGUF
    - Text normalization, word lookup, phoneme-to-token conversion
    - OOV words fall back to uppercase character tokens
    - Test: `./test_e2e_inference --text "Hello world"`

### TOKENIZER COMPLETE ✅

The model now accepts raw text input without requiring external tokenization:
- **Main Binary**: `./magpie-tts -t "Your text here" -o output.wav`
  - Uses graph-reuse (fastest, 154+ fps)
  - Run `./magpie-tts --help` for all options
- **Test binaries** also accept `--text`:
  - `./test_graph_reuse --text "Hello world"` (fastest, 154 fps)
  - `./test_e2e_optimized --text "Hello world"` (GPU-optimized, 134 fps)
  - `./test_e2e_inference --text "Hello world"` (standard, 64 fps)
- **Dictionary**: 125,854 English word pronunciations (IPA)
- **Vocabulary**: 96 phoneme tokens + punctuation + special tokens
- **Text Normalization** (matches NeMo behavior):
  - Numbers: "12" → "twelve", "101" → "one hundred and one"
  - Years: "2024" → "twenty twenty four"
  - Ordinals: "1st" → "first", "23rd" → "twenty third"
  - Currency: "$50" → "fifty dollars"
  - Percent: "50%" → "fifty percent"
- **Default**: Running without `--text` uses built-in test sentence

### Next Steps (Optimization)
1. **Debug cached vs uncached divergence** - Investigate why outputs differ
   - Compare hidden states at each layer
   - Verify position embeddings are equivalent
2. ~~**EOS detection tuning**~~ ✅ Fixed
3. **Graph reuse** - Pre-build decoder graph once, reuse for each step
4. **Streaming support** - Incremental decoding for real-time TTS
5. **Batch context priming** - Process all 110 context frames in one pass

### Architecture Summary

```
Text Input
    │
    ▼
┌────────────────────┐
│ Text Tokenizer     │  ✅ GGML (built-in, CMUdict IPA)
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Text Embedding     │  ✅ GGML (max diff: 0.0)
│ + Position Emb     │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Text Encoder       │  ✅ GGML (6 layers, max diff: 0.008)
│ (6 transformer     │
│  layers)           │
└─────────┬──────────┘
          │
          ├────────────────────────┐
          │                        │
          ▼                        ▼
    ┌──────────┐          ┌────────────────────┐
    │ Encoder  │          │ Baked Context      │
    │ Output   │          │ (Speaker Embedding)│
    └────┬─────┘          └─────────┬──────────┘
         │                          │
         │    ┌─────────────────────┘
         │    │
         ▼    ▼
┌────────────────────┐
│ Main Decoder       │  ✅ GGML (12 layers, max diff: 0.003)
│ (12 layers with    │
│  cross-attention)  │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Final Projection   │  ✅ GGML (max diff: 0.000001)
│ (768 → 16192)      │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Local Transformer  │  ✅ GGML (exact match, 8 codebooks)
│ (1 layer, 256-dim) │
└─────────┬──────────┘
          │
          ▼
    ┌──────────┐
    │ Audio    │
    │ Codes    │  [8 codebooks × N frames]
    └────┬─────┘
         │
         ▼
┌────────────────────┐
│ Audio Codec        │  ✅ GGML (306 tensors, max diff: 0.004)
│ (HiFiGAN Decoder)  │
└─────────┬──────────┘
          │
          ▼
    ┌──────────┐
    │ Audio    │
    │ Waveform │  [N samples @ 22050 Hz]
    └──────────┘
```

## Key Architecture Points

### Model Structure
- **Text Encoder**: 6 layers, 12 heads, d=768, bidirectional, conv FFN (kernel=3)
- **Decoder**: 12 layers, 12 SA heads + 1 XA head, causal, conv FFN (kernel=1)
- **Local Transformer**: 1 layer, d=256, autoregressive over 8 codebooks
- **Audio Codec**: HiFiGAN decoder, FSQ (no learned codebook)

### Special Tokens
- Text: BOS=2378, EOS=2379
- Audio: BOS=2016, EOS=2017

### KV Cache Design
- Self-attention: K/V cache per decoder layer
- Cross-attention: Computed once from encoder output, cached
- Shape: `[max_seq, n_heads * d_head]` per layer

## Reference Code
- **Nemotron ASR**: `/var/data/nvidia-speech/nemotron-asr.cpp/src/nemo-ggml.h` - Similar GGML patterns
- **NeMo source**: `.venv/lib/python3.10/site-packages/nemo/collections/tts/models/magpietts.py`

## Implementation Notes (for next session)

### Patterns established
- **Embedding lookup**: Use `ggml_get_rows(ctx, weight_matrix, indices)` - weight matrix is `[d_model, vocab_size]` in GGML column-major order
- **Reference testing**: Binary files in `test_data/reference/` have format: `4 x int64 shape header + float32 data`. Shape is GGML-reversed. Tokens are stored as float32 (converted from int64).
- **Graph building pattern**: Create context with `no_alloc=true`, build tensors, use `ggml_gallocr_reserve()` + `ggml_gallocr_alloc_graph()`, then `ggml_backend_tensor_set()` for inputs, `ggml_backend_graph_compute()`, and `ggml_backend_tensor_get()` for outputs.

### Key files to reference
- `nemotron-asr.cpp/src/nemo-ggml.cpp` - Similar GGML patterns for ASR model
- `scripts/dump_reference.py` - Generates reference tensors, shows PyTorch layer structure
- `MAGPIE_ARCHITECTURE.md` - Model architecture and tensor shapes

### Audio embedding implementation hint
```cpp
// Sum embeddings from all 8 codebooks
// codes shape: [8, seq] or [8] for single frame
// Each audio_emb_w[i] is [d_model, vocab_per_cb]
ggml_tensor * sum = ggml_get_rows(ctx, embeddings->audio_emb_w[0], codes_cb0);
for (int cb = 1; cb < 8; cb++) {
    ggml_tensor * emb = ggml_get_rows(ctx, embeddings->audio_emb_w[cb], codes_cb_i);
    sum = ggml_add(ctx, sum, emb);
}
```

### RMS norm implementation hint
```cpp
// GGML has ggml_rms_norm() built-in, then multiply by weight
ggml_tensor * norm = ggml_rms_norm(ctx, input, eps);
return ggml_mul(ctx, norm, weight);  // element-wise with broadcast
```

### Gotchas
- **Data format**: Reference files must be saved in Fortran (column-major) order for GGML compatibility
- GGML tensors are column-major, PyTorch is row-major - use `flatten('F')` when saving
- The model uses **LayerNorm** (not RMSNorm) - uses `ggml_norm()`, NOT `ggml_rms_norm()`
- **NeMo encoder uses causal self-attention** (is_causal=True) - unusual for encoders
- Attention mask must be **F32** to match attention scores type
- **ggml_permute() behavior**: To transform [a,b,c] to [b,c,a], use permute(2,0,1,3) not (1,2,0,3)

### Self-attention implementation hints
```cpp
// Causal mask must be F16 for CUDA compatibility
struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, seq_len, seq_len);
ggml_set_name(mask, "causal_mask");
ggml_set_input(mask);  // Filled at runtime

// Fill causal mask: 0 for allowed, -inf for masked
for (int i = 0; i < seq_len; i++) {
    for (int j = 0; j < seq_len; j++) {
        mask[i * seq_len + j] = (j <= i) ? 0.0f : -INFINITY;
    }
}
```

### Causal Conv1D implementation pattern
```cpp
// Pad input on left with (kernel_size - 1) zeros
struct ggml_tensor * padded = ggml_pad_ext(ctx, input, 0, 0, pad_left, 0, 0, 0, 0, 0);

// Permute weight from [kernel, in_ch, out_ch] to [in_ch, out_ch, kernel]
// Use empirically found permutation (2, 0, 1, 3)
struct ggml_tensor * w_perm = ggml_cont(ctx, ggml_permute(ctx, weight, 2, 0, 1, 3));

// Sum matmuls for each kernel position
for (int k = 0; k < kernel_size; k++) {
    struct ggml_tensor * input_k = ggml_view_2d(ctx, padded, ...offset k...);
    struct ggml_tensor * w_k = ggml_view_2d(ctx, w_perm, ...offset k...);
    struct ggml_tensor * term = ggml_mul_mat(ctx, w_k, input_k);
    output = (k == 0) ? term : ggml_add(ctx, output, term);
}
```

## Commands

```bash
# Run reference dumping
uv run scripts/dump_reference.py --text "Hello world" --output-dir test_data/reference

# Convert model to GGUF
uv run scripts/convert_magpie_to_gguf.py ../magpie_tts_multilingual_357m/magpie_tts_multilingual_357m.nemo weights/magpie-357m-q8.gguf -q q8

# Run model inspection
uv run inspect_inference.py --save-audio
```

## To Resume
1. Read this STATUS.md
2. Read GGML_PORT_PLAN.md for implementation roadmap
3. Reference `tests/test_text_embedding.cpp` for testing pattern
4. Use test_data/reference/ for layer-by-layer validation
5. Compile with `make test_text_embedding` (or other targets), run `./test_text_embedding`

## Resume Prompt (2026-01-31)

```
Continue the Magpie TTS GGML port. END-TO-END PIPELINE WORKING!

STATUS - FULL PIPELINE VERIFIED:
- Full 6-layer encoder: max diff 0.008 vs PyTorch ✅
- Full 12-layer decoder: max diff 0.003 vs PyTorch ✅
- Final projection: max diff 0.000001 vs PyTorch ✅
- Local transformer: EXACT MATCH for all 8 codebooks ✅
- Audio codec (HiFiGAN): max diff 0.004 vs PyTorch ✅
- End-to-end inference: WORKING ✅

VERIFIED TESTS:
- ./test_full_encoder_v2 (Full encoder: 0.008 max diff)
- ./test_full_decoder (Full decoder: 0.003 max diff)
- ./test_final_proj (Final projection: 0.000001 max diff)
- ./test_local_transformer (All 8 codebooks match exactly)
- ./test_codec_fsq (FSQ dequantization: exact match)
- ./test_codec_decode (Full codec: 0.004 max diff)
- ./test_e2e_inference (Full pipeline: text → audio WAV file)

USAGE:
  make test_e2e_inference && ./test_e2e_inference --text "Hello world"
  # Generates output.wav from text input (uses built-in tokenizer)

REMAINING OPTIMIZATION:
1. ~~KV cache for decoder~~ ✅ DONE (GPU-resident cache)
2. ~~EOS detection tuning~~ ✅ DONE
3. ~~Temperature/top-k sampling~~ ✅ DONE
4. ~~Graph reuse~~ ✅ DONE (batched context + allocator reuse, 154 fps)
5. Streaming/real-time support

KEY FILES:
- src/magpie.cpp, src/magpie.h - main TTS model implementation
- src/nano-codec.cpp - audio codec implementation
- src/magpie-tts.cpp - main CLI binary
- tests/test_e2e_inference.cpp - end-to-end test
- test_data/reference/ - PyTorch reference tensors (column-major format)
```

## Quick Test Commands
```bash
# Main binary (recommended)
./magpie-tts -t "Hello, how are you?" -o output.wav   # Graph-reuse (154 fps, FASTEST)
./magpie-tts -t "Hello" -o hello.wav --temp 0.5       # Custom temperature
./magpie-tts --help                                   # Show all options

# Test binaries
./test_graph_reuse --text "Hello, how are you?"        # Graph-reuse (154 fps)
./test_e2e_optimized --text "Hello, how are you?"      # GPU-optimized (134 fps)
./test_e2e_inference --text "Hello, how are you?"      # Standard (64 fps)

# Component tests
./test_full_encoder_v2   # Encoder: 6 layers, max diff 0.008
./test_full_decoder      # Decoder: 12 layers, max diff 0.003
./test_final_proj        # Final projection: max diff 0.000001
./test_local_transformer # Local transformer: exact match all 8 codebooks
./test_codec_fsq         # FSQ dequantization: exact match
./test_codec_decode      # Full codec decoder: max diff 0.004

# Performance comparison
./test_graph_reuse --compare    # Compare all versions (recommended)
./test_e2e_optimized --compare-all  # Compare GPU-optimized versions

# Generate reference data (if needed)
uv run scripts/dump_reference.py --text "Hello world" --output-dir test_data/reference
```
