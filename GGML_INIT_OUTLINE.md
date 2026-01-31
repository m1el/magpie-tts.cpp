# Magpie TTS GGML Initialization Architecture

This document outlines how the GGML context, graphs, and model initialization will work for Magpie TTS.

## Overview

The architecture follows the pattern used in nemotron-asr.cpp with adaptations for:
1. **Transformer encoder** (bidirectional, 6 layers)
2. **Transformer decoder** (causal with cross-attention, 12 layers, KV cache)
3. **Local transformer** (autoregressive codebook refinement, 1 layer)
4. **Audio codec decoder** (HiFiGAN-based, separate model)

## Core Structures

### 1. Hyperparameters (`magpie_hparams`)

```cpp
struct magpie_hparams {
    // Model dimensions
    int32_t d_model         = 768;    // hidden dimension
    int32_t d_ffn           = 3072;   // feedforward dimension
    int32_t d_head          = 64;     // attention head dimension

    // Encoder
    int32_t enc_layers      = 6;      // text encoder layers
    int32_t enc_heads       = 12;     // encoder self-attention heads
    int32_t enc_kernel      = 3;      // encoder conv kernel size

    // Decoder
    int32_t dec_layers      = 12;     // main decoder layers
    int32_t dec_sa_heads    = 12;     // decoder self-attention heads
    int32_t dec_xa_heads    = 1;      // decoder cross-attention heads
    int32_t dec_xa_d_head   = 128;    // cross-attention head dim
    int32_t dec_kernel      = 1;      // decoder conv kernel size (pointwise)

    // Local transformer
    int32_t lt_dim          = 256;    // local transformer dimension
    int32_t lt_layers       = 1;      // local transformer layers
    int32_t lt_heads        = 4;      // inferred from lt_dim and qkv size

    // Vocabulary
    int32_t text_vocab_size = 2380;   // phoneme vocabulary
    int32_t num_codebooks   = 8;      // audio codebooks
    int32_t codebook_size   = 2016;   // codebook entries
    int32_t vocab_per_cb    = 2024;   // codebook + special tokens

    // Context
    int32_t num_speakers    = 5;      // baked speaker embeddings
    int32_t context_frames  = 110;    // frames per baked speaker

    // Special tokens
    int32_t text_bos_id     = 2378;
    int32_t text_eos_id     = 2379;
    int32_t audio_bos_id    = 2016;
    int32_t audio_eos_id    = 2017;

    // Inference
    int32_t max_dec_steps   = 500;    // maximum decoder steps
    int32_t sample_rate     = 22050;

    float   eps             = 1e-5f;  // layer norm epsilon
};
```

### 2. Weight Structures

#### Text Encoder Weights
```cpp
struct magpie_encoder_layer {
    // Pre-norm for self-attention
    ggml_tensor * norm_self_w;        // [d_model]

    // Self-attention (bidirectional)
    ggml_tensor * sa_qkv_w;           // [3*d_model, d_model]
    ggml_tensor * sa_out_w;           // [d_model, d_model]

    // Pre-norm for FFN
    ggml_tensor * norm_ff_w;          // [d_model]

    // FFN with conv (kernel=3)
    ggml_tensor * ff_proj_w;          // [d_ffn, d_model, 3]
    ggml_tensor * ff_out_w;           // [d_model, d_ffn, 3]
};

struct magpie_encoder {
    ggml_tensor * pos_emb_w;          // [max_seq, d_model]
    std::vector<magpie_encoder_layer> layers;  // [enc_layers]
    ggml_tensor * norm_out_w;         // [d_model]
};
```

#### Decoder Weights
```cpp
struct magpie_decoder_layer {
    // Self-attention (causal)
    ggml_tensor * norm_self_w;        // [d_model]
    ggml_tensor * sa_qkv_w;           // [3*d_model, d_model]
    ggml_tensor * sa_out_w;           // [d_model, d_model]

    // Cross-attention
    ggml_tensor * norm_xa_q_w;        // [d_model]
    ggml_tensor * xa_q_w;             // [d_xa_head * n_xa_heads, d_model]
    ggml_tensor * xa_kv_w;            // [2 * d_xa_head * n_xa_heads, d_model]
    ggml_tensor * xa_out_w;           // [d_model, d_xa_head * n_xa_heads]
    ggml_tensor * norm_xa_mem_w;      // [d_model] for memory norm

    // FFN with conv (kernel=1)
    ggml_tensor * norm_ff_w;          // [d_model]
    ggml_tensor * ff_proj_w;          // [d_ffn, d_model, 1]
    ggml_tensor * ff_out_w;           // [d_model, d_ffn, 1]
};

struct magpie_decoder {
    ggml_tensor * pos_emb_w;          // [max_seq, d_model]
    std::vector<magpie_decoder_layer> layers;  // [dec_layers]
    ggml_tensor * norm_out_w;         // [d_model]
};
```

#### Local Transformer Weights
```cpp
struct magpie_local_transformer {
    ggml_tensor * in_proj_w;          // [lt_dim, d_model]
    ggml_tensor * in_proj_b;          // [lt_dim]
    ggml_tensor * pos_emb_w;          // [10, lt_dim]  (max 8 codebooks + margin)

    // Single layer
    ggml_tensor * norm_self_w;        // [lt_dim]
    ggml_tensor * sa_qkv_w;           // [3*lt_dim, lt_dim]
    ggml_tensor * sa_out_w;           // [lt_dim, lt_dim]
    ggml_tensor * norm_ff_w;          // [lt_dim]
    ggml_tensor * ff_proj_w;          // [lt_ff_dim, lt_dim, 1]
    ggml_tensor * ff_out_w;           // [lt_dim, lt_ff_dim, 1]

    // Per-codebook output projections
    ggml_tensor * out_proj_w[8];      // [vocab_per_cb, lt_dim] each
    ggml_tensor * out_proj_b[8];      // [vocab_per_cb] each
};
```

#### Embeddings
```cpp
struct magpie_embeddings {
    ggml_tensor * text_emb_w;         // [text_vocab_size, d_model]
    ggml_tensor * audio_emb_w[8];     // [vocab_per_cb, d_model] per codebook
    ggml_tensor * baked_context_w;    // [num_speakers, context_frames * d_model]
};
```

#### Final Projection
```cpp
struct magpie_final_proj {
    ggml_tensor * weight;             // [num_codebooks * vocab_per_cb, d_model]
    ggml_tensor * bias;               // [num_codebooks * vocab_per_cb]
};
```

### 3. Full Model Structure

```cpp
struct magpie_model {
    magpie_hparams hparams;

    // Weight structures
    magpie_embeddings embeddings;
    magpie_encoder encoder;
    magpie_decoder decoder;
    magpie_final_proj final_proj;
    magpie_local_transformer local_transformer;

    // GGML contexts and backends
    ggml_context * ctx_w;             // weights context (no allocation)
    ggml_backend_t backend;           // compute backend (CPU/CUDA/Metal)
    ggml_backend_buffer_t buffer_w;   // weight buffer on backend

    // Tensor name mapping for loading
    std::map<std::string, ggml_tensor *> tensors;
};
```

### 4. KV Cache for Autoregressive Decoding

```cpp
struct magpie_kv_cache {
    // Decoder self-attention KV cache
    // Shape: [n_layers, max_seq, n_heads * d_head]
    std::vector<ggml_tensor *> k_cache;  // one per decoder layer
    std::vector<ggml_tensor *> v_cache;  // one per decoder layer

    // Cross-attention KV cache (computed once from encoder output)
    // Shape: [n_layers, enc_seq, n_xa_heads * d_xa_head]
    std::vector<ggml_tensor *> xa_k_cache;  // one per decoder layer
    std::vector<ggml_tensor *> xa_v_cache;  // one per decoder layer

    int32_t seq_len;                  // current sequence length
    int32_t max_seq;                  // maximum sequence length

    ggml_context * ctx;               // context for cache tensors
    ggml_backend_buffer_t buffer;     // buffer on backend

    void reset() { seq_len = 0; }
};
```

### 5. Runtime State

```cpp
struct magpie_state {
    // Allocator for compute graphs
    ggml_gallocr_t allocr;

    // KV cache for decoder
    magpie_kv_cache kv_cache;

    // Current generation state
    std::vector<int32_t> generated_codes;  // [num_codebooks * n_frames]
    int32_t n_generated_frames;

    // Cached encoder output
    std::vector<float> encoder_output;  // [enc_seq, d_model]
    int32_t enc_seq_len;

    void reset() {
        kv_cache.reset();
        generated_codes.clear();
        n_generated_frames = 0;
    }
};
```

### 6. Context (combines everything)

```cpp
struct magpie_context {
    magpie_model model;
    magpie_state state;

    // Inference settings
    int n_threads = 4;
    float temperature = 0.7f;
    int top_k = 80;
    int speaker_id = 0;

    // Optional: audio codec for waveform generation
    struct magpie_codec * codec = nullptr;
};
```

## Initialization Flow

### 1. Load Model (`magpie_init`)

```cpp
magpie_context * magpie_init(const char * model_path, magpie_backend_type backend) {
    magpie_context * ctx = new magpie_context();

    // 1. Initialize backend (CPU, CUDA, or Metal)
    if (!init_backend(ctx->model, backend)) {
        return nullptr;
    }

    // 2. Open GGUF file and read metadata
    gguf_context * gguf_ctx = gguf_init_from_file(model_path, {.no_alloc = true, ...});

    // 3. Read hyperparameters from GGUF KV pairs
    read_hparams(gguf_ctx, ctx->model.hparams);

    // 4. Create weight context (no_alloc = true, just tensor metadata)
    ctx->model.ctx_w = ggml_init({
        .mem_size = tensor_overhead * n_tensors,
        .no_alloc = true,
    });

    // 5. Create tensors from GGUF metadata and map to model structure
    create_tensors(gguf_ctx, ctx->model);

    // 6. Allocate backend buffer for all weights
    ctx->model.buffer_w = ggml_backend_alloc_ctx_tensors(ctx->model.ctx_w, ctx->model.backend);

    // 7. Load tensor data from GGUF file into backend buffer
    load_tensor_data(gguf_ctx, ctx->model);

    // 8. Initialize KV cache
    init_kv_cache(ctx);

    // 9. Create graph allocator
    ctx->state.allocr = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(ctx->model.backend)
    );

    return ctx;
}
```

### 2. Create Tensors and Map to Structure

```cpp
void create_tensors(gguf_context * gguf_ctx, magpie_model & model) {
    // For each tensor in GGUF:
    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        ggml_tensor * t = ggml_dup_tensor(model.ctx_w, meta_tensor);
        ggml_set_name(t, name);

        // Map to model structure based on name
        if (strcmp(name, "text_embedding.weight") == 0) {
            model.embeddings.text_emb_w = t;
        } else if (strncmp(name, "encoder.layers.", 15) == 0) {
            // Parse layer index and component
            int layer_idx = parse_layer_idx(name);
            map_encoder_layer_tensor(name, t, model.encoder.layers[layer_idx]);
        } else if (strncmp(name, "decoder.layers.", 15) == 0) {
            // ... similar for decoder
        }
        // ... etc

        model.tensors[name] = t;
    }
}
```

### 3. Initialize KV Cache

```cpp
void init_kv_cache(magpie_context * ctx) {
    const auto & hp = ctx->model.hparams;
    int max_seq = hp.max_dec_steps + hp.context_frames + 10;  // margin

    // Create context for cache tensors
    size_t cache_size = sizeof(float) * max_seq * hp.d_model * hp.dec_layers * 4;
    ctx->state.kv_cache.ctx = ggml_init({.mem_size = cache_size, .no_alloc = true});

    // Create cache tensors
    for (int l = 0; l < hp.dec_layers; l++) {
        ctx->state.kv_cache.k_cache.push_back(
            ggml_new_tensor_2d(ctx->state.kv_cache.ctx, GGML_TYPE_F16,
                               hp.dec_sa_heads * hp.d_head, max_seq)
        );
        // ... similar for v_cache, xa_k_cache, xa_v_cache
    }

    // Allocate cache buffer on backend
    ctx->state.kv_cache.buffer = ggml_backend_alloc_ctx_tensors(
        ctx->state.kv_cache.ctx, ctx->model.backend
    );

    ctx->state.kv_cache.max_seq = max_seq;
}
```

## Graph Building

### 1. Graph Creation Pattern

Each inference step creates a new computation graph:

```cpp
ggml_cgraph * build_encode_graph(magpie_context * ctx, ggml_tensor * text_tokens) {
    // 1. Create temporary context for graph building
    ggml_context * ctx0 = ggml_init({.mem_size = graph_mem_size, .no_alloc = true});

    // 2. Build computation graph
    ggml_tensor * output = build_encoder(ctx0, text_tokens, &ctx->model);

    // 3. Create graph
    ggml_cgraph * gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, output);

    return gf;
}
```

### 2. Memory Allocation with gallocr

```cpp
void run_graph(magpie_context * ctx, ggml_cgraph * gf) {
    // 1. Reserve memory for graph
    ggml_gallocr_reserve(ctx->state.allocr, gf);

    // 2. Allocate tensors
    ggml_gallocr_alloc_graph(ctx->state.allocr, gf);

    // 3. Run computation
    ggml_backend_graph_compute(ctx->model.backend, gf);
}
```

## Inference Pipeline

### Full TTS Pipeline

```cpp
std::vector<float> magpie_synthesize(
    magpie_context * ctx,
    const int32_t * text_tokens,
    int n_tokens
) {
    // 1. Reset state
    ctx->state.reset();

    // 2. Encode text
    encode_text(ctx, text_tokens, n_tokens);

    // 3. Get baked context
    prepare_context(ctx, ctx->speaker_id);

    // 4. Autoregressive decoding
    while (ctx->state.n_generated_frames < ctx->model.hparams.max_dec_steps) {
        // Build decoder step graph
        ggml_cgraph * gf = build_decoder_step_graph(ctx);
        run_graph(ctx, gf);

        // Get logits, apply local transformer, sample codes
        std::vector<int32_t> frame_codes = sample_frame(ctx, gf);

        // Check for EOS
        if (is_eos(frame_codes)) break;

        // Append to generated codes
        append_codes(ctx, frame_codes);

        // Update KV cache position
        ctx->state.kv_cache.seq_len++;

        // Free temporary graph context
        ggml_free(gf->ctx);
    }

    // 5. Decode audio with codec
    return decode_audio(ctx->codec, ctx->state.generated_codes);
}
```

### Decoder Step with KV Cache

```cpp
ggml_cgraph * build_decoder_step_graph(magpie_context * ctx) {
    ggml_context * ctx0 = ggml_init(...);

    // Get current audio codes
    ggml_tensor * codes = get_current_codes(ctx0, ctx);

    // Embed and sum across codebooks
    ggml_tensor * audio_emb = build_audio_embedding(ctx0, codes, &ctx->model);

    // For first step: prepend context embedding
    if (ctx->state.kv_cache.seq_len == 0) {
        ggml_tensor * context = get_baked_context(ctx0, ctx);
        audio_emb = ggml_concat(ctx0, context, audio_emb, 1);
    }

    // Add position embedding (offset by current position)
    int pos = ctx->state.kv_cache.seq_len;
    audio_emb = ggml_add(ctx0, audio_emb,
        ggml_view_2d(ctx0, ctx->model.decoder.pos_emb_w,
                     hp.d_model, 1, ..., pos * hp.d_model * sizeof(float)));

    // Run decoder layers with KV cache
    ggml_tensor * x = audio_emb;
    for (int l = 0; l < hp.dec_layers; l++) {
        x = build_decoder_layer_with_cache(ctx0, x, l, ctx);
    }

    // Final norm
    x = build_rms_norm(ctx0, x, ctx->model.decoder.norm_out_w);

    // Final projection -> logits
    ggml_tensor * logits = build_final_proj(ctx0, x, &ctx->model.final_proj);

    ggml_cgraph * gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, logits);

    return gf;
}
```

## Audio Codec Model (Separate)

The audio codec is a separate model for decoding audio codes to waveform:

```cpp
struct magpie_codec {
    // Hyperparameters
    int32_t latent_dim = 32;
    int32_t num_codebooks = 8;
    int32_t hop_length = 1024;

    // FSQ parameters (no learned codebook, uses formula)
    int32_t fsq_levels[4];  // per-dimension quantization levels

    // HiFiGAN decoder weights
    // ... (upsample convs, residual blocks, etc.)

    ggml_context * ctx_w;
    ggml_backend_t backend;
    ggml_backend_buffer_t buffer_w;
};

std::vector<float> magpie_codec_decode(
    magpie_codec * codec,
    const int32_t * codes,  // [num_codebooks, n_frames]
    int n_frames
);
```

## File Structure

```
magpie.cpp/
├── src/
│   ├── magpie.h              # Main API and structures
│   ├── magpie.cpp            # Initialization and high-level inference
│   ├── magpie-encoder.cpp    # Encoder graph building
│   ├── magpie-decoder.cpp    # Decoder graph building with KV cache
│   ├── magpie-local.cpp      # Local transformer
│   ├── magpie-codec.h        # Audio codec structures
│   ├── magpie-codec.cpp      # Audio codec implementation
│   └── magpie-sample.cpp     # Sampling utilities (top-k, temperature)
├── scripts/
│   └── ...
└── examples/
    └── main.cpp              # Example usage
```

## Key Differences from Nemotron ASR

| Aspect | Nemotron ASR | Magpie TTS |
|--------|--------------|------------|
| Direction | Audio → Text | Text → Audio |
| Encoder | Conformer (conv + attention) | Transformer (attention only) |
| Decoder | LSTM (stateful) | Transformer (KV cache) |
| Cross-attention | None | Every decoder layer |
| Output | Token sequence | Audio codes → waveform |
| Codec | Mel spectrogram input | HiFiGAN decoder output |

## Implementation Order

1. **Phase 1: Core structures and loading**
   - Define all structures in `magpie.h`
   - Implement `magpie_init` and GGUF loading
   - Test: Load model, verify tensor shapes

2. **Phase 2: Encoder**
   - Implement text embedding
   - Implement encoder self-attention (bidirectional)
   - Implement conv-based FFN
   - Test: Compare encoder output with reference

3. **Phase 3: Decoder with KV cache**
   - Implement KV cache management
   - Implement causal self-attention with cache
   - Implement cross-attention with cache
   - Test: Compare decoder output with reference

4. **Phase 4: Local transformer + sampling**
   - Implement local transformer
   - Implement top-k sampling
   - Test: Generate codes, compare with reference

5. **Phase 5: Audio codec**
   - Implement FSQ dequantization
   - Implement HiFiGAN decoder
   - Test: Decode codes to audio

6. **Phase 6: End-to-end**
   - Full inference pipeline
   - Performance optimization
   - Quantization testing
