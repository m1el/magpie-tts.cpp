#ifndef MAGPIE_H
#define MAGPIE_H

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cstdint>
#include <string>
#include <vector>
#include <map>

// Backend type for inference
enum magpie_backend_type {
    MAGPIE_BACKEND_CPU   = 0,
    MAGPIE_BACKEND_CUDA  = 1,
    MAGPIE_BACKEND_METAL = 2,
    MAGPIE_BACKEND_AUTO  = 3,  // Auto-detect: prefer CUDA if available
};

//
// Hyperparameters
//

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
    int32_t lt_ffn_dim      = 1024;   // local transformer FFN dimension
    int32_t lt_layers       = 1;      // local transformer layers
    int32_t lt_heads        = 1;      // single head (d_head = 256)

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

//
// Weight Structures - Encoder
//

struct magpie_encoder_layer {
    // Pre-norm for self-attention
    struct ggml_tensor * norm_self_w;        // [d_model]

    // Self-attention (bidirectional)
    struct ggml_tensor * sa_qkv_w;           // [3*d_model, d_model]
    struct ggml_tensor * sa_out_w;           // [d_model, d_model]

    // Pre-norm for FFN
    struct ggml_tensor * norm_ff_w;          // [d_model]

    // FFN with conv (kernel=3)
    struct ggml_tensor * ff_proj_w;          // [d_ffn, d_model, 3]
    struct ggml_tensor * ff_out_w;           // [d_model, d_ffn, 3]
};

struct magpie_encoder {
    struct ggml_tensor * pos_emb_w;          // [max_seq, d_model]
    std::vector<magpie_encoder_layer> layers;  // [enc_layers]
    struct ggml_tensor * norm_out_w;         // [d_model]
};

//
// Weight Structures - Decoder
//

struct magpie_decoder_layer {
    // Self-attention (causal)
    struct ggml_tensor * norm_self_w;        // [d_model]
    struct ggml_tensor * sa_qkv_w;           // [3*d_model, d_model]
    struct ggml_tensor * sa_out_w;           // [d_model, d_model]

    // Cross-attention
    struct ggml_tensor * norm_xa_q_w;        // [d_model]
    struct ggml_tensor * xa_q_w;             // [d_xa_head * n_xa_heads, d_model]
    struct ggml_tensor * xa_kv_w;            // [2 * d_xa_head * n_xa_heads, d_model]
    struct ggml_tensor * xa_out_w;           // [d_model, d_xa_head * n_xa_heads]
    struct ggml_tensor * norm_xa_mem_w;      // [d_model] for memory norm

    // FFN with conv (kernel=1)
    struct ggml_tensor * norm_ff_w;          // [d_model]
    struct ggml_tensor * ff_proj_w;          // [d_ffn, d_model, 1]
    struct ggml_tensor * ff_out_w;           // [d_model, d_ffn, 1]
};

struct magpie_decoder {
    struct ggml_tensor * pos_emb_w;          // [max_seq, d_model]
    std::vector<magpie_decoder_layer> layers;  // [dec_layers]
    struct ggml_tensor * norm_out_w;         // [d_model]
};

//
// Weight Structures - Local Transformer
//

struct magpie_local_transformer {
    struct ggml_tensor * in_proj_w;          // [lt_dim, d_model]
    struct ggml_tensor * in_proj_b;          // [lt_dim]
    struct ggml_tensor * pos_emb_w;          // [10, lt_dim]  (max 8 codebooks + margin)

    // Single layer
    struct ggml_tensor * norm_self_w;        // [lt_dim]
    struct ggml_tensor * sa_qkv_w;           // [3*lt_dim, lt_dim]
    struct ggml_tensor * sa_out_w;           // [lt_dim, lt_dim]
    struct ggml_tensor * norm_ff_w;          // [lt_dim]
    struct ggml_tensor * ff_proj_w;          // [lt_ff_dim, lt_dim, 1]
    struct ggml_tensor * ff_out_w;           // [lt_dim, lt_ff_dim, 1]

    // Per-codebook output projections
    struct ggml_tensor * out_proj_w[8];      // [vocab_per_cb, lt_dim] each
    struct ggml_tensor * out_proj_b[8];      // [vocab_per_cb] each
};

//
// Weight Structures - Embeddings
//

struct magpie_embeddings {
    struct ggml_tensor * text_emb_w;         // [text_vocab_size, d_model]
    struct ggml_tensor * audio_emb_w[8];     // [vocab_per_cb, d_model] per codebook
    struct ggml_tensor * baked_context_w;    // [num_speakers, context_frames * d_model]
};

//
// Weight Structures - Final Projection
//

struct magpie_final_proj {
    struct ggml_tensor * weight;             // [num_codebooks * vocab_per_cb, d_model]
    struct ggml_tensor * bias;               // [num_codebooks * vocab_per_cb]
};

//
// Full Model Structure
//

struct magpie_model {
    magpie_hparams hparams;

    // Weight structures
    magpie_embeddings embeddings;
    magpie_encoder encoder;
    magpie_decoder decoder;
    magpie_final_proj final_proj;
    magpie_local_transformer local_transformer;

    // GGML contexts and backends
    struct ggml_context * ctx_w;             // weights context (no allocation)
    ggml_backend_t backend;                  // compute backend (CPU/CUDA/Metal)
    ggml_backend_buffer_t buffer_w;          // weight buffer on backend
    magpie_backend_type backend_type;        // which backend is in use

    // Tensor name mapping for loading
    std::map<std::string, struct ggml_tensor *> tensors;
};

//
// KV Cache for Autoregressive Decoding
//

struct magpie_kv_cache {
    // Decoder self-attention KV cache
    // Shape: [n_layers, max_seq, n_heads * d_head]
    std::vector<struct ggml_tensor *> k_cache;  // one per decoder layer
    std::vector<struct ggml_tensor *> v_cache;  // one per decoder layer

    // Cross-attention KV cache (computed once from encoder output)
    // Shape: [n_layers, enc_seq, n_xa_heads * d_xa_head]
    std::vector<struct ggml_tensor *> xa_k_cache;  // one per decoder layer
    std::vector<struct ggml_tensor *> xa_v_cache;  // one per decoder layer

    int32_t seq_len;                  // current sequence length
    int32_t max_seq;                  // maximum sequence length
    int32_t enc_seq_len;              // encoder sequence length (for cross-attention)

    struct ggml_context * ctx;        // context for cache tensors
    ggml_backend_buffer_t buffer;     // buffer on backend

    void reset() {
        seq_len = 0;
        enc_seq_len = 0;
    }
};

//
// Runtime State
//

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
        encoder_output.clear();
        enc_seq_len = 0;
    }
};

//
// Context (combines everything)
//

struct magpie_context {
    magpie_model model;
    magpie_state state;

    // Inference settings
    int n_threads;
    float temperature;
    int top_k;
    int speaker_id;

    // Optional: audio codec for waveform generation
    struct magpie_codec * codec;

    magpie_context() : n_threads(4), temperature(0.7f), top_k(80), speaker_id(0), codec(nullptr) {}
};

//
// API Functions
//

// Initialize with automatic backend selection (prefers CUDA if available)
struct magpie_context * magpie_init(const char * model_path);

// Initialize with specific backend
struct magpie_context * magpie_init_with_backend(const char * model_path, magpie_backend_type backend);

// Free context and all associated memory
void magpie_free(struct magpie_context * ctx);

// Get current backend name
const char * magpie_get_backend_name(struct magpie_context * ctx);

// Load model weights from file (with backend selection)
bool magpie_model_load(const std::string & path, magpie_model & model, magpie_backend_type backend = MAGPIE_BACKEND_AUTO);

//
// Graph Building Functions
//

// Build encoder graph: text tokens -> encoder output
struct ggml_cgraph * magpie_build_encoder_graph(
    struct magpie_context * ctx,
    struct ggml_context * ctx0,
    struct ggml_tensor * text_tokens);

// Build decoder step graph: single autoregressive step
struct ggml_cgraph * magpie_build_decoder_step_graph(
    struct magpie_context * ctx,
    struct ggml_context * ctx0);

// Build local transformer graph: refine codebook predictions
struct ggml_tensor * magpie_build_local_transformer(
    struct magpie_context * ctx,
    struct ggml_context * ctx0,
    struct ggml_tensor * decoder_hidden,
    struct ggml_tensor * prev_codes);

// Build local transformer step 0: get logits for first codebook
struct ggml_tensor * magpie_build_local_transformer_step0(
    struct ggml_context * ctx,
    struct ggml_tensor * decoder_hidden,
    struct magpie_local_transformer * lt,
    const struct magpie_hparams * hparams);

// Build local transformer for a sequence: run transformer layer
struct ggml_tensor * magpie_build_local_transformer_seq(
    struct ggml_context * ctx,
    struct ggml_tensor * seq_input,
    struct magpie_local_transformer * lt,
    const struct magpie_hparams * hparams);

// Full local transformer sampling: autoregressively sample all 8 codebooks
std::vector<int32_t> magpie_local_transformer_sample_all(
    struct magpie_context * mctx,
    const float * decoder_hidden_data,
    float temperature,
    int top_k);

//
// Component Building Functions
//

// Build text embedding: token indices -> embeddings
struct ggml_tensor * magpie_build_text_embedding(
    struct ggml_context * ctx,
    struct ggml_tensor * tokens,
    struct magpie_embeddings * embeddings);

// Build audio embedding: sum of codebook embeddings
struct ggml_tensor * magpie_build_audio_embedding(
    struct ggml_context * ctx,
    struct ggml_tensor * codes,       // [num_codebooks] or [num_codebooks, seq]
    struct magpie_embeddings * embeddings);

// Build encoder layer: bidirectional self-attention + conv FFN
struct ggml_tensor * magpie_build_encoder_layer(
    struct ggml_context * ctx,
    struct ggml_tensor * input,       // [d_model, seq]
    struct ggml_tensor * pos_emb,     // position embeddings
    struct magpie_encoder_layer * layer,
    const magpie_hparams * hparams);

// Build encoder layer with shared mask (for efficiency when stacking layers)
struct ggml_tensor * magpie_build_encoder_layer_with_mask(
    struct ggml_context * ctx,
    struct ggml_tensor * input,       // [d_model, seq]
    struct ggml_tensor * pos_emb,     // position embeddings
    struct magpie_encoder_layer * layer,
    const magpie_hparams * hparams,
    struct ggml_tensor * shared_mask);  // pre-created F16 causal mask

// Build full encoder: text embedding + position embeddings + 6 layers + final norm
struct ggml_tensor * magpie_build_full_encoder(
    struct ggml_context * ctx,
    struct ggml_tensor * input,       // [d_model, seq] (already embedded)
    struct magpie_encoder * encoder,
    const magpie_hparams * hparams);

// Build decoder layer: causal self-attention + cross-attention + FFN
struct ggml_tensor * magpie_build_decoder_layer(
    struct ggml_context * ctx,
    struct ggml_tensor * input,       // [d_model, 1] (current step)
    struct ggml_tensor * encoder_out, // [d_model, enc_seq]
    int layer_idx,
    struct magpie_decoder_layer * layer,
    struct magpie_kv_cache * kv_cache,
    const magpie_hparams * hparams);

// Build single-step cached decoder layer for autoregressive generation
// Uses KV cache for self-attention, pre-cached K/V for cross-attention
struct ggml_tensor * magpie_build_decoder_layer_cached(
    struct ggml_context * ctx,
    struct ggml_tensor * input,           // [d_model, 1] - current step
    struct ggml_tensor * xa_k_cached,     // [d_xa_head * n_xa_heads, enc_seq]
    struct ggml_tensor * xa_v_cached,     // [d_xa_head * n_xa_heads, enc_seq]
    struct ggml_tensor * sa_k_cache_in,   // [d_model, cache_len] or nullptr for first step
    struct ggml_tensor * sa_v_cache_in,   // [d_model, cache_len] or nullptr
    int layer_idx,
    struct magpie_decoder_layer * layer,
    const magpie_hparams * hparams,
    int pos_offset,                       // position offset for positional embeddings
    struct ggml_tensor * pos_emb_w,       // [d_model, max_seq]
    struct ggml_tensor ** sa_k_cache_out, // output: updated K cache
    struct ggml_tensor ** sa_v_cache_out);// output: updated V cache

// Build RMS norm (legacy - prefer layer_norm)
struct ggml_tensor * magpie_build_rms_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    struct ggml_tensor * weight,
    float eps);

// Build Layer norm (without bias)
// LayerNorm: (x - mean(x)) / sqrt(var(x) + eps) * weight
struct ggml_tensor * magpie_build_layer_norm(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    struct ggml_tensor * weight,
    float eps);

// Add position embeddings
struct ggml_tensor * magpie_build_add_position_embeddings(
    struct ggml_context * ctx,
    struct ggml_tensor * input,       // [d_model, seq]
    struct ggml_tensor * pos_emb_w,   // [d_model, max_seq]
    int offset);                       // starting position (for KV cache)

// Build self-attention (bidirectional or causal)
struct ggml_tensor * magpie_build_self_attention(
    struct ggml_context * ctx,
    struct ggml_tensor * input,       // [d_model, seq]
    struct ggml_tensor * qkv_weight,  // [3*d_model, d_model]
    struct ggml_tensor * out_weight,  // [d_model, d_model]
    int n_heads,
    bool is_causal);

// Build self-attention with shared mask (for efficiency when stacking layers)
struct ggml_tensor * magpie_build_self_attention_with_mask(
    struct ggml_context * ctx,
    struct ggml_tensor * input,       // [d_model, seq]
    struct ggml_tensor * qkv_weight,  // [3*d_model, d_model]
    struct ggml_tensor * out_weight,  // [d_model, d_model]
    int n_heads,
    bool is_causal,
    struct ggml_tensor * shared_mask);  // pre-created F16 causal mask, or nullptr

// Build self-attention with KV cache (for autoregressive decoding)
// Processes a single step, using cached K/V from previous steps
// k_cache_in/v_cache_in: [d_model, cache_len] or nullptr for first step
// k_cache_out/v_cache_out: [d_model, cache_len + 1] updated cache
struct ggml_tensor * magpie_build_self_attention_cached(
    struct ggml_context * ctx,
    struct ggml_tensor * input,       // [d_model, 1] - single step
    struct ggml_tensor * qkv_weight,  // [3*d_model, d_model]
    struct ggml_tensor * out_weight,  // [d_model, d_model]
    struct ggml_tensor * k_cache_in,  // [d_model, cache_len] or nullptr
    struct ggml_tensor * v_cache_in,  // [d_model, cache_len] or nullptr
    int n_heads,
    struct ggml_tensor ** k_cache_out,  // output: [d_model, cache_len + 1]
    struct ggml_tensor ** v_cache_out); // output: [d_model, cache_len + 1]

// Build cross-attention with pre-cached K/V from encoder
// K/V are precomputed from encoder output and cached
struct ggml_tensor * magpie_build_cross_attention_cached(
    struct ggml_context * ctx,
    struct ggml_tensor * query,       // [d_model, 1] - decoder current step
    struct ggml_tensor * k_cached,    // [d_xa_head * n_xa_heads, enc_seq] - pre-computed K
    struct ggml_tensor * v_cached,    // [d_xa_head * n_xa_heads, enc_seq] - pre-computed V
    struct ggml_tensor * q_weight,    // [d_xa_head * n_xa_heads, d_model]
    struct ggml_tensor * out_weight,  // [d_model, d_xa_head * n_xa_heads]
    int n_heads,
    int d_head);

// Precompute cross-attention K/V from encoder output (call once per utterance)
void magpie_precompute_cross_attention_kv(
    struct ggml_context * ctx,
    struct ggml_tensor * encoder_out,  // [d_model, enc_seq]
    struct ggml_tensor * kv_weight,    // [2 * d_xa_head * n_xa_heads, d_model]
    struct ggml_tensor * norm_mem_w,   // [d_model] memory norm weight
    float eps,
    struct ggml_tensor ** k_out,       // output: [d_xa_head * n_xa_heads, enc_seq]
    struct ggml_tensor ** v_out);      // output: [d_xa_head * n_xa_heads, enc_seq]

// Build conv-based feed-forward network
struct ggml_tensor * magpie_build_conv_ffn(
    struct ggml_context * ctx,
    struct ggml_tensor * input,       // [d_model, seq]
    struct ggml_tensor * proj_weight, // [d_ffn, d_model, kernel]
    struct ggml_tensor * out_weight,  // [d_model, d_ffn, kernel]
    int kernel_size);                  // 3 for encoder, 1 for decoder

// Build cross-attention (decoder attending to encoder output)
// Query from decoder, Key/Value from encoder
struct ggml_tensor * magpie_build_cross_attention(
    struct ggml_context * ctx,
    struct ggml_tensor * query,       // [d_model, dec_seq] - decoder hidden states
    struct ggml_tensor * memory,      // [d_model, enc_seq] - encoder output
    struct ggml_tensor * q_weight,    // [d_xa_head * n_xa_heads, d_model]
    struct ggml_tensor * kv_weight,   // [2 * d_xa_head * n_xa_heads, d_model]
    struct ggml_tensor * out_weight,  // [d_model, d_xa_head * n_xa_heads]
    int n_heads,
    int d_head);

// Build final projection: decoder output -> logits
struct ggml_tensor * magpie_build_final_proj(
    struct ggml_context * ctx,
    struct ggml_tensor * input,       // [d_model]
    struct magpie_final_proj * proj);

//
// Inference Functions
//

// Encode text tokens to encoder output
bool magpie_encode_text(
    struct magpie_context * ctx,
    const int32_t * tokens,
    int n_tokens);

// Run one decoder step, returns logits
bool magpie_decode_step(
    struct magpie_context * ctx,
    std::vector<float> & logits);

// Sample next frame codes from logits
std::vector<int32_t> magpie_sample_frame(
    struct magpie_context * ctx,
    const std::vector<float> & logits);

// Full synthesis: text tokens -> audio codes
std::vector<int32_t> magpie_synthesize_codes(
    struct magpie_context * ctx,
    const int32_t * tokens,
    int n_tokens);

// Optimized synthesis using KV cache (O(n) per step instead of O(n²))
std::vector<int32_t> magpie_synthesize_codes_cached(
    struct magpie_context * ctx,
    const int32_t * tokens,
    int n_tokens);

//
// Audio Codec (separate model)
//

// Codec hyperparameters
struct magpie_codec_hparams {
    int32_t sample_rate     = 22050;
    int32_t num_codebooks   = 8;
    int32_t codebook_size   = 2016;
    int32_t hop_length      = 1024;
    int32_t latent_dim      = 32;    // 8 codebooks * 4 dims per cb

    // FSQ levels per codebook (same for all 8)
    int32_t fsq_levels[4]   = {8, 7, 6, 6};

    // HiFiGAN decoder structure
    int32_t pre_conv_kernel     = 7;
    int32_t post_conv_kernel    = 3;
    int32_t base_channels       = 864;

    // Upsample rates: 8, 8, 4, 2, 2 (product = 1024 = hop_length)
    int32_t num_upsample_layers = 5;
    int32_t up_sample_rates[5]  = {8, 8, 4, 2, 2};
    int32_t up_channels[5]      = {432, 216, 108, 54, 27};  // out channels

    // Residual block structure
    int32_t resblock_kernel_sizes[3] = {3, 7, 11};
    int32_t resblock_dilations[3]    = {1, 3, 5};
};

// Single residual block weights
struct magpie_codec_resblock {
    struct ggml_tensor * input_act_alpha;   // [1, ch/2, 1] HalfSnake alpha
    struct ggml_tensor * input_conv_w;      // [out_ch, in_ch, kernel]
    struct ggml_tensor * input_conv_b;      // [out_ch]
    struct ggml_tensor * skip_act_alpha;    // [1, ch/2, 1] HalfSnake alpha
    struct ggml_tensor * skip_conv_w;       // [out_ch, in_ch, kernel]
    struct ggml_tensor * skip_conv_b;       // [out_ch]
};

// HiFiGAN residual block (3 inner blocks with different dilations)
struct magpie_codec_hifigan_resblock {
    std::vector<magpie_codec_resblock> inner_blocks;  // [3] for dilations 1,3,5
};

// HiFiGAN residual layer (3 blocks with different kernels, averaged)
struct magpie_codec_reslayer {
    std::vector<magpie_codec_hifigan_resblock> res_blocks;  // [3] for kernels 3,7,11
};

// Upsample stage
struct magpie_codec_upsample {
    struct ggml_tensor * act_alpha;     // [1, in_ch/2, 1] HalfSnake alpha
    struct ggml_tensor * conv_w;        // [out_ch, 1, kernel] (groups=out_ch)
    struct ggml_tensor * conv_b;        // [out_ch]
};

// FSQ quantizer (parameters only, decode is formula-based)
struct magpie_codec_fsq {
    struct ggml_tensor * dim_base_index;  // [1, 4, 1] = {1, 8, 56, 336}
    struct ggml_tensor * num_levels;      // [1, 4, 1] = {8, 7, 6, 6}
};

// Full codec structure
struct magpie_codec {
    magpie_codec_hparams hparams;

    // Pre-conv: latent_dim -> base_channels
    struct ggml_tensor * pre_conv_w;    // [base_ch, latent_dim, kernel]
    struct ggml_tensor * pre_conv_b;    // [base_ch]

    // Upsample layers (5 stages)
    std::vector<magpie_codec_upsample> upsample_layers;  // [5]

    // Residual layers (5 stages, one per upsample)
    std::vector<magpie_codec_reslayer> res_layers;  // [5]

    // Post-conv: final_ch -> 1
    struct ggml_tensor * post_act_alpha;  // [1, final_ch/2, 1]
    struct ggml_tensor * post_conv_w;     // [1, final_ch, kernel]
    struct ggml_tensor * post_conv_b;     // [1]

    // FSQ parameters (8 codebooks)
    std::vector<magpie_codec_fsq> fsqs;  // [8]

    // GGML contexts and backends
    struct ggml_context * ctx_w;
    ggml_backend_t backend;
    ggml_backend_buffer_t buffer_w;
    magpie_backend_type backend_type;

    // Tensor name mapping
    std::map<std::string, struct ggml_tensor *> tensors;
};

// Initialize audio codec
struct magpie_codec * magpie_codec_init(const char * codec_path);
struct magpie_codec * magpie_codec_init_with_backend(const char * codec_path, magpie_backend_type backend);

// Free codec
void magpie_codec_free(struct magpie_codec * codec);

// Load codec weights from GGUF file
bool magpie_codec_load(const std::string & path, magpie_codec & codec, magpie_backend_type backend = MAGPIE_BACKEND_AUTO);

// Decode audio codes to waveform
std::vector<float> magpie_codec_decode(
    struct magpie_codec * codec,
    const int32_t * codes,  // [num_codebooks, n_frames]
    int n_frames);

//
// Codec Graph Building Functions
//

// Build FSQ dequantization: codes [num_cb, T] -> latent [latent_dim, T]
struct ggml_tensor * magpie_codec_build_fsq_dequant(
    struct ggml_context * ctx,
    struct ggml_tensor * codes,       // [num_codebooks, n_frames] int32
    struct magpie_codec * codec);

// Build HalfSnake activation: x + (1/alpha) * sin²(alpha * x) on first half, LeakyReLU on second
struct ggml_tensor * magpie_codec_build_half_snake(
    struct ggml_context * ctx,
    struct ggml_tensor * input,       // [channels, T]
    struct ggml_tensor * alpha);      // [1, channels/2, 1]

// Build causal conv1d with left padding
struct ggml_tensor * magpie_codec_build_causal_conv1d(
    struct ggml_context * ctx,
    struct ggml_tensor * input,       // [in_ch, T]
    struct ggml_tensor * weight,      // [out_ch, in_ch, kernel]
    struct ggml_tensor * bias,        // [out_ch]
    int dilation = 1);

// Build causal conv transpose 1d (upsample)
struct ggml_tensor * magpie_codec_build_conv_transpose1d(
    struct ggml_context * ctx,
    struct ggml_tensor * input,       // [in_ch, T]
    struct ggml_tensor * weight,      // [out_ch, 1, kernel] (groups=out_ch means depthwise)
    struct ggml_tensor * bias,        // [out_ch]
    int stride);

// Build single residual block
struct ggml_tensor * magpie_codec_build_residual_block(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    struct magpie_codec_resblock * block,
    int dilation);

// Build HiFiGAN residual block (3 inner blocks)
struct ggml_tensor * magpie_codec_build_hifigan_resblock(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    struct magpie_codec_hifigan_resblock * block);

// Build HiFiGAN residual layer (3 blocks averaged)
struct ggml_tensor * magpie_codec_build_reslayer(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    struct magpie_codec_reslayer * layer);

// Build full HiFiGAN decoder graph
struct ggml_tensor * magpie_codec_build_decoder(
    struct ggml_context * ctx,
    struct ggml_tensor * latent,      // [latent_dim, T]
    struct magpie_codec * codec);

//
// Utility Functions
//

// Check if frame contains EOS token
bool magpie_is_eos(const std::vector<int32_t> & frame_codes, int32_t eos_id);

// Get baked speaker context
struct ggml_tensor * magpie_get_baked_context(
    struct ggml_context * ctx,
    struct magpie_embeddings * embeddings,
    int speaker_id,
    int context_frames,
    int d_model);

#endif // MAGPIE_H
