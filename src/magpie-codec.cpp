#include "magpie.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <regex>

//
// Codec backend initialization (reuse pattern from magpie.cpp)
//

static bool init_codec_backend(magpie_codec & codec, magpie_backend_type backend) {
    codec.backend_type = backend;

    if (backend == MAGPIE_BACKEND_AUTO) {
#ifdef GGML_USE_CUDA
        backend = MAGPIE_BACKEND_CUDA;
#elif defined(GGML_USE_METAL)
        backend = MAGPIE_BACKEND_METAL;
#else
        backend = MAGPIE_BACKEND_CPU;
#endif
        codec.backend_type = backend;
    }

    switch (backend) {
        case MAGPIE_BACKEND_CUDA:
#ifdef GGML_USE_CUDA
            codec.backend = ggml_backend_cuda_init(0);
            if (!codec.backend) {
                fprintf(stderr, "magpie_codec: failed to init CUDA, falling back to CPU\n");
                codec.backend = ggml_backend_cpu_init();
                codec.backend_type = MAGPIE_BACKEND_CPU;
            }
#else
            fprintf(stderr, "magpie_codec: CUDA not compiled in, using CPU\n");
            codec.backend = ggml_backend_cpu_init();
            codec.backend_type = MAGPIE_BACKEND_CPU;
#endif
            break;

        case MAGPIE_BACKEND_METAL:
#ifdef GGML_USE_METAL
            codec.backend = ggml_backend_metal_init();
            if (!codec.backend) {
                fprintf(stderr, "magpie_codec: failed to init Metal, falling back to CPU\n");
                codec.backend = ggml_backend_cpu_init();
                codec.backend_type = MAGPIE_BACKEND_CPU;
            }
#else
            fprintf(stderr, "magpie_codec: Metal not compiled in, using CPU\n");
            codec.backend = ggml_backend_cpu_init();
            codec.backend_type = MAGPIE_BACKEND_CPU;
#endif
            break;

        case MAGPIE_BACKEND_CPU:
        default:
            codec.backend = ggml_backend_cpu_init();
            codec.backend_type = MAGPIE_BACKEND_CPU;
            break;
    }

    return codec.backend != nullptr;
}

//
// Hyperparameter loading
//

static void read_codec_hparams(gguf_context * gguf_ctx, magpie_codec_hparams & hparams) {
    auto get_i32 = [&](const char * key, int32_t def) -> int32_t {
        int idx = gguf_find_key(gguf_ctx, key);
        return idx >= 0 ? (int32_t)gguf_get_val_u32(gguf_ctx, idx) : def;
    };

    hparams.sample_rate     = get_i32("codec.sample_rate", hparams.sample_rate);
    hparams.num_codebooks   = get_i32("codec.num_codebooks", hparams.num_codebooks);
    hparams.codebook_size   = get_i32("codec.codebook_size", hparams.codebook_size);
    hparams.hop_length      = get_i32("codec.hop_length", hparams.hop_length);
    hparams.latent_dim      = get_i32("codec.latent_dim", hparams.latent_dim);
}

//
// Tensor mapping helpers
//

// Parse layer index from tensor name like "audio_decoder.res_layers.2.res_blocks.1.res_blocks.0..."
static bool parse_indices(const char * name, const char * prefix, int * idx1, int * idx2 = nullptr, int * idx3 = nullptr) {
    const char * p = strstr(name, prefix);
    if (!p) return false;
    p += strlen(prefix);

    if (idx1) {
        if (*p != '.') return false;
        *idx1 = atoi(++p);
        while (*p && *p != '.') p++;
    }
    if (idx2) {
        if (!strstr(p, ".res_blocks.")) return false;
        p = strstr(p, ".res_blocks.") + strlen(".res_blocks.");
        *idx2 = atoi(p);
        while (*p && *p != '.') p++;
    }
    if (idx3) {
        if (!strstr(p, ".res_blocks.")) return false;
        p = strstr(p, ".res_blocks.") + strlen(".res_blocks.");
        *idx3 = atoi(p);
    }
    return true;
}

static void map_codec_tensor(const char * name, ggml_tensor * t, magpie_codec & codec) {
    // Short tensor names format:
    // dec.pre.{weight,bias} - pre conv
    // dec.post.{weight,bias} - post conv
    // dec.post_act.alpha - post activation
    // dec.act.{i}.activation.snake_act.alpha - upsample activations
    // dec.up.{i}.c.{weight,bias} - upsample conv
    // dec.rl.{i}.rb.{j}.rb.{k}.{in_act,in_conv,sk_act,sk_conv}.{alpha,weight,bias}

    // Pre-conv
    if (strstr(name, "dec.pre.weight")) {
        codec.pre_conv_w = t;
        return;
    }
    if (strstr(name, "dec.pre.bias")) {
        codec.pre_conv_b = t;
        return;
    }

    // Post-conv
    if (strstr(name, "dec.post.weight")) {
        codec.post_conv_w = t;
        return;
    }
    if (strstr(name, "dec.post.bias")) {
        codec.post_conv_b = t;
        return;
    }
    if (strstr(name, "dec.post_act.alpha")) {
        codec.post_act_alpha = t;
        return;
    }

    // Upsample layers: dec.up.{i}
    if (strstr(name, "dec.up.")) {
        int i = -1;
        const char * p = strstr(name, "dec.up.") + strlen("dec.up.");
        i = atoi(p);
        if (i >= 0 && i < (int)codec.upsample_layers.size()) {
            if (strstr(name, ".weight")) {
                codec.upsample_layers[i].conv_w = t;
            } else if (strstr(name, ".bias")) {
                codec.upsample_layers[i].conv_b = t;
            }
        }
        return;
    }

    // Activations (before upsample): dec.act.{i}
    if (strstr(name, "dec.act.") && strstr(name, "alpha")) {
        int i = -1;
        const char * p = strstr(name, "dec.act.") + strlen("dec.act.");
        i = atoi(p);
        if (i >= 0 && i < (int)codec.upsample_layers.size()) {
            codec.upsample_layers[i].act_alpha = t;
        }
        return;
    }

    // Residual layers: dec.rl.{i}.rb.{j}.rb.{k}
    if (strstr(name, "dec.rl.")) {
        int i = -1, j = -1, k = -1;

        // Parse: rl.{i}
        const char * p = strstr(name, "dec.rl.") + strlen("dec.rl.");
        i = atoi(p);

        // Parse: rb.{j}
        p = strstr(p, ".rb.");
        if (!p) return;
        p += strlen(".rb.");
        j = atoi(p);

        // Parse: rb.{k}
        p = strstr(p, ".rb.");
        if (!p) return;
        p += strlen(".rb.");
        k = atoi(p);

        if (i < 0 || i >= (int)codec.res_layers.size()) return;
        if (j < 0 || j >= (int)codec.res_layers[i].res_blocks.size()) return;
        if (k < 0 || k >= (int)codec.res_layers[i].res_blocks[j].inner_blocks.size()) return;

        auto & block = codec.res_layers[i].res_blocks[j].inner_blocks[k];

        if (strstr(name, ".in_act.alpha")) {
            block.input_act_alpha = t;
        } else if (strstr(name, ".in_conv.weight")) {
            block.input_conv_w = t;
        } else if (strstr(name, ".in_conv.bias")) {
            block.input_conv_b = t;
        } else if (strstr(name, ".sk_act.alpha")) {
            block.skip_act_alpha = t;
        } else if (strstr(name, ".sk_conv.weight")) {
            block.skip_conv_w = t;
        } else if (strstr(name, ".sk_conv.bias")) {
            block.skip_conv_b = t;
        }
        return;
    }

    // FSQ parameters: vq.fsqs.{i}
    if (strstr(name, "vq.fsqs.")) {
        int i = -1;
        const char * p = strstr(name, "vq.fsqs.") + strlen("vq.fsqs.");
        i = atoi(p);
        if (i >= 0 && i < (int)codec.fsqs.size()) {
            if (strstr(name, ".dim_base_index")) {
                codec.fsqs[i].dim_base_index = t;
            } else if (strstr(name, ".num_levels")) {
                codec.fsqs[i].num_levels = t;
            }
        }
        return;
    }
}

//
// Model loading
//

bool magpie_codec_load(const std::string & path, magpie_codec & codec, magpie_backend_type backend) {
    fprintf(stderr, "magpie_codec: loading from %s\n", path.c_str());

    // Initialize backend
    if (!init_codec_backend(codec, backend)) {
        fprintf(stderr, "magpie_codec: failed to init backend\n");
        return false;
    }

    // Open GGUF file
    struct gguf_init_params gguf_params = {
        .no_alloc = true,
        .ctx = &codec.ctx_w,
    };

    struct gguf_context * gguf_ctx = gguf_init_from_file(path.c_str(), gguf_params);
    if (!gguf_ctx) {
        fprintf(stderr, "magpie_codec: failed to open %s\n", path.c_str());
        return false;
    }

    // Read hyperparameters
    read_codec_hparams(gguf_ctx, codec.hparams);

    const int n_tensors = gguf_get_n_tensors(gguf_ctx);
    fprintf(stderr, "magpie_codec: %d tensors\n", n_tensors);

    // Pre-allocate structures based on hparams
    const auto & hp = codec.hparams;

    // Upsample layers
    codec.upsample_layers.resize(hp.num_upsample_layers);

    // Residual layers: 5 layers, each with 3 res_blocks (kernels 3,7,11), each with 3 inner blocks (dilations 1,3,5)
    codec.res_layers.resize(hp.num_upsample_layers);
    for (int i = 0; i < hp.num_upsample_layers; i++) {
        codec.res_layers[i].res_blocks.resize(3);  // 3 kernel sizes
        for (int j = 0; j < 3; j++) {
            codec.res_layers[i].res_blocks[j].inner_blocks.resize(3);  // 3 dilations
        }
    }

    // FSQ codebooks
    codec.fsqs.resize(hp.num_codebooks);

    // Compute buffer size
    size_t ctx_size = 0;
    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        size_t offset = gguf_get_tensor_offset(gguf_ctx, i);
        struct ggml_tensor * t = ggml_get_tensor(codec.ctx_w, name);
        size_t size = ggml_nbytes(t);
        ctx_size += size;

        // Store tensor in map
        codec.tensors[name] = t;
    }

    // Allocate buffer on backend
    codec.buffer_w = ggml_backend_alloc_buffer(codec.backend, ctx_size + 1024 * 1024);  // +1MB padding
    if (!codec.buffer_w) {
        fprintf(stderr, "magpie_codec: failed to allocate buffer (%zu bytes)\n", ctx_size);
        gguf_free(gguf_ctx);
        return false;
    }

    // Create allocator and allocate tensors
    ggml_tallocr alloc = ggml_tallocr_new(codec.buffer_w);
    for (auto & pair : codec.tensors) {
        ggml_tallocr_alloc(&alloc, pair.second);
    }

    // Load tensor data from file
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "magpie_codec: failed to reopen file\n");
        gguf_free(gguf_ctx);
        return false;
    }

    // Get data offset in GGUF
    size_t data_offset = gguf_get_data_offset(gguf_ctx);

    // Load each tensor
    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        size_t tensor_offset = gguf_get_tensor_offset(gguf_ctx, i);
        struct ggml_tensor * t = codec.tensors[name];
        size_t size = ggml_nbytes(t);

        // Read from file
        std::vector<uint8_t> buf(size);
        fseek(f, data_offset + tensor_offset, SEEK_SET);
        if (fread(buf.data(), 1, size, f) != size) {
            fprintf(stderr, "magpie_codec: failed to read tensor %s\n", name);
            fclose(f);
            gguf_free(gguf_ctx);
            return false;
        }

        // Copy to backend
        ggml_backend_tensor_set(t, buf.data(), 0, size);

        // Map tensor to structure
        map_codec_tensor(name, t, codec);
    }

    fclose(f);
    gguf_free(gguf_ctx);

    // Verify critical tensors
    if (!codec.pre_conv_w || !codec.pre_conv_b) {
        fprintf(stderr, "magpie_codec: missing pre_conv tensors\n");
        return false;
    }
    if (!codec.post_conv_w || !codec.post_conv_b) {
        fprintf(stderr, "magpie_codec: missing post_conv tensors\n");
        return false;
    }

    fprintf(stderr, "magpie_codec: loaded successfully\n");
    fprintf(stderr, "  sample_rate: %d\n", hp.sample_rate);
    fprintf(stderr, "  num_codebooks: %d\n", hp.num_codebooks);
    fprintf(stderr, "  hop_length: %d\n", hp.hop_length);
    fprintf(stderr, "  backend: %s\n",
        codec.backend_type == MAGPIE_BACKEND_CUDA ? "CUDA" :
        codec.backend_type == MAGPIE_BACKEND_METAL ? "Metal" : "CPU");

    return true;
}

//
// Context management
//

struct magpie_codec * magpie_codec_init(const char * codec_path) {
    return magpie_codec_init_with_backend(codec_path, MAGPIE_BACKEND_AUTO);
}

struct magpie_codec * magpie_codec_init_with_backend(const char * codec_path, magpie_backend_type backend) {
    magpie_codec * codec = new magpie_codec();

    if (!magpie_codec_load(codec_path, *codec, backend)) {
        delete codec;
        return nullptr;
    }

    return codec;
}

void magpie_codec_free(struct magpie_codec * codec) {
    if (!codec) return;

    if (codec->buffer_w) {
        ggml_backend_buffer_free(codec->buffer_w);
    }
    if (codec->ctx_w) {
        ggml_free(codec->ctx_w);
    }
    if (codec->backend) {
        ggml_backend_free(codec->backend);
    }

    delete codec;
}

//
// Graph building functions
//

// HalfSnake: first half uses Snake, second half uses LeakyReLU
// Snake: x + (1/alpha) * sin²(alpha * x)
struct ggml_tensor * magpie_codec_build_half_snake(
    struct ggml_context * ctx,
    struct ggml_tensor * input,       // [T, channels] - ne[0]=T, ne[1]=channels
    struct ggml_tensor * alpha)       // [first_half_ch] - alpha for snake activation
{
    int64_t T = input->ne[0];
    int64_t channels = input->ne[1];

    // First half gets Snake activation, second half gets LeakyReLU
    // Alpha tensor size determines the split point
    int64_t first_half_ch = ggml_nelements(alpha);
    int64_t second_half_ch = channels - first_half_ch;

    // Split into first and second half of channels
    // For [T, C] layout, we split on ne[1] (channels dimension)
    // view_2d(src, ne0, ne1, nb1, offset)
    struct ggml_tensor * first_half = ggml_view_2d(ctx, input, T, first_half_ch, input->nb[1], 0);
    struct ggml_tensor * second_half = ggml_view_2d(ctx, input, T, second_half_ch, input->nb[1], first_half_ch * input->nb[1]);

    // Make views contiguous to ensure proper compute graph handling
    struct ggml_tensor * first_cont = ggml_cont(ctx, first_half);
    struct ggml_tensor * second_cont = ggml_cont(ctx, second_half);
    ggml_set_name(first_cont, "first_cont");
    ggml_set_name(second_cont, "second_cont");

    // Snake on first half: x + (1/alpha) * sin²(alpha * x)
    // alpha is [first_half_ch], reshape to [1, first_half_ch] for broadcast over T
    struct ggml_tensor * alpha_bc = ggml_reshape_2d(ctx, alpha, 1, first_half_ch);

    // alpha * x (alpha broadcasts over T dimension)
    struct ggml_tensor * ax = ggml_mul(ctx, first_cont, alpha_bc);
    // sin(alpha * x)
    struct ggml_tensor * sin_ax = ggml_sin(ctx, ax);
    // sin²(alpha * x)
    struct ggml_tensor * sin2_ax = ggml_sqr(ctx, sin_ax);
    // (1/alpha) * sin²
    struct ggml_tensor * scaled = ggml_div(ctx, sin2_ax, alpha_bc);
    // x + scaled
    struct ggml_tensor * snake_out = ggml_add(ctx, first_cont, scaled);

    // LeakyReLU on second half (slope = 0.01, PyTorch default)
    struct ggml_tensor * lrelu_out = ggml_leaky_relu(ctx, second_cont, 0.01f, false);
    ggml_set_name(lrelu_out, "lrelu_out");
    ggml_set_name(snake_out, "snake_out");

    // Concatenate on channel dimension (dim=1)
    struct ggml_tensor * output = ggml_concat(ctx, snake_out, lrelu_out, 1);
    ggml_set_name(output, "half_snake_out");

    return output;
}

// Causal Conv1D: left-pad by (kernel_size - 1) * dilation
struct ggml_tensor * magpie_codec_build_causal_conv1d(
    struct ggml_context * ctx,
    struct ggml_tensor * input,       // [T, in_ch] for ggml_conv_1d (ne[0]=T, ne[1]=IC)
    struct ggml_tensor * weight,      // [K, IC, OC] in GGML (ne[0]=K, ne[1]=IC, ne[2]=OC)
    struct ggml_tensor * bias,        // [OC]
    int dilation)
{
    // ggml_conv_1d expects:
    //   kernel a: [K, IC, OC] (ne[0]=K, ne[1]=IC, ne[2]=OC)
    //   data b:   [L, IC] (ne[0]=L, ne[1]=IC)
    // Output: [OL, OC] where OL = (L + 2*padding - dilation*(K-1) - 1) / stride + 1
    int kernel_size = weight->ne[0];
    int in_ch = weight->ne[1];
    int out_ch = weight->ne[2];
    int64_t T = input->ne[0];
    (void)in_ch;  // Used for documentation

    // Effective kernel size with dilation
    int eff_kernel = (kernel_size - 1) * dilation + 1;
    int pad_left = eff_kernel - 1;

    // Pad input on left (dimension 0): [T, IC] -> [T + pad_left, IC]
    struct ggml_tensor * padded = ggml_pad_ext(ctx, input, pad_left, 0, 0, 0, 0, 0, 0, 0);

    // ggml_conv_1d(ctx, kernel, data, stride, padding, dilation)
    // kernel = weight [K, IC, OC]
    // data = padded [L, IC]
    // Output will be [OL, OC]
    struct ggml_tensor * conv_out = ggml_conv_1d(ctx, weight, padded, 1, 0, dilation);

    // Add bias: [OC] broadcast to [OL, OC]
    if (bias) {
        // bias is [OC], need to broadcast to [OL, OC]
        // conv_out is [OL, OC], so bias needs to be [1, OC] then repeated
        struct ggml_tensor * bias_2d = ggml_reshape_2d(ctx, bias, 1, out_ch);
        struct ggml_tensor * bias_bc = ggml_repeat(ctx, bias_2d, conv_out);
        conv_out = ggml_add(ctx, conv_out, bias_bc);
    }

    return conv_out;
}

// Grouped ConvTranspose1D for upsampling
// NeMo nano-codec uses groups=out_ch with in_ch = 2*out_ch
// So each output channel is produced from 2 input channels.
// Input/output use [T, C] layout (ne[0]=T, ne[1]=C)
//
// Weight: [K, 1, in_ch] where in_ch = 2*out_ch
// For group g (0 to out_ch-1):
//   - Uses input channels [2g, 2g+1]
//   - Uses weight[:, 0, 2g:2g+2]
//   - Produces output channel g
//
// Since GGML doesn't support grouped conv_transpose_1d, we process each group
// by extracting the 2 input channels, applying conv_transpose, and concatenating.
struct ggml_tensor * magpie_codec_build_conv_transpose1d(
    struct ggml_context * ctx,
    struct ggml_tensor * input,       // [T, in_ch] - ne[0]=T, ne[1]=in_ch
    struct ggml_tensor * weight,      // [K, 1, in_ch] from GGUF
    struct ggml_tensor * bias,        // [out_ch]
    int stride)
{
    int64_t T = input->ne[0];
    int64_t in_ch = input->ne[1];

    // Weight is [K, 1, in_ch]
    int64_t K = weight->ne[0];
    int64_t w_in_ch = weight->ne[2];
    GGML_ASSERT(in_ch == w_in_ch);

    // For groups=out_ch with in_ch=2*out_ch: 2 input channels per output channel
    int64_t in_per_group = 2;
    int64_t out_ch = in_ch / in_per_group;

    // ConvTranspose1d output length: (T - 1) * stride + K
    int64_t out_T = (T - 1) * stride + K;

    // Trimming for causal: trim right by (kernel_size - stride)
    int64_t trim_right = K - stride;
    int64_t final_T = out_T - trim_right;  // Should be T * stride

    // Process each group (output channel)
    // For group g:
    //   - Extract input [T, 2] for channels [2g, 2g+1]
    //   - Extract weight [K, 1, 2] for those channels
    //   - Apply conv_transpose with IC=2, OC=1
    //   - Output is [final_T, 1]

    struct ggml_tensor * output = nullptr;

    for (int64_t g = 0; g < out_ch; g++) {
        // Extract 2 input channels for this group: [T, 2]
        struct ggml_tensor * in_g = ggml_view_2d(ctx, input, T, in_per_group,
                                                  input->nb[1],
                                                  g * in_per_group * input->nb[1]);
        in_g = ggml_cont(ctx, in_g);

        // Extract weight for this group: [K, 1, 2]
        // weight layout: ne[0]=K, ne[1]=1, ne[2]=in_ch
        struct ggml_tensor * w_g = ggml_view_3d(ctx, weight, K, 1, in_per_group,
                                                 weight->nb[1], weight->nb[2],
                                                 g * in_per_group * weight->nb[2]);
        w_g = ggml_cont(ctx, w_g);

        // ggml_conv_transpose_1d expects:
        //   kernel: [K, OC, IC] where ne[0]=K, ne[1]=OC, ne[2]=IC
        //   input:  [T, IC] where ne[0]=T, ne[1]=IC
        //   output: [T', OC]
        //
        // Our extracted w_g is [K, 1, 2] = [K, OC=1, IC=2]
        // This is already the correct format - no permute needed!
        //
        // Apply conv_transpose_1d: [T, 2] with kernel [K, 1, 2] -> [out_T, 1]
        struct ggml_tensor * out_g = ggml_conv_transpose_1d(ctx, w_g, in_g, stride, 0, 1);

        // Trim right side for causality: [out_T, 1] -> [final_T, 1]
        if (trim_right > 0) {
            out_g = ggml_view_2d(ctx, out_g, final_T, 1, out_g->nb[1], 0);
            out_g = ggml_cont(ctx, out_g);
        }

        // Concatenate groups: [final_T, 1] * out_ch -> [final_T, out_ch]
        if (output == nullptr) {
            output = out_g;
        } else {
            output = ggml_concat(ctx, output, out_g, 1);
        }
    }

    ggml_set_name(output, "conv_transpose_out");

    // Add bias: [out_ch] broadcast to [final_T, out_ch]
    if (bias && ggml_nelements(bias) > 0) {
        struct ggml_tensor * bias_2d = ggml_reshape_2d(ctx, bias, 1, out_ch);
        struct ggml_tensor * bias_bc = ggml_repeat(ctx, bias_2d, output);
        output = ggml_add(ctx, output, bias_bc);
    }

    return output;
}

// Single residual block
struct ggml_tensor * magpie_codec_build_residual_block(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    struct magpie_codec_resblock * block,
    int dilation)
{
    // Residual block:
    // h = half_snake(input)
    // h = conv1d(h, dilation)
    // h = half_snake(h)
    // h = conv1d(h, dilation=1)
    // return input + h

    struct ggml_tensor * h = input;

    // Input activation (HalfSnake)
    h = magpie_codec_build_half_snake(ctx, h, block->input_act_alpha);

    // Input conv (dilated)
    h = magpie_codec_build_causal_conv1d(ctx, h, block->input_conv_w, block->input_conv_b, dilation);

    // Skip activation (HalfSnake)
    h = magpie_codec_build_half_snake(ctx, h, block->skip_act_alpha);

    // Skip conv (non-dilated)
    h = magpie_codec_build_causal_conv1d(ctx, h, block->skip_conv_w, block->skip_conv_b, 1);

    // Residual connection
    struct ggml_tensor * output = ggml_add(ctx, input, h);

    return output;
}

// HiFiGAN residual block (3 inner blocks with dilations 1, 3, 5)
struct ggml_tensor * magpie_codec_build_hifigan_resblock(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    struct magpie_codec_hifigan_resblock * block)
{
    struct ggml_tensor * out = input;

    // Apply 3 inner blocks sequentially with dilations [1, 3, 5]
    const int dilations[3] = {1, 3, 5};
    for (int i = 0; i < 3; i++) {
        out = magpie_codec_build_residual_block(ctx, out, &block->inner_blocks[i], dilations[i]);
    }

    return out;
}

// HiFiGAN residual layer (3 blocks with kernels 3, 7, 11, averaged)
struct ggml_tensor * magpie_codec_build_reslayer(
    struct ggml_context * ctx,
    struct ggml_tensor * input,
    struct magpie_codec_reslayer * layer)
{
    // Run 3 parallel blocks and average
    struct ggml_tensor * sum = nullptr;

    for (int i = 0; i < 3; i++) {
        struct ggml_tensor * block_out = magpie_codec_build_hifigan_resblock(ctx, input, &layer->res_blocks[i]);

        if (sum == nullptr) {
            sum = block_out;
        } else {
            sum = ggml_add(ctx, sum, block_out);
        }
    }

    // Average by dividing by 3
    struct ggml_tensor * output = ggml_scale(ctx, sum, 1.0f / 3.0f);

    return output;
}

// FSQ dequantization: index -> continuous values
struct ggml_tensor * magpie_codec_build_fsq_dequant(
    struct ggml_context * ctx,
    struct ggml_tensor * codes,       // [num_codebooks, n_frames] int32
    struct magpie_codec * codec)
{
    // FSQ dequantization formula:
    // For each codebook i and each frame:
    //   index = codes[i, t]
    //   For each dimension d (4 dims per codebook):
    //     nonneg[d] = (index // dim_base_index[d]) % num_levels[d]
    //     code[d] = (nonneg[d] - num_levels[d]//2) / (num_levels[d]//2)
    //
    // dim_base_index = [1, 8, 56, 336]
    // num_levels = [8, 7, 6, 6]

    const int num_cb = codec->hparams.num_codebooks;
    const int dims_per_cb = 4;
    const int latent_dim = num_cb * dims_per_cb;  // 32
    int64_t n_frames = codes->ne[1];

    // This is complex to do in GGML graph directly because it involves integer division
    // For now, we'll handle FSQ dequantization at runtime in CPU before passing to graph

    // Create placeholder for dequantized latent
    struct ggml_tensor * latent = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, latent_dim, n_frames);
    ggml_set_name(latent, "fsq_latent");
    ggml_set_input(latent);  // Will be filled at runtime

    return latent;
}

// Full decoder graph
struct ggml_tensor * magpie_codec_build_decoder(
    struct ggml_context * ctx,
    struct ggml_tensor * latent,      // [T, latent_dim] GGML layout (ne[0]=T, ne[1]=C)
    struct magpie_codec * codec)
{
    struct ggml_tensor * out = latent;

    // Pre-conv: [T, latent_dim] -> [T, base_ch]
    out = magpie_codec_build_causal_conv1d(ctx, out, codec->pre_conv_w, codec->pre_conv_b, 1);

    // Upsample stages
    for (int i = 0; i < codec->hparams.num_upsample_layers; i++) {
        auto & up = codec->upsample_layers[i];
        auto & res = codec->res_layers[i];
        int stride = codec->hparams.up_sample_rates[i];

        // Activation (HalfSnake)
        out = magpie_codec_build_half_snake(ctx, out, up.act_alpha);

        // Upsample ConvTranspose
        out = magpie_codec_build_conv_transpose1d(ctx, out, up.conv_w, up.conv_b, stride);

        // Residual layer
        out = magpie_codec_build_reslayer(ctx, out, &res);
    }

    // Post activation
    out = magpie_codec_build_half_snake(ctx, out, codec->post_act_alpha);

    // Post conv: [T, final_ch] -> [T, 1]
    out = magpie_codec_build_causal_conv1d(ctx, out, codec->post_conv_w, codec->post_conv_b, 1);

    // Tanh output activation
    out = ggml_tanh(ctx, out);

    // Squeeze channel dimension: [T, 1] -> [T]
    out = ggml_reshape_1d(ctx, out, out->ne[0]);

    return out;
}

//
// FSQ dequantization (CPU runtime function)
//

static void fsq_dequantize_cpu(
    const int32_t * codes,      // [num_cb, n_frames]
    float * latent,             // [n_frames, latent_dim] GGML layout (ne[0]=T, ne[1]=C)
    int num_cb,
    int n_frames)
{
    // FSQ parameters (hardcoded for now, matches nano-codec)
    const int dims_per_cb = 4;
    const int dim_base_index[4] = {1, 8, 56, 336};
    const int num_levels[4] = {8, 7, 6, 6};

    for (int cb = 0; cb < num_cb; cb++) {
        for (int t = 0; t < n_frames; t++) {
            int index = codes[cb * n_frames + t];

            for (int d = 0; d < dims_per_cb; d++) {
                // nonneg = (index // dim_base_index[d]) % num_levels[d]
                int nonneg = (index / dim_base_index[d]) % num_levels[d];

                // code = (nonneg - num_levels[d]//2) / (num_levels[d]//2)
                int half_levels = num_levels[d] / 2;
                float code = (float)(nonneg - half_levels) / (float)half_levels;

                // Store in GGML layout: ne[0]=T varies fastest
                // Element at (t, c) is at index: t + c * n_frames
                int c = cb * dims_per_cb + d;
                int latent_idx = t + c * n_frames;
                latent[latent_idx] = code;
            }
        }
    }
}

//
// Main decode function
//

std::vector<float> magpie_codec_decode(
    struct magpie_codec * codec,
    const int32_t * codes,      // [num_codebooks, n_frames]
    int n_frames)
{
    if (!codec) {
        fprintf(stderr, "magpie_codec_decode: null codec\n");
        return {};
    }

    const auto & hp = codec->hparams;
    const int num_cb = hp.num_codebooks;
    const int latent_dim = hp.latent_dim;

    // Compute output length
    int64_t out_samples = n_frames * hp.hop_length;

    // Step 1: FSQ dequantization (CPU)
    std::vector<float> latent_data(latent_dim * n_frames);
    fsq_dequantize_cpu(codes, latent_data.data(), num_cb, n_frames);

    // Step 2: Build compute graph
    // Need large memory for the extensive per-channel operations
    struct ggml_init_params params = {
        .mem_size   = 1024 * 1024 * 1024,  // 1 GB for graph
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    if (!ctx0) {
        fprintf(stderr, "magpie_codec_decode: failed to init ggml context\n");
        return {};
    }

    // Create input tensor
    // GGML conv_1d expects: ne[0]=seq_length(T), ne[1]=in_channels(C)
    struct ggml_tensor * latent = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_frames, latent_dim);
    ggml_set_name(latent, "latent_input");
    ggml_set_input(latent);

    // Build decoder graph
    struct ggml_tensor * audio = magpie_codec_build_decoder(ctx0, latent, codec);
    ggml_set_name(audio, "audio_output");
    ggml_set_output(audio);

    // Create compute graph with large capacity
    // The graph is very large due to per-channel conv_transpose operations
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 131072, false);
    ggml_build_forward_expand(gf, audio);

    // Allocate tensors
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(codec->backend));
    if (!ggml_gallocr_reserve(allocr, gf)) {
        fprintf(stderr, "magpie_codec_decode: failed to reserve graph\n");
        ggml_gallocr_free(allocr);
        ggml_free(ctx0);
        return {};
    }

    if (!ggml_gallocr_alloc_graph(allocr, gf)) {
        fprintf(stderr, "magpie_codec_decode: failed to alloc graph\n");
        ggml_gallocr_free(allocr);
        ggml_free(ctx0);
        return {};
    }

    // Set input data
    ggml_backend_tensor_set(latent, latent_data.data(), 0, latent_data.size() * sizeof(float));

    // Compute
    if (ggml_backend_graph_compute(codec->backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "magpie_codec_decode: graph compute failed\n");
        ggml_gallocr_free(allocr);
        ggml_free(ctx0);
        return {};
    }

    // Get output
    std::vector<float> output(out_samples);
    ggml_backend_tensor_get(audio, output.data(), 0, out_samples * sizeof(float));

    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(ctx0);

    return output;
}
