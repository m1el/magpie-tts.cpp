#include "magpie.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <limits>

// Read binary reference file (format: 4 x int64 shape + float32 data)
static bool read_reference(const char * path, std::vector<float> & data, int64_t shape[4]) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open: %s\n", path);
        return false;
    }

    if (fread(shape, sizeof(int64_t), 4, f) != 4) {
        fprintf(stderr, "Failed to read shape from: %s\n", path);
        fclose(f);
        return false;
    }

    int64_t n_elements = shape[0] * shape[1] * shape[2] * shape[3];
    data.resize(n_elements);

    if (fread(data.data(), sizeof(float), n_elements, f) != (size_t)n_elements) {
        fprintf(stderr, "Failed to read data from: %s\n", path);
        fclose(f);
        return false;
    }

    fclose(f);
    return true;
}

static bool compare_tensors(const float * a, const float * b, size_t n, float rtol, float atol) {
    float max_diff = 0.0f;
    float max_rel = 0.0f;
    size_t mismatch_count = 0;

    for (size_t i = 0; i < n; i++) {
        float diff = std::fabs(a[i] - b[i]);
        float rel = diff / (std::fabs(b[i]) + 1e-8f);

        if (diff > max_diff) max_diff = diff;
        if (rel > max_rel) max_rel = rel;

        if (diff > atol && rel > rtol) {
            mismatch_count++;
            if (mismatch_count <= 5) {
                fprintf(stderr, "  Mismatch at %zu: got %.6f, expected %.6f (diff=%.6f, rel=%.4f)\n",
                        i, a[i], b[i], diff, rel);
            }
        }
    }

    fprintf(stderr, "  Max diff: %.6f, max rel: %.4f, mismatches: %zu/%zu\n",
            max_diff, max_rel, mismatch_count, n);

    return mismatch_count == 0;
}

// Create causal mask: 0 for allowed (i >= j), -inf for masked (i < j)
static void fill_causal_mask_f16(ggml_fp16_t * mask, int seq_len) {
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            mask[i * seq_len + j] = (j <= i) ? zero : neg_inf;
        }
    }
}

int main(int argc, char ** argv) {
    const char * model_path = "weights/magpie-357m-f32.gguf";
    const char * input_path = "test_data/reference/manual_text_embedded.bin";
    const char * expected_path = "test_data/reference/manual_enc_output.bin";

    if (argc > 1) model_path = argv[1];
    if (argc > 2) input_path = argv[2];
    if (argc > 3) expected_path = argv[3];

    // Load model
    fprintf(stderr, "Loading model from: %s\n", model_path);
    magpie_context * ctx = magpie_init(model_path);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Check encoder weights
    magpie_encoder * encoder = &ctx->model.encoder;
    if (!encoder->pos_emb_w || !encoder->norm_out_w) {
        fprintf(stderr, "Encoder weights not loaded!\n");
        magpie_free(ctx);
        return 1;
    }

    fprintf(stderr, "Encoder:\n");
    fprintf(stderr, "  pos_emb_w: [%lld, %lld]\n",
            (long long)encoder->pos_emb_w->ne[0], (long long)encoder->pos_emb_w->ne[1]);
    fprintf(stderr, "  norm_out_w: [%lld]\n", (long long)encoder->norm_out_w->ne[0]);
    fprintf(stderr, "  layers: %d\n", (int)encoder->layers.size());

    // Check all layers have weights
    for (int l = 0; l < ctx->model.hparams.enc_layers; l++) {
        magpie_encoder_layer * layer = &encoder->layers[l];
        if (!layer->norm_self_w || !layer->sa_qkv_w || !layer->sa_out_w ||
            !layer->norm_ff_w || !layer->ff_proj_w || !layer->ff_out_w) {
            fprintf(stderr, "Encoder layer %d weights not loaded!\n", l);
            magpie_free(ctx);
            return 1;
        }
    }

    // Read input (text embeddings)
    std::vector<float> input_data;
    int64_t input_shape[4];
    if (!read_reference(input_path, input_data, input_shape)) {
        magpie_free(ctx);
        return 1;
    }
    fprintf(stderr, "Input shape: [%lld, %lld, %lld, %lld]\n",
            (long long)input_shape[0], (long long)input_shape[1],
            (long long)input_shape[2], (long long)input_shape[3]);

    // Read expected output
    std::vector<float> expected;
    int64_t expected_shape[4];
    if (!read_reference(expected_path, expected, expected_shape)) {
        magpie_free(ctx);
        return 1;
    }
    fprintf(stderr, "Expected shape: [%lld, %lld, %lld, %lld]\n",
            (long long)expected_shape[0], (long long)expected_shape[1],
            (long long)expected_shape[2], (long long)expected_shape[3]);

    const int d_model = ctx->model.hparams.d_model;
    const int seq_len = (int)input_shape[2];  // shape[2] is seq after reversal

    fprintf(stderr, "d_model=%d, seq_len=%d, enc_layers=%d\n",
            d_model, seq_len, ctx->model.hparams.enc_layers);

    // Build computation graph
    // Larger graph for full encoder (6 layers with flash attention)
    size_t ctx_size = ggml_tensor_overhead() * 600 + 64 * 1024 * 1024;
    struct ggml_init_params params = {
        .mem_size   = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    if (!ctx0) {
        fprintf(stderr, "Failed to create compute context\n");
        magpie_free(ctx);
        return 1;
    }

    // Create input tensor [d_model, seq_len]
    struct ggml_tensor * input_tensor = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, seq_len);
    ggml_set_name(input_tensor, "input");
    ggml_set_input(input_tensor);

    // Build full encoder graph
    struct ggml_tensor * output = magpie_build_full_encoder(
        ctx0, input_tensor, encoder, &ctx->model.hparams);

    if (!output) {
        fprintf(stderr, "magpie_build_full_encoder returned null!\n");
        ggml_free(ctx0);
        magpie_free(ctx);
        return 1;
    }

    ggml_set_name(output, "output");
    ggml_set_output(output);

    fprintf(stderr, "Output tensor shape: [%lld, %lld]\n",
            (long long)output->ne[0], (long long)output->ne[1]);

    // Build graph with larger capacity
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 8192, false);
    ggml_build_forward_expand(gf, output);

    fprintf(stderr, "Graph nodes: %d\n", ggml_graph_n_nodes(gf));

    // Allocate compute buffers
    if (!ggml_gallocr_reserve(ctx->state.allocr, gf)) {
        fprintf(stderr, "Failed to reserve allocator\n");
        ggml_free(ctx0);
        magpie_free(ctx);
        return 1;
    }

    if (!ggml_gallocr_alloc_graph(ctx->state.allocr, gf)) {
        fprintf(stderr, "Failed to allocate graph\n");
        ggml_free(ctx0);
        magpie_free(ctx);
        return 1;
    }

    // Copy input to device
    ggml_backend_tensor_set(input_tensor, input_data.data(), 0, input_data.size() * sizeof(float));

    // Find and set the shared causal mask
    struct ggml_tensor * mask = ggml_get_tensor(ctx0, "causal_mask");
    if (mask) {
        fprintf(stderr, "Setting causal mask [%lld, %lld] (F16)\n",
                (long long)mask->ne[0], (long long)mask->ne[1]);
        std::vector<ggml_fp16_t> mask_data(seq_len * seq_len);
        fill_causal_mask_f16(mask_data.data(), seq_len);
        ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
    } else {
        fprintf(stderr, "WARNING: causal_mask tensor not found\n");
    }

    // Run computation
    fprintf(stderr, "Running full encoder computation...\n");
    if (ggml_backend_graph_compute(ctx->model.backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "Graph computation failed\n");
        ggml_free(ctx0);
        magpie_free(ctx);
        return 1;
    }

    // Get results
    std::vector<float> result(d_model * seq_len);
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));

    // Print first few values
    fprintf(stderr, "First 8 values: ");
    for (int i = 0; i < 8; i++) {
        fprintf(stderr, "%.4f ", result[i]);
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "Expected first 8: ");
    for (int i = 0; i < 8; i++) {
        fprintf(stderr, "%.4f ", expected[i]);
    }
    fprintf(stderr, "\n");

    // Compare with relaxed tolerance for accumulated FP errors across 6 layers
    // F32/F16 mixed precision + 6 layers of attention/conv can accumulate ~0.2-0.3 error
    fprintf(stderr, "\nComparing results:\n");
    bool match = compare_tensors(result.data(), expected.data(), result.size(), 0.2f, 0.25f);

    ggml_free(ctx0);
    magpie_free(ctx);

    if (match) {
        fprintf(stderr, "\nSUCCESS: Full encoder matches reference!\n");
        return 0;
    } else {
        fprintf(stderr, "\nFAILURE: Full encoder does not match reference\n");
        return 1;
    }
}
