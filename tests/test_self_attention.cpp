#include "magpie.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <limits>

static bool read_reference(const char * path, std::vector<float> & data, int64_t shape[4]) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open: %s\n", path);
        return false;
    }

    if (fread(shape, sizeof(int64_t), 4, f) != 4) {
        fclose(f);
        return false;
    }

    int64_t n_elements = shape[0] * shape[1] * shape[2] * shape[3];
    data.resize(n_elements);

    if (fread(data.data(), sizeof(float), n_elements, f) != (size_t)n_elements) {
        fclose(f);
        return false;
    }

    fclose(f);
    return true;
}

static bool compare_tensors(const float * a, const float * b, size_t n, float rtol, float atol) {
    float max_diff = 0.0f;
    size_t mismatch_count = 0;

    for (size_t i = 0; i < n; i++) {
        float diff = std::fabs(a[i] - b[i]);
        float rel = diff / (std::fabs(b[i]) + 1e-8f);

        if (diff > max_diff) max_diff = diff;

        if (diff > atol && rel > rtol) {
            mismatch_count++;
            if (mismatch_count <= 5) {
                fprintf(stderr, "  Mismatch at %zu: got %.6f, expected %.6f (diff=%.6f)\n",
                        i, a[i], b[i], diff);
            }
        }
    }

    fprintf(stderr, "  Max diff: %.6f, mismatches: %zu/%zu\n", max_diff, mismatch_count, n);
    return mismatch_count == 0;
}

// Create causal mask: 0 for allowed (i >= j), -inf for masked (i < j)
// Uses ggml_fp16_t (uint16_t) for CUDA compatibility
static void fill_causal_mask_f16(ggml_fp16_t * mask, int seq_len) {
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            // Position i can attend to position j if j <= i (causal)
            mask[i * seq_len + j] = (j <= i) ? zero : neg_inf;
        }
    }
}

int main() {
    const char * model_path = "weights/magpie-357m-f32.gguf";

    // Test self-attention: input is norm output, expected is attention output
    const char * input_path = "test_data/reference/hook_encoder_layers_0_norm_self.bin";
    const char * expected_path = "test_data/reference/hook_encoder_layers_0_self_attention.bin";

    fprintf(stderr, "=== Testing Self-Attention (Causal) ===\n");
    fprintf(stderr, "Loading model...\n");
    magpie_context * ctx = magpie_init(model_path);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    magpie_encoder_layer * layer = &ctx->model.encoder.layers[0];

    // Read input (norm output)
    std::vector<float> input_data;
    int64_t input_shape[4];
    if (!read_reference(input_path, input_data, input_shape)) {
        magpie_free(ctx);
        return 1;
    }
    fprintf(stderr, "Input shape: [%lld, %lld, %lld, %lld]\n",
            (long long)input_shape[0], (long long)input_shape[1],
            (long long)input_shape[2], (long long)input_shape[3]);

    // Read expected (attention output)
    std::vector<float> expected;
    int64_t expected_shape[4];
    if (!read_reference(expected_path, expected, expected_shape)) {
        magpie_free(ctx);
        return 1;
    }
    fprintf(stderr, "Expected shape: [%lld, %lld, %lld, %lld]\n",
            (long long)expected_shape[0], (long long)expected_shape[1],
            (long long)expected_shape[2], (long long)expected_shape[3]);

    const int d_model = 768;
    const int seq_len = (int)input_shape[2];

    // Build graph
    size_t ctx_size = ggml_tensor_overhead() * 100 + 16 * 1024 * 1024;
    struct ggml_init_params params = {
        .mem_size   = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, seq_len);
    ggml_set_name(input, "input");
    ggml_set_input(input);

    // Self-attention with causal masking (encoder uses causal attention in this model)
    struct ggml_tensor * output = magpie_build_self_attention(
        ctx0, input, layer->sa_qkv_w, layer->sa_out_w,
        ctx->model.hparams.enc_heads, true);  // is_causal = true

    if (!output) {
        fprintf(stderr, "magpie_build_self_attention returned null!\n");
        ggml_free(ctx0);
        magpie_free(ctx);
        return 1;
    }

    ggml_set_name(output, "output");
    ggml_set_output(output);

    fprintf(stderr, "Output shape: [%lld, %lld]\n",
            (long long)output->ne[0], (long long)output->ne[1]);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 4096, false);
    ggml_build_forward_expand(gf, output);

    ggml_gallocr_reserve(ctx->state.allocr, gf);
    ggml_gallocr_alloc_graph(ctx->state.allocr, gf);

    // Set input data
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    // Find and set the causal mask tensor (F16 for CUDA compatibility)
    struct ggml_tensor * mask = ggml_get_tensor(ctx0, "causal_mask");
    if (mask) {
        fprintf(stderr, "Setting causal mask [%lld, %lld] (F16)\n",
                (long long)mask->ne[0], (long long)mask->ne[1]);
        std::vector<ggml_fp16_t> mask_data(seq_len * seq_len);
        fill_causal_mask_f16(mask_data.data(), seq_len);
        ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
    } else {
        fprintf(stderr, "Warning: causal_mask tensor not found\n");
    }

    fprintf(stderr, "Running computation...\n");
    ggml_backend_graph_compute(ctx->model.backend, gf);

    std::vector<float> result(d_model * seq_len);
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));

    fprintf(stderr, "First 8 values: ");
    for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", result[i]);
    fprintf(stderr, "\n");

    fprintf(stderr, "Expected first 8: ");
    for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", expected[i]);
    fprintf(stderr, "\n");

    fprintf(stderr, "\nComparing:\n");
    bool match = compare_tensors(result.data(), expected.data(), result.size(), 0.05f, 0.05f);

    ggml_free(ctx0);
    magpie_free(ctx);

    if (match) {
        fprintf(stderr, "\nSUCCESS: Self-attention matches!\n");
        return 0;
    } else {
        fprintf(stderr, "\nFAILURE: Self-attention does not match\n");
        return 1;
    }
}
