#include "magpie.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

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

int main() {
    const char * model_path = "weights/magpie-357m-f32.gguf";
    const char * input_path = "test_data/reference/debug_norm_ff_input.bin";
    const char * expected_path = "test_data/reference/debug_ffn_output.bin";

    fprintf(stderr, "=== Testing Conv FFN ===\n");
    fprintf(stderr, "Loading model...\n");
    magpie_context * ctx = magpie_init(model_path);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    magpie_encoder_layer * layer = &ctx->model.encoder.layers[0];

    fprintf(stderr, "FFN weights:\n");
    fprintf(stderr, "  ff_proj_w: [%lld, %lld, %lld]\n",
            (long long)layer->ff_proj_w->ne[0],
            (long long)layer->ff_proj_w->ne[1],
            (long long)layer->ff_proj_w->ne[2]);
    fprintf(stderr, "  ff_out_w: [%lld, %lld, %lld]\n",
            (long long)layer->ff_out_w->ne[0],
            (long long)layer->ff_out_w->ne[1],
            (long long)layer->ff_out_w->ne[2]);

    // Read input
    std::vector<float> input_data;
    int64_t input_shape[4];
    if (!read_reference(input_path, input_data, input_shape)) {
        magpie_free(ctx);
        return 1;
    }
    fprintf(stderr, "Input shape: [%lld, %lld, %lld, %lld]\n",
            (long long)input_shape[0], (long long)input_shape[1],
            (long long)input_shape[2], (long long)input_shape[3]);

    // Read expected
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
    const int kernel_size = 3;

    fprintf(stderr, "d_model=%d, seq_len=%d, kernel_size=%d\n", d_model, seq_len, kernel_size);

    // Print first few input values
    fprintf(stderr, "Input first 8: ");
    for (int i = 0; i < 8; i++) fprintf(stderr, "%.4f ", input_data[i]);
    fprintf(stderr, "\n");

    // Build graph
    size_t ctx_size = ggml_tensor_overhead() * 100 + 256 * 1024 * 1024;  // More memory for views
    struct ggml_init_params params = {
        .mem_size   = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, seq_len);
    ggml_set_name(input, "input");
    ggml_set_input(input);

    struct ggml_tensor * output = magpie_build_conv_ffn(
        ctx0, input, layer->ff_proj_w, layer->ff_out_w, kernel_size);

    if (!output) {
        fprintf(stderr, "magpie_build_conv_ffn returned null!\n");
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

    fprintf(stderr, "Graph nodes: %d\n", ggml_graph_n_nodes(gf));

    ggml_gallocr_reserve(ctx->state.allocr, gf);
    ggml_gallocr_alloc_graph(ctx->state.allocr, gf);

    // Set input data
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

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
        fprintf(stderr, "\nSUCCESS: Conv FFN matches!\n");
        return 0;
    } else {
        fprintf(stderr, "\nFAILURE: Conv FFN does not match\n");
        return 1;
    }
}
