#include "magpie.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

// Read binary reference file (format: 4 x int64 shape + float32 data)
static bool read_reference(const char * path, std::vector<float> & data, int64_t shape[4]) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open: %s\n", path);
        return false;
    }

    // Read shape (4 x int64)
    if (fread(shape, sizeof(int64_t), 4, f) != 4) {
        fprintf(stderr, "Failed to read shape from: %s\n", path);
        fclose(f);
        return false;
    }

    // Calculate total elements
    int64_t n_elements = shape[0] * shape[1] * shape[2] * shape[3];
    data.resize(n_elements);

    // Read data
    if (fread(data.data(), sizeof(float), n_elements, f) != (size_t)n_elements) {
        fprintf(stderr, "Failed to read data from: %s\n", path);
        fclose(f);
        return false;
    }

    fclose(f);
    return true;
}

// Compare tensors with tolerance
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

int main(int argc, char ** argv) {
    const char * model_path = "weights/magpie-357m-f32.gguf";
    const char * input_path = "test_data/reference/manual_enc_with_pos.bin";
    const char * expected_path = "test_data/reference/hook_encoder_layers_0_norm_self.bin";

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

    // Check encoder layer 0 norm weights are loaded
    if (!ctx->model.encoder.layers[0].norm_self_w) {
        fprintf(stderr, "encoder.layers[0].norm_self_w not loaded!\n");
        magpie_free(ctx);
        return 1;
    }

    ggml_tensor * norm_w = ctx->model.encoder.layers[0].norm_self_w;
    fprintf(stderr, "norm_self_w shape: [%lld]\n", (long long)norm_w->ne[0]);

    // Read input (text embedding + position embedding)
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
    const float eps = ctx->model.hparams.eps;
    // Input shape from file: [1, 768, 14, 1] where:
    // - shape[1] = 768 = d_model
    // - shape[2] = 14 = seq_len
    // PyTorch tensor was [batch=1, seq=14, d_model=768], reversed and padded
    const int seq_len = (int)input_shape[2];

    fprintf(stderr, "d_model=%d, seq_len=%d, eps=%g\n", d_model, seq_len, eps);

    // Build computation graph
    size_t ctx_size = ggml_tensor_overhead() * 20 + 1024 * 1024;
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

    // Build RMS norm graph
    struct ggml_tensor * normed = magpie_build_rms_norm(ctx0, input_tensor, norm_w, eps);

    if (!normed) {
        fprintf(stderr, "magpie_build_rms_norm returned null!\n");
        ggml_free(ctx0);
        magpie_free(ctx);
        return 1;
    }

    ggml_set_name(normed, "normed");
    ggml_set_output(normed);

    fprintf(stderr, "Normed tensor shape: [%lld, %lld]\n",
            (long long)normed->ne[0], (long long)normed->ne[1]);

    // Build graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, normed);

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

    // Run computation
    fprintf(stderr, "Running computation...\n");
    if (ggml_backend_graph_compute(ctx->model.backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "Graph computation failed\n");
        ggml_free(ctx0);
        magpie_free(ctx);
        return 1;
    }

    // Get results
    std::vector<float> result(d_model * seq_len);
    ggml_backend_tensor_get(normed, result.data(), 0, result.size() * sizeof(float));

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

    // Compare - use relaxed tolerance for floating-point precision differences
    // RMS norm can have small differences between GGML and PyTorch implementations
    fprintf(stderr, "\nComparing results:\n");
    bool match = compare_tensors(result.data(), expected.data(), result.size(), 1e-2f, 1e-2f);

    ggml_free(ctx0);
    magpie_free(ctx);

    if (match) {
        fprintf(stderr, "\nSUCCESS: RMS norm matches reference!\n");
        return 0;
    } else {
        fprintf(stderr, "\nFAILURE: RMS norm does not match reference\n");
        return 1;
    }
}
