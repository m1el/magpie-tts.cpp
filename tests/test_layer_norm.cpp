// Test LayerNorm implementation against PyTorch reference
#include "../src/magpie.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

struct tensor_header {
    int64_t dims[4];
};

static std::vector<float> load_reference(const char * path, int64_t * shape) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", path);
        return {};
    }

    tensor_header header;
    fread(&header, sizeof(header), 1, f);

    // Copy shape
    for (int i = 0; i < 4; i++) {
        shape[i] = header.dims[i];
    }

    // Calculate total elements (in GGML order: ne[0] is innermost)
    size_t n_elements = 1;
    for (int i = 0; i < 4; i++) {
        if (header.dims[i] > 0) {
            n_elements *= header.dims[i];
        }
    }

    std::vector<float> data(n_elements);
    fread(data.data(), sizeof(float), n_elements, f);
    fclose(f);

    return data;
}

int main(int argc, char ** argv) {
    const char * model_path = "weights/magpie-357m-f32.gguf";
    if (argc > 1) {
        model_path = argv[1];
    }

    printf("Loading model from %s\n", model_path);
    magpie_context * mctx = magpie_init(model_path);
    if (!mctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    printf("Model loaded successfully\n");
    const auto & hp = mctx->model.hparams;
    const float eps = hp.eps;

    // Load reference data: layer 0 output (to use as input to layer 1)
    // and layer 1 norm output
    int64_t shape0[4];
    std::vector<float> input_data = load_reference("test_data/reference/manual_enc_layer0_out.bin", shape0);
    if (input_data.empty()) {
        fprintf(stderr, "Failed to load layer 0 output reference\n");
        magpie_free(mctx);
        return 1;
    }

    int64_t shape_norm[4];
    std::vector<float> ref_norm = load_reference("test_data/reference/debug_l1_norm.bin", shape_norm);
    if (ref_norm.empty()) {
        fprintf(stderr, "Failed to load norm reference\n");
        magpie_free(mctx);
        return 1;
    }

    // Reference files store shape as [batch, d_model, seq, 1]
    // GGML tensors are [d_model, seq]
    printf("Input file shape: [%lld, %lld, %lld, %lld]\n",
           (long long)shape0[0], (long long)shape0[1], (long long)shape0[2], (long long)shape0[3]);
    printf("Norm ref file shape: [%lld, %lld, %lld, %lld]\n",
           (long long)shape_norm[0], (long long)shape_norm[1], (long long)shape_norm[2], (long long)shape_norm[3]);

    // Extract actual dimensions (skip batch dim)
    int64_t d_model = shape0[1];  // d_model is at index 1
    int64_t seq_len = shape0[2];  // seq is at index 2
    printf("Using: d_model=%lld, seq_len=%lld\n", (long long)d_model, (long long)seq_len);

    // Create compute context
    size_t ctx_size = ggml_tensor_overhead() * 16 + 1024 * 1024;
    struct ggml_init_params params = {
        .mem_size   = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context * ctx0 = ggml_init(params);

    // Build graph: just layer norm
    struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, seq_len);
    ggml_set_name(input, "input");
    ggml_set_input(input);

    // Get layer 1 norm weight
    auto & layer = mctx->model.encoder.layers[1];
    struct ggml_tensor * norm_out = magpie_build_layer_norm(ctx0, input, layer.norm_self_w, eps);
    ggml_set_name(norm_out, "norm_out");
    ggml_set_output(norm_out);

    // Build graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, norm_out);

    // Allocate
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mctx->model.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    // Set input data
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    // Compute
    ggml_backend_graph_compute(mctx->model.backend, gf);

    // Get output
    std::vector<float> output(d_model * seq_len);
    ggml_backend_tensor_get(norm_out, output.data(), 0, output.size() * sizeof(float));

    // Compare with reference
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    int max_idx = 0;
    for (size_t i = 0; i < output.size(); i++) {
        float diff = fabsf(output[i] - ref_norm[i]);
        sum_diff += diff;
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
    }
    float avg_diff = sum_diff / output.size();

    printf("\nLayerNorm comparison with PyTorch reference:\n");
    printf("  Max diff: %f at index %d (ggml=%f, ref=%f)\n",
           max_diff, max_idx, output[max_idx], ref_norm[max_idx]);
    printf("  Avg diff: %f\n", avg_diff);

    // Also compare RMSNorm to show the difference
    struct ggml_tensor * rms_out = magpie_build_rms_norm(ctx0, input, layer.norm_self_w, eps);
    ggml_set_name(rms_out, "rms_out");
    ggml_set_output(rms_out);

    struct ggml_cgraph * gf2 = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf2, rms_out);
    ggml_gallocr_reserve(allocr, gf2);
    ggml_gallocr_alloc_graph(allocr, gf2);
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_graph_compute(mctx->model.backend, gf2);

    std::vector<float> rms_output(d_model * seq_len);
    ggml_backend_tensor_get(rms_out, rms_output.data(), 0, rms_output.size() * sizeof(float));

    float rms_max_diff = 0.0f;
    float rms_sum_diff = 0.0f;
    for (size_t i = 0; i < rms_output.size(); i++) {
        float diff = fabsf(rms_output[i] - ref_norm[i]);
        rms_sum_diff += diff;
        if (diff > rms_max_diff) {
            rms_max_diff = diff;
        }
    }

    printf("\nRMSNorm comparison (old implementation):\n");
    printf("  Max diff: %f\n", rms_max_diff);
    printf("  Avg diff: %f\n", rms_sum_diff / output.size());

    printf("\n");
    if (max_diff < 0.001f) {
        printf("SUCCESS: LayerNorm matches PyTorch reference!\n");
    } else if (max_diff < 0.01f) {
        printf("GOOD: LayerNorm close to reference (acceptable numerical error)\n");
    } else {
        printf("WARNING: LayerNorm has significant difference from reference\n");
    }

    // Cleanup
    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    magpie_free(mctx);

    return (max_diff < 0.01f) ? 0 : 1;
}
