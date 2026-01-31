// Test just the LayerNorm step against PyTorch reference
#include "../src/magpie.h"
#include <cstdio>
#include <cmath>
#include <vector>

static std::vector<float> load_reference(const char * path, int64_t * shape) {
    FILE * f = fopen(path, "rb");
    if (!f) return {};

    int64_t dims[4];
    if (fread(dims, sizeof(int64_t), 4, f) != 4) {
        fclose(f);
        return {};
    }
    for (int i = 0; i < 4; i++) shape[i] = dims[i];

    size_t n = 1;
    for (int i = 0; i < 4; i++) if (dims[i] > 0) n *= dims[i];

    std::vector<float> data(n);
    if (fread(data.data(), sizeof(float), n, f) != n) {
        fclose(f);
        return {};
    }
    fclose(f);
    return data;
}

int main() {
    printf("Loading model...\n");
    magpie_context * mctx = magpie_init("weights/magpie-357m-f32.gguf");
    if (!mctx) return 1;

    const auto & hp = mctx->model.hparams;

    // Load input (enc_input)
    int64_t in_shape[4];
    auto input_data = load_reference("test_data/reference/enc_input.bin", in_shape);

    // Load expected norm output
    int64_t norm_shape[4];
    auto expected_norm = load_reference("test_data/reference/enc_l0_norm1.bin", norm_shape);

    // Shape is [d_model, seq, 1, 1] in GGML format
    int64_t d_model = in_shape[0];
    int64_t seq_len = in_shape[1];

    printf("d_model=%lld, seq_len=%lld\n", (long long)d_model, (long long)seq_len);

    // Build compute graph: just norm
    size_t ctx_size = ggml_tensor_overhead() * 16 + 1024 * 1024;
    struct ggml_init_params params = {
        .mem_size = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    struct ggml_context * ctx0 = ggml_init(params);

    struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, seq_len);
    ggml_set_name(input, "input");
    ggml_set_input(input);

    auto & layer = mctx->model.encoder.layers[0];
    struct ggml_tensor * norm_out = magpie_build_layer_norm(ctx0, input, layer.norm_self_w, hp.eps);
    ggml_set_name(norm_out, "norm_out");
    ggml_set_output(norm_out);

    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, norm_out);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mctx->model.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(input, input_data.data(), 0, d_model * seq_len * sizeof(float));
    ggml_backend_graph_compute(mctx->model.backend, gf);

    std::vector<float> result(d_model * seq_len);
    ggml_backend_tensor_get(norm_out, result.data(), 0, result.size() * sizeof(float));

    // Compare
    float max_diff = 0;
    for (size_t i = 0; i < result.size(); i++) {
        float diff = fabsf(result[i] - expected_norm[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("Norm comparison:\n");
    printf("  GGML first5:  %.6f %.6f %.6f %.6f %.6f\n",
           result[0], result[1], result[2], result[3], result[4]);
    printf("  PyTorch first5: %.6f %.6f %.6f %.6f %.6f\n",
           expected_norm[0], expected_norm[1], expected_norm[2], expected_norm[3], expected_norm[4]);
    printf("  Max diff: %f\n", max_diff);

    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    magpie_free(mctx);

    if (max_diff < 0.001f) {
        printf("SUCCESS: Norm matches!\n");
        return 0;
    } else {
        printf("FAILURE: Norm mismatch\n");
        return 1;
    }
}
