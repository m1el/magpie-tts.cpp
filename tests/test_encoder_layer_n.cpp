#include "magpie.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

static bool read_reference(const char * path, std::vector<float> & data, int64_t shape[4]) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Failed to open: %s\n", path); return false; }
    if (fread(shape, sizeof(int64_t), 4, f) != 4) { fclose(f); return false; }
    int64_t n = shape[0] * shape[1] * shape[2] * shape[3];
    data.resize(n);
    if (fread(data.data(), sizeof(float), n, f) != (size_t)n) { fclose(f); return false; }
    fclose(f);
    return true;
}

static void fill_causal_mask_f16(ggml_fp16_t * mask, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            mask[i*n + j] = (j <= i) ? ggml_fp32_to_fp16(0.0f) : ggml_fp32_to_fp16(-INFINITY);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <layer_idx> [model_path]\n", argv[0]);
        return 1;
    }

    int layer_idx = atoi(argv[1]);
    const char * model_path = (argc > 2) ? argv[2] : "weights/magpie-357m-f32.gguf";

    char input_path[256], expected_path[256];
    if (layer_idx == 0) {
        snprintf(input_path, sizeof(input_path), "test_data/reference/manual_enc_with_pos.bin");
    } else {
        snprintf(input_path, sizeof(input_path), "test_data/reference/manual_enc_layer%d_out.bin", layer_idx - 1);
    }
    snprintf(expected_path, sizeof(expected_path), "test_data/reference/manual_enc_layer%d_out.bin", layer_idx);

    fprintf(stderr, "Testing encoder layer %d\n", layer_idx);
    fprintf(stderr, "  Input: %s\n", input_path);
    fprintf(stderr, "  Expected: %s\n", expected_path);

    magpie_context * ctx = magpie_init(model_path);
    if (!ctx) { fprintf(stderr, "Failed to load model\n"); return 1; }

    if (layer_idx >= ctx->model.hparams.enc_layers) {
        fprintf(stderr, "Layer %d out of range (max %d)\n", layer_idx, ctx->model.hparams.enc_layers - 1);
        magpie_free(ctx);
        return 1;
    }

    magpie_encoder_layer * layer = &ctx->model.encoder.layers[layer_idx];

    std::vector<float> input_data, expected;
    int64_t input_shape[4], expected_shape[4];
    if (!read_reference(input_path, input_data, input_shape)) { magpie_free(ctx); return 1; }
    if (!read_reference(expected_path, expected, expected_shape)) { magpie_free(ctx); return 1; }

    const int d_model = ctx->model.hparams.d_model;
    const int seq_len = (int)input_shape[2];

    fprintf(stderr, "d_model=%d, seq_len=%d\n", d_model, seq_len);

    size_t ctx_size = ggml_tensor_overhead() * 100 + 32 * 1024 * 1024;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx0 = ggml_init(params);

    struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, seq_len);
    ggml_set_name(input, "input"); ggml_set_input(input);

    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, seq_len, seq_len);
    ggml_set_name(mask, "causal_mask"); ggml_set_input(mask);

    struct ggml_tensor * output = magpie_build_encoder_layer_with_mask(
        ctx0, input, nullptr, layer, &ctx->model.hparams, mask);
    ggml_set_name(output, "output"); ggml_set_output(output);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 2048, false);
    ggml_build_forward_expand(gf, output);

    ggml_gallocr_reserve(ctx->state.allocr, gf);
    ggml_gallocr_alloc_graph(ctx->state.allocr, gf);

    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    std::vector<ggml_fp16_t> mask_data(seq_len * seq_len);
    fill_causal_mask_f16(mask_data.data(), seq_len);
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));

    ggml_backend_graph_compute(ctx->model.backend, gf);

    std::vector<float> result(d_model * seq_len);
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));

    fprintf(stderr, "First 5 values: ");
    for (int i = 0; i < 5; i++) fprintf(stderr, "%.4f ", result[i]);
    fprintf(stderr, "\nExpected first 5: ");
    for (int i = 0; i < 5; i++) fprintf(stderr, "%.4f ", expected[i]);
    fprintf(stderr, "\n");

    float max_diff = 0;
    size_t max_idx = 0, mismatch = 0;
    for (size_t i = 0; i < result.size(); i++) {
        float diff = std::fabs(result[i] - expected[i]);
        if (diff > max_diff) { max_diff = diff; max_idx = i; }
        if (diff > 0.05f) mismatch++;
    }

    fprintf(stderr, "Max diff: %.6f at idx %zu (got=%.6f, exp=%.6f)\n",
            max_diff, max_idx, result[max_idx], expected[max_idx]);
    fprintf(stderr, "Mismatches (>0.05): %zu/%zu\n", mismatch, result.size());

    ggml_free(ctx0);
    magpie_free(ctx);

    return (max_diff < 0.1f) ? 0 : 1;
}
