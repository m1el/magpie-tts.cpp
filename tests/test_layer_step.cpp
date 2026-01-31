// Test full encoder layer against PyTorch reference
#include "../src/magpie.h"
#include <cstdio>
#include <cmath>
#include <vector>

static std::vector<float> load_reference(const char * path, int64_t * shape) {
    FILE * f = fopen(path, "rb");
    if (!f) return {};
    int64_t dims[4];
    if (fread(dims, sizeof(int64_t), 4, f) != 4) { fclose(f); return {}; }
    for (int i = 0; i < 4; i++) shape[i] = dims[i];
    size_t n = 1;
    for (int i = 0; i < 4; i++) if (dims[i] > 0) n *= dims[i];
    std::vector<float> data(n);
    if (fread(data.data(), sizeof(float), n, f) != n) { fclose(f); return {}; }
    fclose(f);
    return data;
}

static void fill_causal_mask_f32(float * mask, int seq_len) {
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            mask[i * seq_len + j] = (j <= i) ? 0.0f : -INFINITY;
        }
    }
}

int main() {
    printf("Loading model...\n");
    magpie_context * mctx = magpie_init("weights/magpie-357m-f32.gguf");
    if (!mctx) return 1;

    const auto & hp = mctx->model.hparams;

    // Load input (encoder input with pos embeddings)
    int64_t in_shape[4];
    auto input_data = load_reference("test_data/reference/enc_input.bin", in_shape);

    // Load expected layer 0 output
    int64_t out_shape[4];
    auto expected = load_reference("test_data/reference/enc_l0_out.bin", out_shape);

    int64_t d_model = in_shape[0];
    int64_t seq_len = in_shape[1];
    printf("d_model=%lld, seq_len=%lld\n", (long long)d_model, (long long)seq_len);

    // Build graph
    size_t ctx_size = ggml_tensor_overhead() * 128 + 32 * 1024 * 1024;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx0 = ggml_init(params);

    struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, seq_len);
    ggml_set_name(input, "input");
    ggml_set_input(input);

    auto & layer = mctx->model.encoder.layers[0];
    struct ggml_tensor * output = magpie_build_encoder_layer(ctx0, input, nullptr, &layer, &hp);
    ggml_set_name(output, "output");
    ggml_set_output(output);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 4096, false);
    ggml_build_forward_expand(gf, output);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mctx->model.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(input, input_data.data(), 0, d_model * seq_len * sizeof(float));

    // Set causal mask (F32)
    struct ggml_tensor * mask = ggml_get_tensor(ctx0, "causal_mask");
    if (mask) {
        std::vector<float> mask_data(seq_len * seq_len);
        fill_causal_mask_f32(mask_data.data(), seq_len);
        ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(float));
    }

    ggml_backend_graph_compute(mctx->model.backend, gf);

    std::vector<float> result(d_model * seq_len);
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));

    // Compare
    float max_diff = 0;
    int max_idx = 0;
    for (size_t i = 0; i < result.size(); i++) {
        float diff = fabsf(result[i] - expected[i]);
        if (diff > max_diff) { max_diff = diff; max_idx = i; }
    }

    printf("Encoder Layer 0 comparison:\n");
    printf("  GGML first5:    %.6f %.6f %.6f %.6f %.6f\n",
           result[0], result[1], result[2], result[3], result[4]);
    printf("  PyTorch first5: %.6f %.6f %.6f %.6f %.6f\n",
           expected[0], expected[1], expected[2], expected[3], expected[4]);
    printf("  Max diff: %f at idx %d\n", max_diff, max_idx);

    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    magpie_free(mctx);

    if (max_diff < 0.05f) {
        printf("SUCCESS: Encoder layer matches!\n");
        return 0;
    } else {
        printf("FAILURE: Encoder layer mismatch\n");
        return 1;
    }
}
