// Test decoder layer (single layer) against PyTorch reference
#include "../src/magpie.h"
#include <cstdio>
#include <cmath>
#include <vector>

static std::vector<float> load_reference(const char * path, int64_t * shape) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", path);
        return {};
    }
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

static void compare_tensors(const char* name, const float* ggml_data, const float* ref_data, size_t n) {
    float max_diff = 0;
    int max_idx = 0;
    float sum_diff = 0;
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(ggml_data[i] - ref_data[i]);
        sum_diff += diff;
        if (diff > max_diff) { max_diff = diff; max_idx = i; }
    }
    printf("%s: max_diff=%.6f at idx %d, avg_diff=%.6f\n",
           name, max_diff, max_idx, sum_diff / n);
    printf("  GGML first5:    %.6f %.6f %.6f %.6f %.6f\n",
           ggml_data[0], ggml_data[1], ggml_data[2], ggml_data[3], ggml_data[4]);
    printf("  PyTorch first5: %.6f %.6f %.6f %.6f %.6f\n",
           ref_data[0], ref_data[1], ref_data[2], ref_data[3], ref_data[4]);
}

int main() {
    printf("Loading model...\n");
    magpie_context * mctx = magpie_init("weights/magpie-357m-f32.gguf");
    if (!mctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const auto & hp = mctx->model.hparams;

    // Load reference data
    int64_t dec_input_shape[4], enc_out_shape[4], l0_out_shape[4];
    auto dec_input_data = load_reference("test_data/reference/dec_input.bin", dec_input_shape);
    auto enc_out_data = load_reference("test_data/reference/dec_encoder_output.bin", enc_out_shape);
    auto l0_out_expected = load_reference("test_data/reference/dec_l0_out.bin", l0_out_shape);

    if (dec_input_data.empty() || enc_out_data.empty() || l0_out_expected.empty()) {
        fprintf(stderr, "Failed to load reference data. Run: uv run scripts/dump_decoder_reference.py\n");
        magpie_free(mctx);
        return 1;
    }

    int64_t d_model = dec_input_shape[0];
    int64_t dec_seq = dec_input_shape[1];
    int64_t enc_seq = enc_out_shape[1];

    printf("d_model=%lld, dec_seq=%lld, enc_seq=%lld\n",
           (long long)d_model, (long long)dec_seq, (long long)enc_seq);

    // Build graph
    size_t ctx_size = ggml_tensor_overhead() * 512 + 128 * 1024 * 1024;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx0 = ggml_init(params);

    // Create input tensors
    struct ggml_tensor * dec_input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, dec_seq);
    ggml_set_name(dec_input, "dec_input");
    ggml_set_input(dec_input);

    struct ggml_tensor * enc_out = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, enc_seq);
    ggml_set_name(enc_out, "enc_out");
    ggml_set_input(enc_out);

    // Build decoder layer 0
    struct ggml_tensor * output = magpie_build_decoder_layer(
        ctx0, dec_input, enc_out, 0,
        &mctx->model.decoder.layers[0], &mctx->state.kv_cache, &hp);

    if (!output) {
        fprintf(stderr, "Failed to build decoder layer\n");
        ggml_free(ctx0);
        magpie_free(mctx);
        return 1;
    }

    ggml_set_name(output, "output");
    ggml_set_output(output);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 8192, false);
    ggml_build_forward_expand(gf, output);
    printf("Graph nodes: %d\n", ggml_graph_n_nodes(gf));

    // Allocate and run
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(mctx->model.backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    // Set input data
    ggml_backend_tensor_set(dec_input, dec_input_data.data(), 0, d_model * dec_seq * sizeof(float));
    ggml_backend_tensor_set(enc_out, enc_out_data.data(), 0, d_model * enc_seq * sizeof(float));

    // Set causal mask for self-attention
    struct ggml_tensor * sa_mask = ggml_get_tensor(ctx0, "dec_sa_mask");
    if (sa_mask) {
        printf("Setting causal mask [%lld, %lld]\n", (long long)sa_mask->ne[0], (long long)sa_mask->ne[1]);
        std::vector<float> mask_data(dec_seq * dec_seq);
        fill_causal_mask_f32(mask_data.data(), dec_seq);
        ggml_backend_tensor_set(sa_mask, mask_data.data(), 0, mask_data.size() * sizeof(float));
    }

    printf("Computing...\n");
    ggml_backend_graph_compute(mctx->model.backend, gf);

    // Get result
    std::vector<float> result(d_model * dec_seq);
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));

    // Compare
    printf("\n=== Decoder Layer 0 Comparison ===\n");
    compare_tensors("Layer 0 output", result.data(), l0_out_expected.data(), result.size());

    // Calculate overall accuracy
    float max_diff = 0;
    for (size_t i = 0; i < result.size(); i++) {
        float diff = fabsf(result[i] - l0_out_expected[i]);
        if (diff > max_diff) max_diff = diff;
    }

    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    magpie_free(mctx);

    if (max_diff < 0.1f) {
        printf("\nSUCCESS: Decoder layer 0 matches within tolerance (max_diff=%.6f)!\n", max_diff);
        return 0;
    } else {
        printf("\nFAILURE: Decoder layer 0 has large errors (max_diff=%.6f)\n", max_diff);
        return 1;
    }
}
